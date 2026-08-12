from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data as data_module
from . import graph, model_builder, presets, storage, train_job

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="MNIST Model Builder & Tester")


class NodeIn(BaseModel):
    id: str
    type: str
    config: dict = Field(default_factory=dict)


class EdgeIn(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class CreateModelRequest(BaseModel):
    name: str
    nodes: list[NodeIn]
    edges: list[EdgeIn]
    source: str = "custom"


@app.get("/api/presets")
def get_presets():
    return presets.PRESETS


@app.get("/api/models")
def get_models():
    return storage.list_models()


@app.post("/api/models", status_code=201)
def create_model(body: CreateModelRequest):
    nodes = [n.model_dump() for n in body.nodes]
    edges = [{"from": e.from_, "to": e.to} for e in body.edges]
    try:
        graph.validate_and_infer(nodes, edges)
    except graph.GraphValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": exc.message, "node_id": exc.node_id})

    model_id = storage.create_model(name=body.name, nodes=nodes, edges=edges, source=body.source)
    return {"model_id": model_id}


class TrainRequest(BaseModel):
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 20
    optimizer: str = "Adam"
    early_stopping_patience: int = 5


@app.post("/api/models/{model_id}/train")
async def train_model(model_id: str, body: TrainRequest = TrainRequest()):
    if not storage.model_exists(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    try:
        job_id = await train_job.start_training(model_id, body.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"job_id": job_id}


class RenameRequest(BaseModel):
    name: str


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    if not storage.model_exists(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    if train_job.is_training() and train_job.state.model_id == model_id:
        raise HTTPException(status_code=409, detail=f"Model '{model_id}' is currently training and cannot be deleted.")
    storage.delete_model(model_id)
    return {"deleted": model_id}


@app.patch("/api/models/{model_id}")
def rename_model(model_id: str, body: RenameRequest):
    if not storage.model_exists(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name cannot be empty.")
    storage.rename_model(model_id, name)
    return {"model_id": model_id, "name": name}


@app.websocket("/ws/train/{job_id}")
async def ws_train(websocket: WebSocket, job_id: str):
    await websocket.accept()
    train_job.state.clients.add(websocket)
    if train_job.state.job_id == job_id:
        await websocket.send_json(train_job.state.snapshot())
    else:
        await websocket.send_json({"status": "unknown_job", "job_id": job_id})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        train_job.state.clients.discard(websocket)


@app.get("/api/tester/sample")
def tester_sample():
    sample_ids = data_module.sample_one_per_digit()
    return [{"sample_id": sid, "pixels": data_module.render_digit_png_base64(sid)} for sid in sample_ids]


class PredictRequest(BaseModel):
    model_id: str
    sample_id: int


@app.post("/api/tester/predict")
def tester_predict(body: PredictRequest):
    if not storage.model_exists(body.model_id):
        raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not found.")
    meta = storage.load_meta(body.model_id)
    if meta.get("status") != "trained":
        raise HTTPException(status_code=400, detail=f"Model '{body.model_id}' is not trained yet.")
    if not (0 <= body.sample_id < len(data_module.get_test_dataset())):
        raise HTTPException(status_code=404, detail=f"sample_id {body.sample_id} is out of range.")

    checkpoint = storage.load_checkpoint(body.model_id, map_location="cpu")
    model = model_builder.GraphModel(checkpoint["nodes"], checkpoint["edges"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = data_module.get_model_input_tensor(body.sample_id).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        predicted = int(logits.argmax(dim=1).item())

    true_label = data_module.get_true_label(body.sample_id)
    return {"predicted_digit": predicted, "true_digit": true_label, "correct": predicted == true_label}


# NOTE: further API routes (/api/*) are added here in later phases. They
# must be registered before the StaticFiles mount below, since that mount
# is a catch-all for "/".

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
