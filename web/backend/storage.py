"""Save/load models to/from web_models/."""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

WEB_MODELS_DIR = Path(__file__).resolve().parent.parent / "web_models"
HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]


def list_models() -> list[dict]:
    WEB_MODELS_DIR.mkdir(exist_ok=True)
    models = []
    for model_dir in sorted(WEB_MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["model_id"] = model_dir.name
        models.append(meta)
    return models


def _slugify(name: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def create_model(name: str, nodes: list[dict], edges: list[dict], source: str) -> str:
    """Persist a validated graph as a new saved model with status 'untrained'."""
    WEB_MODELS_DIR.mkdir(exist_ok=True)
    base_slug = _slugify(name)
    model_id = base_slug
    suffix = 1
    while (WEB_MODELS_DIR / model_id).exists():
        suffix += 1
        model_id = f"{base_slug}-{suffix}"

    model_dir = WEB_MODELS_DIR / model_id
    model_dir.mkdir()

    with open(model_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    meta = {
        "name": name,
        "source": source,
        "status": "untrained",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model_id


def model_exists(model_id: str) -> bool:
    return (WEB_MODELS_DIR / model_id / "meta.json").exists()


def load_graph(model_id: str) -> dict:
    with open(WEB_MODELS_DIR / model_id / "graph.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(model_id: str) -> dict:
    with open(WEB_MODELS_DIR / model_id / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def update_status(model_id: str, status: str, **extra) -> None:
    meta_path = WEB_MODELS_DIR / model_id / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["status"] = status
    meta.update(extra)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def init_history(model_id: str) -> None:
    path = WEB_MODELS_DIR / model_id / "history.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()


def append_history(model_id: str, row: dict) -> None:
    path = WEB_MODELS_DIR / model_id / "history.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writerow({k: row.get(k) for k in HISTORY_FIELDS})


def save_checkpoint(model_id: str, state_dict, nodes: list[dict], edges: list[dict], epoch: int, val_f1: float, val_acc: float) -> None:
    path = WEB_MODELS_DIR / model_id / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": state_dict,
            "nodes": nodes,
            "edges": edges,
            "epoch": epoch,
            "val_f1": val_f1,
            "val_acc": val_acc,
        },
        path,
    )


def load_checkpoint(model_id: str, map_location=None) -> dict:
    return torch.load(WEB_MODELS_DIR / model_id / "checkpoint.pt", map_location=map_location)


def delete_model(model_id: str) -> None:
    shutil.rmtree(WEB_MODELS_DIR / model_id)


def rename_model(model_id: str, new_name: str) -> None:
    meta = load_meta(model_id)
    meta["name"] = new_name
    with open(WEB_MODELS_DIR / model_id / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
