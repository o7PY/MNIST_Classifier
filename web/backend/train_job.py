"""Background training job runner + WebSocket progress broadcaster.

The actual PyTorch training loop is synchronous/blocking (GPU or CPU bound),
so it runs in a plain OS thread rather than an asyncio task - otherwise it
would freeze the single-threaded asyncio event loop (and with it every other
request, including the WebSocket sends) for the whole duration of an epoch.
The training thread pushes progress back onto the event loop via
asyncio.run_coroutine_threadsafe, which is the supported way to schedule a
coroutine from a non-event-loop thread.

Only one training job may run at a time (webspec.md's concurrency
constraint), enforced by `state.status`.
"""

import asyncio
import threading
from typing import Optional

import torch
import torch.nn as nn
from fastapi import WebSocket
from sklearn.metrics import f1_score

from . import data as data_module
from . import model_builder, storage

MAX_EPOCHS_CAP = 100


class TrainingState:
    def __init__(self):
        self.job_id: Optional[str] = None
        self.model_id: Optional[str] = None
        self.status: str = "idle"  # idle | training | done | failed
        self.history: list[dict] = []
        self.error: Optional[str] = None
        self.best_val_f1: Optional[float] = None
        self.clients: set[WebSocket] = set()

    def reset(self, job_id: str, model_id: str) -> None:
        self.job_id = job_id
        self.model_id = model_id
        self.status = "training"
        self.history = []
        self.error = None
        self.best_val_f1 = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "history": self.history,
            "latest": self.history[-1] if self.history else None,
            "best_val_f1": self.best_val_f1,
            "error": self.error,
        }


state = TrainingState()


def is_training() -> bool:
    return state.status == "training"


async def broadcast(message: dict) -> None:
    dead = []
    for ws in list(state.clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.clients.discard(ws)


async def start_training(model_id: str, hyperparams: dict) -> str:
    if is_training():
        raise RuntimeError(f"Training already in progress for model '{state.model_id}'.")

    job_id = model_id
    state.reset(job_id, model_id)
    storage.update_status(model_id, "training")
    storage.init_history(model_id)

    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=_run_training_sync, args=(model_id, hyperparams, loop), daemon=True)
    thread.start()
    return job_id


def _run_training_sync(model_id: str, hyperparams: dict, loop: asyncio.AbstractEventLoop) -> None:
    try:
        graph_data = storage.load_graph(model_id)
        nodes, edges = graph_data["nodes"], graph_data["edges"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_builder.GraphModel(nodes, edges).to(device)

        lr = float(hyperparams.get("learning_rate", 1e-3))
        batch_size = int(hyperparams.get("batch_size", 128))
        max_epochs = min(int(hyperparams.get("max_epochs", 20)), MAX_EPOCHS_CAP)
        optimizer_name = hyperparams.get("optimizer", "Adam")
        patience = int(hyperparams.get("early_stopping_patience", 5))

        train_loader, val_loader = data_module.get_train_val_loaders(batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_f1 = -1.0
        best_acc = 0.0
        bad_epochs = 0

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
            state.history.append(row)
            storage.append_history(model_id, row)
            asyncio.run_coroutine_threadsafe(broadcast({"status": "training", "model_id": model_id, "latest": row}), loop)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                bad_epochs = 0
                storage.save_checkpoint(model_id, model.state_dict(), nodes, edges, epoch, val_f1, val_acc)
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                break

        state.status = "done"
        state.best_val_f1 = best_f1
        storage.update_status(model_id, "trained", val_f1=best_f1, val_acc=best_acc)
        asyncio.run_coroutine_threadsafe(
            broadcast({"status": "done", "model_id": model_id, "best_val_f1": best_f1, "best_val_acc": best_acc}), loop
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client instead of dying silently
        state.status = "failed"
        state.error = str(exc)
        storage.update_status(model_id, "failed")
        asyncio.run_coroutine_threadsafe(broadcast({"status": "failed", "model_id": model_id, "error": str(exc)}), loop)


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    return total_loss / total, correct / total, val_f1
