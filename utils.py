import csv
import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn("CUDA not available, falling back to CPU.")
    return torch.device("cpu")


class EarlyStopping:
    """Tracks whether a monitored metric has stopped improving.

    mode="max" -> higher metric is better (e.g. accuracy, F1).
    mode="min" -> lower metric is better (e.g. loss).
    """

    def __init__(self, patience: int = 7, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > self.best
        return metric < self.best

    def step(self, metric: float) -> bool:
        """Update state with the latest metric.

        Returns True if training should stop.
        """
        if self._is_improvement(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    @property
    def is_best(self) -> bool:
        return self.num_bad_epochs == 0


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location=None) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class MetricLogger:
    """Accumulates per-epoch metric rows and flushes them incrementally to a CSV file."""

    FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"]

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()

    def log(self, **row) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)
