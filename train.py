import argparse
import dataclasses
import os
import time

import torch
import torch.nn as nn

from config import Config
from data import get_dataloaders
from engine import evaluate, train_one_epoch
from model import MLP
from utils import EarlyStopping, MetricLogger, get_device, save_checkpoint, set_seed


def parse_args(cfg: Config) -> Config:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST.")
    for f in dataclasses.fields(cfg):
        if f.name == "hidden_sizes":
            parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
            continue
        arg_name = "--" + f.name.replace("_", "-")
        arg_type = type(getattr(cfg, f.name))
        if arg_type is bool:
            parser.add_argument(arg_name, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            parser.add_argument(arg_name, type=arg_type, default=None)

    args = parser.parse_args()
    overrides = {k.replace("-", "_"): v for k, v in vars(args).items() if v is not None}
    if "hidden_sizes" in overrides:
        pass  # already correctly named via dest
    return dataclasses.replace(cfg, **overrides)


def main():
    cfg = Config()
    cfg = parse_args(cfg)

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        num_workers=cfg.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model_kwargs = dict(
        input_dim=784,
        hidden_sizes=cfg.hidden_sizes,
        num_classes=10,
        dropout=cfg.dropout,
        use_batchnorm=cfg.use_batchnorm,
        activation=cfg.activation,
    )
    model = MLP(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    early_stopping = EarlyStopping(patience=cfg.patience, mode="max")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    logger = MetricLogger(os.path.join(cfg.log_dir, "history.csv"))

    best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    last_path = os.path.join(cfg.checkpoint_dir, "last_model.pt")

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)
        val_loss, val_acc, val_f1 = val_result["loss"], val_result["accuracy"], val_result["macro_f1"]

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            lr=current_lr,
        )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"| lr={current_lr:.2e} | {dt:.1f}s"
        )

        should_stop = early_stopping.step(val_f1)

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "model_kwargs": model_kwargs,
            "config": dataclasses.asdict(cfg),
        }
        save_checkpoint(state, last_path)
        if early_stopping.is_best:
            save_checkpoint(state, best_path)
            print(f"  -> new best (val_f1={val_f1:.4f}), saved to {best_path}")

        if should_stop:
            print(f"Early stopping triggered after epoch {epoch} (patience={cfg.patience}).")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
