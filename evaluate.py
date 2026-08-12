import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from data import get_test_loader
from engine import evaluate
from model import MLP
from utils import get_device


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — MNIST Test Set")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=8)

    ax.spines[:].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confused_pairs(cm: np.ndarray, k: int = 5):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    return pairs[:k]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MLP on the MNIST test set.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = Config()
    device = get_device()
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MLP(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')})")

    test_loader = get_test_loader(batch_size=cfg.batch_size, data_dir=cfg.data_dir, num_workers=cfg.num_workers)

    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, test_loader, criterion, device, return_predictions=True)
    accuracy = result["accuracy"]
    macro_f1 = result["macro_f1"]
    y_true, y_pred = result["y_true"], result["y_pred"]

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(report)

    os.makedirs(cfg.report_dir, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(cfg.report_dir, "confusion_matrix.png"))
    with open(os.path.join(cfg.report_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Test macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    passed = accuracy >= 0.98 or macro_f1 > 0.90
    verdict = "PASS" if passed else "FAIL"
    print(f"Acceptance criteria (accuracy >= 0.98 OR macro_f1 > 0.90): {verdict}")

    confused = top_confused_pairs(cm, k=5)
    confused_lines = [f"- true={t} predicted={p}: {c} cases" for c, t, p in confused]

    summary_path = "results_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# MNIST MLP — Results Summary\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}` (epoch {checkpoint.get('epoch')})\n")
        f.write(f"- Test accuracy: **{accuracy:.4f}**\n")
        f.write(f"- Test macro F1: **{macro_f1:.4f}**\n")
        f.write(f"- Acceptance criteria (accuracy >= 0.98 OR macro F1 > 0.90): **{verdict}**\n\n")
        f.write("## Most confused digit pairs\n")
        f.write("\n".join(confused_lines) + "\n\n")
        f.write("## Artifacts\n")
        f.write("- `outputs/reports/confusion_matrix.png`\n")
        f.write("- `outputs/reports/classification_report.txt`\n")
        f.write("- `outputs/logs/history.csv`\n")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
