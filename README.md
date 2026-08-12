# MNIST Digit Classification

Two related projects in this repo:

1. **CLI MLP classifier** (project root) — a PyTorch dense-only MLP
   (no convolutions) trained on MNIST via the command line. See
   `results_summary.md` for the latest results.
2. **Web App** (`web/`) — a local FastAPI + vanilla-JS app for visually
   building neural network architectures (including CNNs and branching
   graphs), training them on MNIST with live progress, and testing them
   interactively against handwritten digits.

---

## 1. CLI MLP Classifier

### Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA is recommended (falls back to CPU automatically,
  but training will be much slower).

Install dependencies:

```bash
pip install -r requirements.txt
```

### Project layout

```
config.py     Config dataclass — all hyperparameters and paths
data.py       MNIST download, transforms, train/val/test dataloaders
model.py      MLP (nn.Linear stack) definition
engine.py     train_one_epoch / evaluate — shared by train.py and evaluate.py
utils.py      seeding, device selection, early stopping, checkpoint I/O, CSV logger
train.py      training entrypoint
evaluate.py   test-set evaluation entrypoint (metrics, confusion matrix, report)
```

Running either entrypoint the first time automatically downloads MNIST into
`data/` (gitignored — regenerated on first run).

### Train

```bash
python train.py
```

This trains the MLP (default: `784 -> 512 -> 256 -> 128 -> 10`) with early
stopping on validation macro-F1 (patience 7, up to 50 epochs). It prints the
device in use (expect `cuda`) and, per epoch, train/val loss, accuracy, and
macro-F1.

Outputs:
- `outputs/checkpoints/best_model.pt` — best checkpoint by validation macro-F1
- `outputs/checkpoints/last_model.pt` — most recent epoch's checkpoint
- `outputs/logs/history.csv` — per-epoch metrics

#### Overriding hyperparameters

Any field in `Config` (`config.py`) can be overridden via CLI flag, e.g.:

```bash
python train.py --lr 5e-4 --hidden-sizes 1024 512 256 --dropout 0.4 --max-epochs 30
```

Run `python train.py --help` for the full list of flags.

### Evaluate

After training, evaluate the best checkpoint on the held-out test set:

```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

(`--checkpoint` defaults to `outputs/checkpoints/best_model.pt`, so it can be
omitted if you're using the default path.)

This prints test accuracy, macro-F1, and a per-class classification report,
and reports PASS/FAIL against the acceptance criteria (accuracy ≥ 98% OR
macro F1 > 90%). It also writes:

- `outputs/reports/confusion_matrix.png`
- `outputs/reports/classification_report.txt`
- `results_summary.md` — final metrics, PASS/FAIL verdict, and the most
  confused digit pairs

### Reproducibility

All scripts seed `random`, `numpy`, and `torch`/`torch.cuda` from
`Config.seed` (default 42), and the train/val split uses a seeded generator,
so reruns with the same config produce near-identical results.

---

## 2. Web App

An interactive builder/trainer/tester for MNIST models, served locally.

### Requirements

- Everything from the CLI project's `requirements.txt` (PyTorch etc.), plus
  the web app's own dependencies.

Install both:

```bash
pip install -r requirements.txt
pip install -r web/requirements.txt
```

### Run it

```bash
cd web
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open **http://127.0.0.1:8000** in a browser. It reuses the MNIST data
downloaded by the CLI project (`../data/`) instead of re-downloading it, so
run `python train.py` (or just import `data.py`) from the project root at
least once first if `data/` doesn't exist yet — or simply visit the site;
the first training run will trigger the download automatically.

### Using it

- **Trainer** (`/trainer.html`): pick one of four preset architectures
  (Simple MLP, LeNet-5, Small CNN, Residual CNN with a skip connection) or
  build your own by clicking blocks onto the graph canvas and dragging
  connections between them. Save a graph, then click **Train** to train it
  against MNIST with live loss/accuracy/F1 streamed over a WebSocket.
- **Tester** (`/tester.html`): pick any trained model, then click a
  handwritten digit to see its prediction — the page flashes green for a
  correct guess, red for incorrect. **Randomize** pulls a fresh set of 10
  digits (one per class) from the reserved MNIST test split.

### Project layout

```
web/backend/
  main.py           FastAPI app: REST routes + WebSocket endpoint
  graph.py          DAG validation + shape inference (authoritative)
  model_builder.py  Compiles a validated graph into a PyTorch nn.Module
  presets.py        The 4 built-in preset architectures
  train_job.py      Background training runner + progress broadcaster
  data.py           MNIST loading + tester-sample sampling
  storage.py        Save/load models to/from web_models/
web/frontend/
  index.html, trainer.html, tester.html
  js/graph_editor.js   Node-graph editor
  js/trainer.js        Trainer page logic (presets, save, train, progress)
  js/tester.js         Tester page logic (sampling, predict, flash feedback)
  js/train_socket.js   WebSocket client helper
web/web_models/     Saved models (created at runtime, gitignored checkpoints)
```

Each saved model lives in `web/web_models/<model_id>/` with `graph.json`
(architecture), `meta.json` (status/metrics), `history.csv` (per-epoch
metrics), and `checkpoint.pt` (weights — gitignored; retrain to regenerate).
