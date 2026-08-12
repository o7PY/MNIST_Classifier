"""MNIST loading for the web app.

Mirrors the conventions in ../data.py (mean=0.1307, std=0.3081, 54k/6k/10k
split) and points at the same data/ directory so MNIST isn't re-downloaded.
"""

import base64
import io
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

PARENT_DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data")

_test_dataset = None
_by_digit_index: dict[int, list[int]] | None = None


def _transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])


def get_train_val_loaders(batch_size: int, val_split: float = 0.1, seed: int = 42, num_workers: int = 0):
    transform = _transform()
    train_full = datasets.MNIST(root=PARENT_DATA_DIR, train=True, download=True, transform=transform)
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_test_dataset():
    """Raw (image, label) dataset - used by the Tester section to sample
    individual images rather than iterate mini-batches. Cached at module
    level since it's read-only and reused across every tester request."""
    global _test_dataset
    if _test_dataset is None:
        transform = _transform()
        _test_dataset = datasets.MNIST(root=PARENT_DATA_DIR, train=False, download=True, transform=transform)
    return _test_dataset


def _get_by_digit_index() -> dict[int, list[int]]:
    global _by_digit_index
    if _by_digit_index is None:
        dataset = get_test_dataset()
        by_digit: dict[int, list[int]] = {d: [] for d in range(10)}
        for idx, label in enumerate(dataset.targets.tolist()):
            by_digit[label].append(idx)
        _by_digit_index = by_digit
    return _by_digit_index


def sample_one_per_digit() -> list[int]:
    """Returns 10 sample_ids (test-set indices), one uniformly-random image
    per digit class 0-9, drawn only from the reserved test split."""
    by_digit = _get_by_digit_index()
    return [random.choice(by_digit[d]) for d in range(10)]


def render_digit_png_base64(sample_id: int) -> str:
    dataset = get_test_dataset()
    raw = dataset.data[sample_id].numpy()  # raw uint8 pixels, unnormalized
    img = Image.fromarray(raw, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_model_input_tensor(sample_id: int) -> torch.Tensor:
    dataset = get_test_dataset()
    tensor, _label = dataset[sample_id]  # normalized, shape (1, 28, 28)
    return tensor


def get_true_label(sample_id: int) -> int:
    dataset = get_test_dataset()
    return int(dataset.targets[sample_id].item())
