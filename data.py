import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def get_datasets(data_dir: str):
    transform = get_transform()
    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_full, test


def get_dataloaders(batch_size: int, val_split: float, seed: int, data_dir: str, num_workers: int = 2):
    train_full, test = get_datasets(data_dir)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_test_loader(batch_size: int, data_dir: str, num_workers: int = 2):
    transform = get_transform()
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
