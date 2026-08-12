from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Data
    data_dir: str = "data"
    batch_size: int = 128
    val_split: float = 0.1
    num_workers: int = 2

    # Model architecture (dense layers only)
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.3
    use_batchnorm: bool = True
    activation: str = "relu"

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 50
    patience: int = 7

    # Paths
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    report_dir: str = "outputs/reports"
