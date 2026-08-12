import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_sizes=(512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
