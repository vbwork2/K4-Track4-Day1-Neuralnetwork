"""Custom MLP architectures allowed by the lab, with raw output logits."""
from __future__ import annotations
import torch
from torch import nn

EXPECTED_PARAMS = {(256, 128): 47_879, (512, 256): 161_287, (256, 128, 64): 55_687}


class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        hidden = tuple(hidden)
        if hidden not in EXPECTED_PARAMS or in_features != 54 or num_classes != 7:
            raise ValueError("Use M-base, M-wide or M-deep with 54 inputs and seven classes.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")
        layers = []
        width = in_features
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = next_width
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)
        assert count_params(self) == EXPECTED_PARAMS[hidden]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    if init not in {"zeros", "normal", "xavier", "he", "default"}:
        raise ValueError(f"Unknown initialization: {init}")
    if init == "default":
        # Preserve both weights and biases from PyTorch's default initialization.
        return
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(layer.weight)
            elif init == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Measure population standard deviation after each Linear, before ReLU."""
    training = model.training
    model.eval()
    values = []
    try:
        for layer in model.net:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                values.append(float(x.float().std(unbiased=False)))
    finally:
        model.train(training)
    return values
