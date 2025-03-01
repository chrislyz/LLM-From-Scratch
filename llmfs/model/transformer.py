import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...