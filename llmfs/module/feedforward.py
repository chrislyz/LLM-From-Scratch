import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_feats, out_feats)
        self.linear2 = nn.Linear(out_feats, in_feats)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
