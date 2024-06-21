from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): of size (batch_size, seq_length, d_model)
        """
        d_model = x.size(2)
        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature: float, dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    __doc__ = """Applies multi-head attention to an input sequence.
    
    Args:
        embed_dim (int): dimension of embedding, same as d_model referred in the paper.
        n_heads (int): number of attention heads, does not affect the number of learnable parameters,
    as d_k = embed_dim // n_heads.
        
    """

    def __init__(
        self, batch_size: int, embed_dim: int, n_heads: int, d_v: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.d_k = embed_dim // n_heads
        # Refer to "Attention is all you need", section 3.2.2
        # For each of these we use d_k = d_v = d_model/h = 64
        d_v = self.d_k
        self.w_qs = nn.Linear(embed_dim, n_heads * self.d_k, bias=False)
        self.w_ks = nn.Linear(embed_dim, n_heads * self.d_k, bias=False)
        self.w_vs = nn.Linear(embed_dim, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, embed_dim, bias=False)

        self.attn = ScaledDotProductAttention(temperature=self.d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
    ) -> Tuple[Tensor, Tensor]:
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_ks(q).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.w_vs(q).view(batch_size, len_v, self.n_head, self.d_v)

        return self.fc(x)
