import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pe import AbsolutePositionalEncoding, apply_rope, precompute_rot_matrix


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

    def __init__(self, temperature: float = 1.0, dropout: float = 0.1) -> None:
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
        """
        Input tensor of shape [batch_size, num_head, seq_length, head_dim]
        """
        head_dim = k.size(-1)
        # (bsz, num_head, seqlen, head_dim) @ (bsz, num_head, head_dim, seqlen) = (bsz, num_head, seqlen, seqlen)
        score = torch.matmul(q, k.transpose(-2, -1)) * self.temperature / math.sqrt(head_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # apply attention mask masking out <PAD> tokens
        score = self.dropout(F.softmax(score, dim=-1))
        output = torch.matmul(score, v) # (bsz, num_head, seqlen, head_dim)

        return output, score


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer from the paper "Attention Is All You Need"

    Args:
        embed_dim (int): the size of the embedding vector
        num_heads (int): the number of attention heads
        dropout (float): dropout rate
    Shape:
        - Input: (batch_size, seq_length, embed_dim)
        - Output: (batch_size, seq_length, embed_dim)
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        err_msg = "embed_dim needs to be divisible by num_heads"
        assert embed_dim % num_heads == 0, err_msg
        self.head_dim = embed_dim // self.num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)   # W_query matrix
        self.wk = nn.Linear(embed_dim, embed_dim)   # W_key matrix
        self.wv = nn.Linear(embed_dim, embed_dim)   # W_value matrix
        self.attn = ScaledDotProductAttention(dropout)

        nn.init.normal_(self.wq.weight, mean=0, std=0.01)

        self.wo = nn.Linear(embed_dim, embed_dim)   # Final output projection

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_length, embed_dim)
            mask: attention mask of shape (batch_size, seq_length)
        """
        # each of shape (bsz, seqlen, embed_dim)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q, k ,v = self.split(q), self.split(k), self.split(v)   # (bsz, num_heads, seqlen, head_dim)

        # calculate attention score
        output, score = self.attn(q, k, v, mask=mask)  # (bsz, num_heads, seqlen, head_dim) and (bsz, num_heads, seqlen, seqlen)

        output = self.concat(output)
        output = self.wo(output)
        return output

    def split(self, t: torch.Tensor) -> torch.Tensor:
        """Split tensors by specified num_heads.

        Input tensor of shape (batch_size, seq_length, embed_dim)
        """
        # We first unroll embed_dim to (self.num_heads, self.head_dim), where we have
        # $embed_dim = self.num_heads * self.head_dim$ as mentioned above
        # since we want each attention head to operate on the sequence of tokens, we transpose dimension 1 and 2, i.e.,
        # seqlen and num_heads after transpose, we get (bsz, num_heads, seqlen, head_dim),
        return t.view(*t.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)

    def concat(self, t: torch.Tensor) -> torch.Tensor:
        """Concatenate tensors by specified num_heads.

        Args:
            t: resulting tensor of multi-head attention (batch_size, seq_length, num_heads, head_dim)
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        # Make transposed tensor back to contiguous view for efficient matrix multiplication
        t = t.transpose(1, 2).contiguous()  # (bsz, seqlen, num_heads, head_dim)
        # flatten previous unrolled (bsz, seqlen, num_heads, head_dim) back to (bsz, seqlen, embed_dim) or noted as concat
        return t.view(*t.size()[:2], self.embed_dim)


class RoPEAttention(MultiHeadAttention):
    """Implementation of the RoPE Attention paper (during attention).

    Args:
        - embed_dim (int): the size of the embedding dimension (AKA d_model)
        - num_heads (int): the number of attention heads
        - max_len (int): the maximum length of the sequence required by RoPE
        - dropout (float): optional dropout rate, default to 0.0
    """
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len, dropout: float = 0.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.register_buffer("rot_matrix", precompute_rot_matrix(embed_dim // num_heads, max_seq_len))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # split to num_heads head
        q, k ,v = self.split(q), self.split(k), self.split(v)
        q, k = apply_rope(q, k, self.rot_matrix)

        output, score = self.attn(q, k, v, mask=mask)

        output = self.concat(output)
        output = self.wo(output)    # project to final linear layer
        return output