import math
from typing import Tuple

import torch
import torch.nn as nn

from icecream import ic


class AbsolutePositionalEncoding(nn.Module):
    """Applies an absolute positional encoding layer before attention.

    Positional encoding is, to some extent, similar to categorical labels only provided during training. By encoding
    positional information with attention mechanisms, it is able to distinguish the same token at different positions.
    The matrix encoded with positional information can be seen as a rotation matrix that moves attention across some
    hidden dimensions. The term "absolute" refers to the fixed `pe` matrix that is precomputed before training, and
    hence it is not learnable but a persistent tensor store in device.

    Args:
        d_model (int): the number of tokens in the batch, must be an even integer.
        max_len (int): the maximum length of sequences in the batch, it helps encode with varied-length sequences.
        n (int): any value, control the strength of oscillation,
        increasing for positive input and deceasing for negative input (default = 10000 in paper)
        dropout (float): the dropout value (default = 0.1).
    """

    def __init__(self, d_model: int, max_len: int = 5000, n: int = 10000, dropout: float = 0.1) -> None:
        """
        P(k, 2i)   = sin(k / n^{2i/d}),
        P(k, 2i+1) = cos(k / n^{2i/d}),
        where
        - $i$ is the even/odd index of a token,
        - $k$ is the position of a token in the input sequence relative to $i$ (0 <= k < 2/L),
        - $d$ is the dimension of the output in embedding space,
        - $n$ is the user-defined scalar, default to 10,000

        Extensions:
        2D positional encodings are like
        P(x, y, 2i)       = sin(x / n^{4i/D})
        P(x, y, 2i+1)     = cos(x / n^{4i/D})
        P(x, y, 2j+D/2)   = sin(y / n^{4j/D})
        P(x, y, 2j+1+D/2) = cos(y / n^{4j/D})
        """
        super().__init__()
        assert d_model % 2 == 0, "The hidden dimension must be even."
        self.dropout_ = nn.Dropout(p=dropout)

        pos = torch.arange(max_len) # (max_len)
        # divisor is 1/n^{2i/d} = n^{-2i/d} = e^{\log(n^{-2i/d})} = e^{(-2i/d)\log(n)}
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(n) / d_model))  # (d_model/2)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        t = torch.outer(pos, div)   # (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(t)
        pe[:, 1::2] = torch.cos(t)
        pe = pe.unsqueeze(0) # add batch dimension for broadcasting (1, max_len, d_model)
        self.register_buffer('positional_encodings', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
        """
        # absolute positional encodings are not part of the model parameters
        pe = self.positional_encodings[:, :x.size(1), :]    # seq_length < max_len
        x = x + pe  # additive nature
        return self.dropout_(x)


class RelativePositionalEncodingLayer(nn.Module):
    """Applies a relative positional encoding before attention

    """
    pass


def precompute_rot_matrix(d_model: int, max_len: int, n: int = 10000) -> torch.Tensor:
    """Precompute the general form of Rotation Matrix for RoPE

    The general form of Rotation Matrix is described in 3.2.2 General form, where R =
    \begin{pmatrix}
    \cos m\theta_1 & -\sin m\theta_1 & 0 & \cdots & 0 & 0 \\
    \sin m\theta_1 & \cos m\theta_1 & 0 & \cdots & 0 & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
    0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
    \end{pmatrix}
    """
    pos_index = torch.arange(max_len)   # (max_len)
    # exponential works the same way mentioned in AbsolutePositionalEncoding
    theta = torch.exp(torch.arange(0, d_model - 1, 2, dtype=torch.float) * (-math.log(n) / d_model))  # (d_model/2)
    # described in 3.4.2 Computational efficient realization of rotary matrix multiplication equation (34)
    # rotary_matrix = (x_1,\cdots,x_d) \tens (\cos m\theta_1,\cdots,\cos m\theta_{d/2}) + (-x_2,x_1,\cdots,-x_d,x_{d-1}) \tens (\sin m\theta_1,\cdots,\sin m\theta_{d/2})
    m_theta = torch.outer(pos_index, theta)  # (max_len, d_model/2)
    # described in 3.4.3 Long-term decay of RoPE equation (35)
    # pack \cos m\theta_i and \sin m\theta_i together into real part and imaginary part of a complex number
    rot_matrix = torch.polar(torch.ones_like(m_theta), m_theta) # (max_len, d_model/2)
    return rot_matrix


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, rot_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Encoding to X_q and X_k DURING attention

    Args:
        xq: input query tensor of shape (batch_size, num_heads, seq_length, head_dim),
        xk: input key tensor of shape (batch_size, num_heads, seq_length head_dim),
        rot_matrix: precomputed rotation matrix of shape (max_len, d_model/2)
    """
    # remember rot_matrix has a shape of (max_len, d_model/2) where its \cos\theta and \sin\theta has been packed together
    # view_as_complex only supports float32/float64
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))   # (bsz, seqlen, num_heads, head_dim/2, 2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))   # (bsz, seqlen, num_heads, head_dim/2, 2)
    rot_matrix = rot_matrix.unsqueeze(0)    # broadcast to (1, max_len, head_dim/2)
    rot_matrix = rot_matrix.unsqueeze(1)    # broadcast to (1, 1, max_len, head_dim/2)
    xq_out = torch.view_as_real(xq_ * rot_matrix[...,:xq.size(2),:]).flatten(3)    # (bsz, seqlen, num_heads, head_dim)
    xk_out = torch.view_as_real(xk_ * rot_matrix[...,:xq.size(2),:]).flatten(3)    # (bsz, seqlen, num_heads, head_dim)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionalEmbedding(nn.Module):
    """Applies a rotary positional encoding BEFORE attention

    See more in RoFormer: Enhanced Transformer with Rotary Position Embedding (http://arxiv.org/abs/2104.09864)
    """
    def __init__(self, d_model: int, max_len: int, n: int = 10000, dropout: float = 0.1) -> None:
        """
        \theta_i = n^{-2(i-1)/d, i \in [1,2,...,d/2]
        """
        super().__init__()
        assert d_model % 2 == 0, "The hidden dimension must be even."

        pos = torch.arange(max_len) # (max_len)
        # n^{-2(i-1)/d} = e^{\log(n^{-2(i-1)/d}) = e^{2(i-1)/d * \log(n)} = exp(2(i-1) * \log(n) / d)j
        # as i \in [1,2,...,d/2], 2(i-1) is in range [0,d-2] or [0, d-1) with step being 2
        theta = torch.exp(torch.arange(0, d_model-1, 2, dtype=torch.float) * (-math.log(n) / d_model))  # [d_model/2]
        freq = torch.outer(pos, theta)
        rope = torch.polar(torch.ones_like(freq), freq) # (max_len, d_model/2)
        self.register_buffer("rope", rope)  # be aware that rope embeds real part (cos) and imaginary (sin) part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(torch.view_as_real(self.rope)[4])
        return x