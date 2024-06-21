import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    __doc__ = """Applies word embedding for given input

    Args:
        num_embeddings (int): the number of tokens in the vocabulary.
        embedding_dim (int): the size of dimensionality of your embeddings or the number of features to represent one
        token (default = 512).

    The resulting tensor has the (batch_size, seq_length, embedding_dim)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int = 512) -> None:
        super().__init__()


class AbsolutePositionalEncodingLayer(nn.Module):
    __doc__ = """Applies an absolute positional encoding before attention.

    Args:
        d_model (int): the number of tokens in the batch, must be an even integer.
        max_len (int): the maximum length of sequences in the batch, it helps encode with varied-length sequences.
        n (int): any value, control the strength of oscillation,
        increasing for positive input and deceasing for negative input (default = 10000 in paper)
        dropout (float): the dropout value (default = 0.1).
    """

    def __init__(self, d_model: int, max_len: int = 5000, n: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % 2 == 0
        self.dropout_ = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # the reason why d_model has to be an even integer is listed in the following two lines
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('positional_encodings', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # positional encodings are not learnable parameters, hence detach() from the graph
        pe = self.positional_encodings[:x.size(0)].detach().requires_grad_(False)
        x = x + pe
        return self.dropout_(x)


class RelativePositionalEncodingLayer(nn.Module):
    """Applies a relative positional encoding before attention

    """
    pass


class RotaryPositionalEmbedding(nn.Module):
    pass
