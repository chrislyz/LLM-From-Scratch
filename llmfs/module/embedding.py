import torch
import torch.nn as nn

from llmfs.module.pe import AbsolutePositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, 1)   # <PAD>-token is assigned to index of 1
        self.pos_embed = AbsolutePositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the official implementation is
        # output = self.pos_embed(x) + self.tok_embed(x)
        # but `AbsolutePositionalEncoding` class has already operated additive on pos_embed and tok_embed
        output = self.pos_embed(self.tok_embed(x))
        return self.dropout(output)
