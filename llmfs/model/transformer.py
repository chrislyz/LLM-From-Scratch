from typing import Optional

import torch
import torch.nn as nn

from llmfs.module.attention import MultiHeadAttention
from llmfs.module.embedding import TransformerEmbedding
from llmfs.module.feedforward import FeedForward
from llmfs.module.norm import LayerNorm


def build_attn_mask(x: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """Build an attention mask to prevent <PAD> tokens from attending to attention calculation.
    """
    # ready to broadcast to shape of attention score that is (bsz, n_heads, seqlen, seqlen)
    attn_mask = (x != padding_idx).unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, seqlen)
    return attn_mask


def build_causal_mask(x: torch.Tensor) -> torch.Tensor:
    """Build a casual mask to prevent future tokens from leaking into the past.
    """
    seq_len = x.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return causal_mask


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_hidden: int, dropout_p: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.ffn = FeedForward(d_model, d_hidden, dropout_p)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _x = x
        x = self.attn(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_heads: int,
        d_hidden: int,
        n_layers: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len)

        layer_kwargs = {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_hidden": d_hidden,
            "dropout_p": dropout_p,
        }
        self.layers = nn.ModuleList(
            [EncoderLayer(**layer_kwargs) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    A decoder layer contains two sub-layers, a masked multi-head self-attention and a feedforward layer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_heads: int,
        d_hidden: int,
        n_layers: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.inp_embed = TransformerEmbedding(
            vocab_size, d_model, max_len
        )  # output embedding

        # masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)

        # encoder-to-decoder multi-head attention
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_p)

        # feed forward
        self.ffn = FeedForward(d_model, d_hidden, dropout_p)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        enc: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _x = x
        x = self.inp_embed(x)
        x = self.self_attn(x, tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attn(x, enc, enc, src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        num_heads: int,
        d_hidden: int,
        n_layers: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_embed = TransformerEmbedding(vocab_size, d_model, max_len)

        dec_kwargs = {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_hidden": d_hidden,
            "dropout_p": dropout_p,
        }
        self.layers = nn.ModuleList(
            [DecoderLayer(**dec_kwargs) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.out_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.out_proj(x)
        return output


class Transformer(nn.Module):
    def __init__(self, src_pad_idx: int, tgt_pad_idx: int, tgt_sos_idx: int, vocab_size: int, d_model: int, max_len, num_heads: int, d_hidden: int, n_layers: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder = Encoder(vocab_size, d_model, max_len, num_heads, d_hidden, n_layers, dropout_p)
        self.decoder = Decoder(vocab_size, d_model, max_len, num_heads, d_hidden, n_layers, dropout_p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = build_attn_mask(src, self.src_pad_idx)
        tgt_mask = build_attn_mask(tgt, self.tgt_pad_idx) & build_causal_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(enc_src, tgt_mask)
        return output

