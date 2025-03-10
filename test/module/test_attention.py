from __future__ import annotations
import math
from icecream import ic
import llmfs.module.attention as attention
from llmfs.ref.rope_vit import RoPEAttention, compute_axial_cis
import torch
import torch.nn as nn

import tiktoken



class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: list[int], key_shape: list[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Save attentions if debugging
        # tracker.debug('attn', attn)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)


if __name__ == '__main__':
    max_len = 1024
    sentence = """Sometimes to understand a word's meaning you need more than a definition; you need to see the word
    used in a sentence. At YourDictionary, we give you the tools to learn what a word means and how to use it correctly.
    With this sentence maker, simply type a word in the search bar and see a variety of sentences with that word used in
    its different ways. Our sentence generator can provide more context and relevance, ensuring you use a word the right
    way."""
    enc = tiktoken.get_encoding("gpt2")
    encoded_sentence = enc.encode(sentence)
    print(encoded_sentence)

    vocab_size = enc.max_token_value + 1
    embed_dim = 128

    torch.manual_seed(42)

    embedding = nn.Embedding(vocab_size, embed_dim)
    seq = torch.tensor(encoded_sentence).unsqueeze(0)
    ic("The input sequence shape is {}".format(seq.shape))

    def test_mha():
        mha = attention.MultiHeadAttention(embed_dim, 2)
        res = mha(embedding(seq), embedding(seq), embedding(seq))
        print(res[0])
    test_mha()

    def test_torch_mha():
        torch_mha = nn.MultiheadAttention(embed_dim, 2, batch_first=True)
        torch_res, _ = torch_mha(embedding(seq), embedding(seq), embedding(seq))
        print(torch_res[0])
    # test_torch_mha()

    def test_labml_mha():
        labml_mha = MultiHeadAttention(2, embed_dim, 0.0, bias=True)
        labml_res = labml_mha(query=embedding(seq), key=embedding(seq), value=embedding(seq))
        print(labml_res[0])
    # test_labml_mha()

    # Test RoPE attention
    rope = attention.RoPEAttention(embed_dim, 2, 4096)
    rope_res = rope(embedding(seq))
    print(rope_res[0])

    # rope_vit = RoPEAttention(embed_dim, 2)
    # freqs_cis = compute_axial_cis()
    # rope_vit_res = rope_vit(embedding(seq), freqs_cis)
    # print(rope_vit_res[0])