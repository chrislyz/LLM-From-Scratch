import torch

from llmfs.module.pe import AbsolutePositionalEncoding, RotaryPositionalEmbedding


if __name__ == '__main__':
    x = torch.zeros(512, 512)
    pe = AbsolutePositionalEncoding(512, 512)
    pe(x)

    rope = RotaryPositionalEmbedding(512, 512)
    rope(x)