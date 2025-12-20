import torch

from modules.transformer_block import TransformerBlock

d_model = 100
num_heads = 10
block_size = 50
dim_ffn = d_model * 4


def test_transformer_block():
    block = TransformerBlock(d_model, num_heads, block_size, dim_ffn)
    x = torch.rand(5, 30, d_model)
    out = block(x)
    assert x.shape == out.shape
