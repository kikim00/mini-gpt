from modules.multi_head_attention import MultiHeadAttention
import torch

d_model = 50
block_size = 100

def test_multi_head_attention():
    mha = MultiHeadAttention(d_model, 10, 100)
    x = torch.rand(10, block_size, d_model)
    out = mha(x)
    assert x.shape == out.shape

def test_multi_head_attention_less_tokens():
    mha = MultiHeadAttention(d_model, 10, 100)
    x = torch.rand(10, 20, d_model)
    out = mha(x)
    assert x.shape == out.shape

def test_mha_backward():
    mha = MultiHeadAttention(d_model, 10, 100)
    x = torch.rand(10, 20, d_model, requires_grad=True)
    out = mha(x)
    out.sum().backward()
    assert mha.linear.weight.grad != None
    assert mha.heads[0].W_q.weight.grad != None
    assert x.grad is not None
