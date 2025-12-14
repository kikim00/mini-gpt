import torch.nn as nn
import torch

from modules.self_attention import SelfAttention



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # make sure d_model is divisible by num_heads
        self.d_v = d_model // num_heads
        self.d_k = self.d_v # use same dimension for Q, K, V embeddings
        self.block_size = block_size

        self.heads = nn.ModuleList([SelfAttention(d_model, self.d_k, self.d_v, block_size) for _ in range(num_heads)])
        self.linear = nn.Linear(d_model, d_model)

    
    def forward(self, x):
        B, T, C = x.shape

        outs = []
        for i in range(self.num_heads):
            outs.append(self.heads[i](x)) # each out = (B, T, d_v)
        
        outs = torch.cat(outs, -1)
        outs = self.linear(outs)

        return outs



