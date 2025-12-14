import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, block_size):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.block_size = block_size

        self.W_q = nn.Linear(d_model, d_k, bias=False) 
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)


        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer('causal_mask', mask.view(1, block_size, block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        if T > self.block_size:
            raise ValueError(f"T is too big: {T}. block size is {self.block_size}")

        Q = self.W_q(x) # (B, T, d_k)
        K = self.W_k(x) # (B, T, d_k)
        V = self.W_v(x) # (B, T, d_v)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
        mask = self.causal_mask[:, :T, :T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        out = F.softmax(scores, dim=-1) @ V # (B, T, d_v)

        return out


        
