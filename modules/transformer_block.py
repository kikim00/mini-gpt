import torch.nn as nn

from modules.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dim_ffn, dropout=0.1, use_residual_connections=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.use_residual_connections = use_residual_connections

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.block_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn), nn.GELU(), nn.Linear(dim_ffn, d_model)
        )

    def forward(self, x):
        # x = (B, T, C)
        normalized_x = self.ln1(x)
        after_mha = self.mha(normalized_x)  # pre-LN
        x = x + self.dropout(after_mha) if self.use_residual_connections else self.dropout(after_mha)

        normalized_x = self.ln2(x)
        after_ffn = self.ffn(normalized_x)
        x = x + self.dropout(after_ffn) if self.use_residual_connections else self.dropout(after_ffn)
        return x
