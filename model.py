from dataclasses import dataclass
import torch.nn as nn

from modules.token_positional_embedding import TokenPositionalEmbedding
from modules.transformer_block import TransformerBlock

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ffn: int
    dropout: float


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.embeddings = TokenPositionalEmbedding(config.vocab_size, config.block_size, config.d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.d_model, config.n_heads, config.block_size, config.d_ffn) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)
        self.final_linear.weight = self.embeddings.token_emb.weight
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens, targets = None):
        # tokens = (B, T), targets = (B, T)
        B, T = tokens.shape
        if T > self.config.block_size:
            raise ValueError("input bigger than context window")
        
        x = self.embeddings(tokens)
        for block in self.transformer_blocks:
            x = block(x) # (B, T, C)
        before_linear = self.final_ln(x) # (B, T, C)
        logits = self.final_linear(before_linear) # (B, T, vocab_size)

        loss = None if targets is None else self.loss(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
   
        return logits, loss