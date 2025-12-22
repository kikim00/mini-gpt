import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, dim_model, use_positional_embedding=True):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim_model)
        self.pos_emb = nn.Embedding(block_size, dim_model)
        self.use_positional_embedding = use_positional_embedding

    def forward(self, x):
        # x: (B, T) shape ids for tokens
        B, T = x.shape

        token_embeddings = self.token_emb(x)  # (B, T, dim_model)
        position_embeddings = self.pos_emb(
            torch.arange(T, device=x.device)
        )  # (T, dim_model)

        if self.use_positional_embedding:
            return token_embeddings + position_embeddings
        else:
            return token_embeddings