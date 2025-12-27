import logging
from dataclasses import dataclass

import torch.nn as nn

from modules.token_positional_embedding import TokenPositionalEmbedding
from modules.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ffn: int
    dropout: float
    use_positional_embedding: bool
    tie_embedding_weights: bool
    use_residual_connections: bool
    enable_post_ln: bool


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.embeddings = TokenPositionalEmbedding(
            config.vocab_size,
            config.block_size,
            config.d_model,
            config.use_positional_embedding,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.block_size,
                    config.d_ffn,
                    dropout=config.dropout,
                    use_residual_connections=config.use_residual_connections,
                    enable_post_ln=config.enable_post_ln,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)
        self.apply(init_weights)

        if config.tie_embedding_weights:
            logger.info(
                "Tying embedding weights between token embedding and final linear layer."
            )
            self.final_linear.weight = self.embeddings.token_emb.weight
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens, targets=None):
        # tokens = (B, T), targets = (B, T)
        B, T = tokens.shape
        if T > self.config.block_size:
            raise ValueError("input bigger than context window")

        x = self.embeddings(tokens)
        for block in self.transformer_blocks:
            x = block(x)  # (B, T, C)

        if self.config.enable_post_ln:
            before_linear = x  # No need to normalize for post-LN
        else:
            before_linear = self.final_ln(x)  # normalize for pre-LN

        logits = self.final_linear(before_linear)  # (B, T, vocab_size)

        loss = (
            None
            if targets is None
            else self.loss(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )
        )

        return logits, loss
