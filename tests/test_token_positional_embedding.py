import pytest
import torch

from modules.token_positional_embedding import TokenPositionalEmbedding

vocab = 20
block = 10
d_model = 20
batch = 5


@pytest.fixture
def token_pos_embedding():
    yield TokenPositionalEmbedding(vocab, block, d_model)


def test_token_emb(token_pos_embedding):
    x = torch.randint(0, vocab, (batch, block))
    embs = token_pos_embedding(x)
    assert embs.shape == (batch, block, d_model)


def test_token_emb_backward(token_pos_embedding):
    x = torch.randint(0, vocab, (batch, block))
    embs = token_pos_embedding(x)
    loss = embs.sum()
    loss.backward()

    assert token_pos_embedding.token_emb.weight.grad is not None
    assert token_pos_embedding.pos_emb.weight.grad is not None
