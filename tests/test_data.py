from pathlib import Path

import torch

from data import LMStream
from tokenizer import CharTokenizer

# Load data.txt relative to project root to make test location-independent
root_dir = Path(__file__).resolve().parent.parent
text = (root_dir / "data.txt").read_text()
tokenizer = CharTokenizer(text)
ids = torch.tensor(tokenizer.encode(text))


def test_dictionary_size():
    # Current data.txt contains 96 unique characters
    assert tokenizer.vocab_size == 96


def test_stream():
    stream = LMStream(ids, 5, 3, "cpu")
    x_s, y_s = stream.get_batch()
    assert x_s[1][2] == y_s[1][1]
    assert x_s[0][1] == y_s[0][0]
