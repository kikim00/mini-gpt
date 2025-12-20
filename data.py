from typing import Tuple

import torch

from tokenizer import CharTokenizer


def load_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def build_ids(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text))


class LMStream:
    def __init__(
        self, ids: torch.Tensor, block_size: int, batch_size: int, device: str
    ):
        # ids -> N length 1-d tensor
        self.ids = ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        starting_indexes = torch.randint(
            0, len(self.ids) - self.block_size, (self.batch_size, 1)
        )  # (B, 1)
        x_s = starting_indexes + torch.arange(
            self.block_size
        )  # (B, 1) + (T,) -> (B, 1) + (1, T)
        y_s = x_s + 1  # add mask of 1

        x_s = self.ids[x_s]
        y_s = self.ids[y_s]

        return x_s.to(self.device), y_s.to(self.device)
