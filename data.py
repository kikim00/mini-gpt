from tokenizer import CharTokenizer
import torch
from typing import Tuple

def load_text(path: str) -> str:
    with open(path) as f:
        return f.read()

def build_ids(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text))

class LMStream:
    def __init__(self, ids: torch.Tensor, block_size: int, batch_size: int, device: str):
        # ids -> N length 1-d tensor
        self.ids = ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        starting_indexes = torch.randint(0, len(self.ids) - self.block_size, (self.batch_size,))
        x_s = starting_indexes.unsqueeze(1) + torch.arange(self.block_size)
        y_s = starting_indexes.unsqueeze(1) + torch.arange(1, self.block_size+1)

        return x_s.to(self.device), y_s.to(self.device)