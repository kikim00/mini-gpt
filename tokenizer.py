class CharTokenizer:
    def __init__(self, text: str):
        self.vocab = sorted(set(text))
        self.stoi = {c: i for (i, c) in enumerate(self.vocab)}
        self.itos = {i: c for (i, c) in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, s: str) -> list[int]:
        result = []
        for c in s:
            if c in self.stoi:
                result.append(self.stoi[c])

        return result

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
