class CharacterTokenizer:
    """
    Character-level tokenizer.

    Converts characters to integer token IDs and back.
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))

        self.vocab_size = len(self.chars)

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    def encode(self, text: str):
        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, tokens):
        return "".join(
            self.itos[token]
            for token in tokens
        )