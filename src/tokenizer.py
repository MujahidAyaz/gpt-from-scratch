class CharacterTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        # Get every unique character and sort them
        self.vocab = sorted(set(text))

        # Character → integer
        self.stoi = {
            ch: i for i, ch in enumerate(self.vocab)
        }

        # Integer → character
        self.itos = {
            i: ch for i, ch in enumerate(self.vocab)
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str):
        """Convert text into integer token IDs."""
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return "".join(self.itos[i] for i in token_ids)