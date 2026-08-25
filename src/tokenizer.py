class CharacterTokenizer:

    def __init__(self, text=None, chars=None):
        """
        Character-level tokenizer.

        During training:
            CharacterTokenizer(text)

        During validation/test:
            CharacterTokenizer(chars=training_vocabulary)
        """

        if chars is not None:

            self.chars = sorted(chars)

        elif text is not None:

            self.chars = sorted(
                set(text)
            )

        else:

            raise ValueError(
                "Provide either text or chars."
            )

        self.vocab_size = len(
            self.chars
        )

        self.stoi = {
            character: index
            for index, character
            in enumerate(self.chars)
        }

        self.itos = {
            index: character
            for index, character
            in enumerate(self.chars)
        }


    def encode(self, text):

        unknown_characters = set(text) - set(
            self.stoi.keys()
        )

        if unknown_characters:

            raise ValueError(
                "Unknown characters found: "
                + repr(
                    sorted(unknown_characters)
                )
            )

        return [
            self.stoi[character]
            for character in text
        ]


    def decode(self, tokens):

        return "".join(
            self.itos[token]
            for token in tokens
        )