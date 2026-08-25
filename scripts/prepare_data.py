import csv
import sys
import torch

from src.tokenizer import CharacterTokenizer


# -----------------------------
# CSV helper
# -----------------------------

def load_text_from_csv(path: str) -> str:
    """
    Load the text column from a CSV file.
    """

    csv.field_size_limit(sys.maxsize)

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    texts = [
        row["text"]
        for row in rows
        if row["text"]
    ]

    return "\n".join(texts)


# -----------------------------
# Load datasets
# -----------------------------

train_text = load_text_from_csv(
    "data/train.csv"
)

validation_text = load_text_from_csv(
    "data/validation.csv"
)

test_text = load_text_from_csv(
    "data/test.csv"
)


# -----------------------------
# Tokenizer
# -----------------------------

tokenizer = CharacterTokenizer(
    train_text
)


# -----------------------------
# Encode datasets
# -----------------------------

train_data = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long
)

validation_data = torch.tensor(
    tokenizer.encode(validation_text),
    dtype=torch.long
)

test_data = torch.tensor(
    tokenizer.encode(test_text),
    dtype=torch.long
)


# -----------------------------
# Information
# -----------------------------

print("Training characters:", len(train_text))
print("Validation characters:", len(validation_text))
print("Test characters:", len(test_text))

print("\nVocabulary size:", tokenizer.vocab_size)

print("\nTraining tokens:", len(train_data))
print("Validation tokens:", len(validation_data))
print("Test tokens:", len(test_data))


# -----------------------------
# Preview
# -----------------------------

print("\nFirst 100 training tokens:")
print(train_data[:100].tolist())

print("\nDecoded training text:")
print(
    tokenizer.decode(
        train_data[:100].tolist()
    )
)

print("\nDecoded validation text:")
print(
    tokenizer.decode(
        validation_data[:100].tolist()
    )
)

print("\nDecoded test text:")
print(
    tokenizer.decode(
        test_data[:100].tolist()
    )
)