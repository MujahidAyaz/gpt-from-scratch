import csv
import sys

import torch
import torch.nn.functional as F

from src.config import (
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE,
)

from src.dataset import get_batch
from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer


# ============================================================
# Evaluation settings
# ============================================================

EVAL_ITERS = 100


# ============================================================
# CSV loader
# ============================================================

def load_text_from_csv(path):

    csv.field_size_limit(sys.maxsize)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise ValueError(
            f"No data found in {path}"
        )

    return "\n".join(
        row["text"]
        for row in rows
    )


# ============================================================
# Load datasets
# ============================================================

train_text = load_text_from_csv(
    "data/train.csv"
)

test_text = load_text_from_csv(
    "data/test.csv"
)


print(
    f"Training characters: {len(train_text)}"
)

print(
    f"Test characters: {len(test_text)}"
)


# ============================================================
# IMPORTANT:
# Build tokenizer from TRAINING vocabulary
# ============================================================

tokenizer = CharacterTokenizer(
    text=train_text
)

print(
    f"Vocabulary size: {tokenizer.vocab_size}"
)


# ============================================================
# Encode test set using training vocabulary
# ============================================================

test_tokens = tokenizer.encode(
    test_text
)

test_data = torch.tensor(
    test_tokens,
    dtype=torch.long
)


print(
    f"Test tokens: {len(test_data)}"
)


# ============================================================
# Create model
# ============================================================

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=64,
    num_heads=4,
    num_layers=3
).to(DEVICE)


# ============================================================
# Load trained checkpoint
# ============================================================

checkpoint = torch.load(
    "checkpoints/gpt_checkpoint.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Evaluate
# ============================================================

losses = []

with torch.no_grad():

    for _ in range(EVAL_ITERS):

        x, y = get_batch(
            test_data,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=DEVICE
        )

        logits = model(x)

        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            y.reshape(B * T)
        )

        losses.append(
            loss.item()
        )


average_loss = (
    sum(losses)
    / len(losses)
)


# ============================================================
# Results
# ============================================================

print()

print("=" * 60)

print("TEST SET EVALUATION")

print("=" * 60)

print(
    f"Test Loss: {average_loss:.4f}"
)

print("=" * 60)