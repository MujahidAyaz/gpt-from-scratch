import torch
import torch.nn as nn

from src.tokenizer import CharacterTokenizer
from src.dataset import create_sequences, get_batch
from src.model import GPTLanguageModel

from src.config import (
    BLOCK_SIZE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    LEARNING_RATE,
    MAX_ITERS,
    EVAL_INTERVAL,
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -----------------------------
# Load dataset
# -----------------------------

with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("Dataset length:", len(text))


# -----------------------------
# Tokenizer
# -----------------------------

tokenizer = CharacterTokenizer(text)

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)


print("Vocabulary size:", tokenizer.vocab_size)


# -----------------------------
# Dataset
# -----------------------------

X, Y = create_sequences(
    data,
    BLOCK_SIZE
)


print("Number of sequences:", len(X))


# -----------------------------
# Model
# -----------------------------

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
).to(device)


print(
    "Model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Loss function
# -----------------------------

loss_function = nn.CrossEntropyLoss()


# -----------------------------
# Training loop
# -----------------------------

model.train()


for iteration in range(MAX_ITERS):

    # Get batch
    x_batch, y_batch = get_batch(
        X,
        Y,
        BATCH_SIZE
    )

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)


    # -------------------------
    # Forward
    # -------------------------

    logits = model(x_batch)


    # -------------------------
    # Reshape
    # -------------------------

    B, T, C = logits.shape

    logits = logits.view(
        B * T,
        C
    )

    targets = y_batch.view(
        B * T
    )


    # -------------------------
    # Loss
    # -------------------------

    loss = loss_function(
        logits,
        targets
    )


    # -------------------------
    # Backward
    # -------------------------

    optimizer.zero_grad()

    loss.backward()


    # -------------------------
    # Update
    # -------------------------

    optimizer.step()


    # -------------------------
    # Logging
    # -------------------------

    if iteration % EVAL_INTERVAL == 0:

        print(
            f"Step {iteration:4d} | "
            f"Loss: {loss.item():.4f}"
        )


print("\nTraining complete.")