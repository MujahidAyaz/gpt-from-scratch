import csv
import os
import sys

import torch
import torch.nn.functional as F

from src.tokenizer import CharacterTokenizer
from src.dataset import get_batch
from src.model import GPTLanguageModel

from src import config


# ============================================================
# CSV LOADER
# ============================================================

def load_text_from_csv(path: str) -> str:
    """
    Load the `text` column from a CSV file.
    """

    csv.field_size_limit(sys.maxsize)

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        texts = [
            row["text"]
            for row in reader
            if row["text"]
        ]

    return "\n".join(texts)


# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    iteration,
    best_val_loss
):
    """
    Save model and training state.
    """

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "best_val_loss": best_val_loss,
    }

    torch.save(
        checkpoint,
        config.CHECKPOINT_PATH
    )

    print(
        f"Checkpoint saved at step {iteration}."
    )


def load_checkpoint(
    model,
    optimizer
):
    """
    Load checkpoint if one exists.
    """

    if not os.path.exists(
        config.CHECKPOINT_PATH
    ):
        return 0, float("inf")

    checkpoint = torch.load(
        config.CHECKPOINT_PATH,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    iteration = checkpoint["iteration"]

    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Checkpoint loaded from step {iteration}."
    )

    return iteration + 1, best_val_loss


# ============================================================
# LOAD DATA
# ============================================================

train_text = load_text_from_csv(
    "data/train.csv"
)

validation_text = load_text_from_csv(
    "data/validation.csv"
)

test_text = load_text_from_csv(
    "data/test.csv"
)

print("Device:", config.DEVICE)

print(
    "Training characters:",
    len(train_text)
)

print(
    "Validation characters:",
    len(validation_text)
)

print(
    "Test characters:",
    len(test_text)
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = CharacterTokenizer(
    train_text
)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# ============================================================
# ENCODE DATA
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=config.BLOCK_SIZE,
    embedding_dim=config.EMBEDDING_DIM,
    num_heads=config.NUM_HEADS,
    num_layers=config.NUM_LAYERS
).to(config.DEVICE)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Model parameters:",
    total_parameters
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss(data):
    """
    Estimate average loss over multiple batches.
    """

    model.eval()

    losses = torch.zeros(
        config.EVAL_STEPS
    )

    for step in range(
        config.EVAL_STEPS
    ):

        x, y = get_batch(
            data=data,
            batch_size=config.BATCH_SIZE,
            block_size=config.BLOCK_SIZE,
            device=config.DEVICE
        )

        logits = model(x)

        B, T, C = logits.shape

        logits = logits.reshape(
            B * T,
            C
        )

        targets = y.reshape(
            B * T
        )

        loss = F.cross_entropy(
            logits,
            targets
        )

        losses[step] = loss.item()

    model.train()

    return losses.mean().item()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

start_iteration, best_val_loss = load_checkpoint(
    model,
    optimizer
)


# ============================================================
# TRAINING
# ============================================================

model.train()

for iteration in range(
    start_iteration,
    config.MAX_ITERS
):

    # --------------------------------------------------------
    # Get training batch
    # --------------------------------------------------------

    x, y = get_batch(
        data=train_data,
        batch_size=config.BATCH_SIZE,
        block_size=config.BLOCK_SIZE,
        device=config.DEVICE
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(x)

    B, T, C = logits.shape

    logits = logits.reshape(
        B * T,
        C
    )

    targets = y.reshape(
        B * T
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        targets
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if iteration % config.EVAL_INTERVAL == 0:

        train_loss = estimate_loss(
            train_data
        )

        validation_loss = estimate_loss(
            validation_data
        )

        print(
            f"Step {iteration:5d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation_loss < best_val_loss:

            best_val_loss = validation_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                best_val_loss=best_val_loss
            )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\nTraining complete.")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)