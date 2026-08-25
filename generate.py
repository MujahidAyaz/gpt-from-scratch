import csv
import sys

import torch

from src.config import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
)

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# ============================================================
# Load training text
# ============================================================

csv.field_size_limit(sys.maxsize)

with open(
    "data/train.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    text = next(reader)["text"]


# ============================================================
# Tokenizer
# ============================================================

tokenizer = CharacterTokenizer(text)

print(f"Vocabulary size: {tokenizer.vocab_size}")


# ============================================================
# Create model
# ============================================================

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
)

model = model.to(device)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    "checkpoints/gpt_checkpoint.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    Generate tokens autoregressively.

    Parameters
    ----------
    model:
        Trained GPT model.

    idx:
        Starting token IDs.
        Shape: [batch_size, sequence_length]

    max_new_tokens:
        Number of tokens to generate.

    temperature:
        Controls randomness.

        < 1.0 → more deterministic
        = 1.0 → normal sampling
        > 1.0 → more random

    top_k:
        Keep only the k most likely tokens.

    top_p:
        Keep the smallest group of tokens whose
        cumulative probability reaches p.
    """

    for _ in range(max_new_tokens):

        # ----------------------------------------------------
        # Keep only the latest context window
        # ----------------------------------------------------

        idx_cond = idx[:, -BLOCK_SIZE:]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(idx_cond)

        # ----------------------------------------------------
        # Only use predictions for the final token
        # ----------------------------------------------------

        logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        logits = logits / temperature

        # ----------------------------------------------------
        # Top-k filtering
        # ----------------------------------------------------

        if top_k is not None:

            top_k = min(
                top_k,
                logits.size(-1)
            )

            values, _ = torch.topk(
                logits,
                top_k
            )

            minimum_value = values[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < minimum_value,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits
            )

        # ----------------------------------------------------
        # Convert logits → probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Top-p / nucleus sampling
        # ----------------------------------------------------

        if top_p is not None:

            sorted_probabilities, sorted_indices = torch.sort(
                probabilities,
                descending=True
            )

            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim=-1
            )

            remove_tokens = (
                cumulative_probabilities > top_p
            )

            # Keep the first token above the threshold
            remove_tokens[:, 1:] = (
                remove_tokens[:, :-1].clone()
            )

            remove_tokens[:, 0] = False

            sorted_probabilities = sorted_probabilities.masked_fill(
                remove_tokens,
                0.0
            )

            # Re-normalize probabilities
            sorted_probabilities = (
                sorted_probabilities
                / sorted_probabilities.sum(
                    dim=-1,
                    keepdim=True
                )
            )

            # Sample from sorted probabilities
            sampled_position = torch.multinomial(
                sorted_probabilities,
                num_samples=1
            )

            next_token = torch.gather(
                sorted_indices,
                -1,
                sampled_position
            )

        else:

            # ------------------------------------------------
            # Normal sampling
            # ------------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

        # ----------------------------------------------------
        # Append new token
        # ----------------------------------------------------

        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return idx


# ============================================================
# Prompt
# ============================================================

prompt = "First Citizen:"

prompt_tokens = tokenizer.encode(prompt)

context = torch.tensor(
    [prompt_tokens],
    dtype=torch.long,
    device=device
)


# ============================================================
# Generate
# ============================================================

generated_tokens = generate(
    model=model,
    idx=context,
    max_new_tokens=500,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
)


# ============================================================
# Decode
# ============================================================

generated_text = tokenizer.decode(
    generated_tokens[0].tolist()
)


# ============================================================
# Output
# ============================================================

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(generated_text)

print("=" * 60)