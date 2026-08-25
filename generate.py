import torch

from src.config import (
    BLOCK_SIZE,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
)

from src.model import GPTLanguageModel
from src.tokenizer import CharacterTokenizer


# ----------------------------------------
# Device
# ----------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# Load dataset
# ----------------------------------------

with open("data/train.csv", "r", encoding="utf-8") as file:
    import csv
    import sys

    csv.field_size_limit(sys.maxsize)

    reader = csv.DictReader(file)
    text = next(reader)["text"]


# ----------------------------------------
# Tokenizer
# ----------------------------------------

tokenizer = CharacterTokenizer(text)


# ----------------------------------------
# Create model
# ----------------------------------------

model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
)

model = model.to(device)


# ----------------------------------------
# Load trained checkpoint
# ----------------------------------------

checkpoint = torch.load(
    "checkpoints/gpt_checkpoint.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ----------------------------------------
# Generation function
# ----------------------------------------

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens
):
    """
    Generate new tokens autoregressively.

    idx:
        [batch_size, sequence_length]

    Each iteration:
        1. Feed context into GPT
        2. Get logits
        3. Select the next-token probabilities
        4. Sample the next token
        5. Append it to the sequence
    """

    for _ in range(max_new_tokens):

        # Keep only the most recent BLOCK_SIZE tokens
        idx_cond = idx[:, -BLOCK_SIZE:]

        # Forward pass
        logits = model(idx_cond)

        # We only need predictions for the final token
        logits = logits[:, -1, :]

        # Convert logits → probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Append next token
        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return idx


# ----------------------------------------
# Prompt
# ----------------------------------------

prompt = "First Citizen:"

prompt_tokens = tokenizer.encode(prompt)

context = torch.tensor(
    [prompt_tokens],
    dtype=torch.long,
    device=device
)


# ----------------------------------------
# Generate
# ----------------------------------------

generated_tokens = generate(
    model,
    context,
    max_new_tokens=500
)


# ----------------------------------------
# Decode
# ----------------------------------------

generated_text = tokenizer.decode(
    generated_tokens[0].tolist()
)


print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(generated_text)

print("=" * 60)