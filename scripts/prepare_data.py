import torch

from src.tokenizer import CharacterTokenizer
from src.dataset import create_sequences


# Load text
with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Create tokenizer
tokenizer = CharacterTokenizer(text)

print(f"Dataset length: {len(text)} characters")
print(f"Vocabulary size: {tokenizer.vocab_size}")


# Encode entire dataset
data = tokenizer.encode(text)

print(f"First 50 token IDs:")
print(data[:50])


# Convert to tensor
data = torch.tensor(data, dtype=torch.long)


# Create training sequences
block_size = 32

X, Y = create_sequences(data, block_size)


print(f"\nInput shape:  {X.shape}")
print(f"Target shape: {Y.shape}")


# Show first example
print("\nFirst input:")
print(X[0])

print("\nFirst target:")
print(Y[0])


# Decode first example
print("\nDecoded input:")
print(tokenizer.decode(X[0].tolist()))

print("\nDecoded target:")
print(tokenizer.decode(Y[0].tolist()))