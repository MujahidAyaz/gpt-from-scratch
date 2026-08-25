# GPT From Scratch — Decoder-Only Transformer Language Model

A compact **GPT-style decoder-only Transformer language model implemented from scratch in PyTorch**, designed to demonstrate the core mechanics behind modern autoregressive language models.

This project covers the complete language-modeling pipeline — from character-level tokenization and causal self-attention to Transformer blocks, training, checkpointing, evaluation, and autoregressive text generation.

The implementation intentionally avoids **Hugging Face Transformers and pretrained language models**, allowing the underlying architecture and training process to be examined directly.

---

## Overview

This project implements and trains a small **decoder-only Transformer** on Shakespearean dialogue.

Rather than treating a language model as a black box, the project builds the essential components manually:

```text
Raw Text
   ↓
Character Tokenization
   ↓
Token IDs
   ↓
Context Windows
   ↓
Token + Position Embeddings
   ↓
Causal Multi-Head Self-Attention
   ↓
Feed-Forward Network
   ↓
Residual Connections
   ↓
Layer Normalization
   ↓
Transformer Blocks
   ↓
Language Model Head
   ↓
Next-Token Prediction
   ↓
Cross-Entropy Loss
   ↓
Backpropagation
   ↓
AdamW Optimization
   ↓
Checkpointing
   ↓
Autoregressive Generation
```

The result is a functional, end-to-end GPT-style language model that can learn the statistical patterns of its training corpus and generate new Shakespeare-like text.

---

## Key Features

* Decoder-only Transformer architecture
* Character-level tokenizer implemented from scratch
* Learnable token embeddings
* Learnable positional embeddings
* Scaled dot-product attention
* Causal attention masking
* Multi-head self-attention
* Pre-LayerNorm Transformer blocks
* GELU feed-forward networks
* Residual connections
* Cross-entropy language-modeling objective
* AdamW optimization
* Training and validation monitoring
* Best-checkpoint tracking
* Test-set evaluation
* Autoregressive text generation
* Temperature sampling
* Top-K sampling
* Top-P / nucleus sampling
* Context-window management
* CPU-compatible training

---

# Architecture

The model follows the fundamental architecture of a GPT-style decoder-only Transformer.

```text
                    Input Text
                       │
                       ▼
                Character Tokenizer
                       │
                       ▼
                    Token IDs
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Token Embedding   Position Embedding
              │                 │
              └────────┬────────┘
                       ▼
                    Addition
                       │
                       ▼
              ┌─────────────────┐
              │ Transformer     │
              │ Block           │
              │                 │
              │ LayerNorm       │
              │      ↓          │
              │ Causal MHA      │
              │      ↓          │
              │ Residual Add    │
              │      ↓          │
              │ LayerNorm       │
              │      ↓          │
              │ Feed Forward    │
              │      ↓          │
              │ Residual Add    │
              └────────┬────────┘
                       │
                       ▼
                 Transformer
                    Blocks
                       │
                       ▼
                  Final LayerNorm
                       │
                       ▼
                Language Model Head
                       │
                       ▼
                 Vocabulary Logits
                       │
                       ▼
              Next-Token Prediction
```

### Model Configuration

| Parameter              |                    Value |
| ---------------------- | -----------------------: |
| Architecture           | Decoder-only Transformer |
| Tokenization           |          Character-level |
| Vocabulary Size        |                       65 |
| Context Length         |                       32 |
| Embedding Dimension    |                       64 |
| Attention Heads        |                        4 |
| Transformer Layers     |                        3 |
| Head Dimension         |                       16 |
| Feed-Forward Dimension |                      256 |
| Parameters             |                    ~160K |
| Training Device        |                      CPU |
| Objective              |    Next-token prediction |

---

# Core Implementation

## Tokenization

The project uses a character-level tokenizer to keep the language-modeling pipeline transparent.

Every unique character in the corpus is assigned an integer ID.

For example:

```text
"First"
```

is converted into a sequence of numerical token IDs:

```text
[18, 47, 56, 57, 58]
```

The model therefore operates entirely on numerical representations.

Character-level tokenization is intentionally simple and makes it easier to inspect the relationship between raw text, tokens, predictions, and vocabulary probabilities.

---

## Context Windows

The model uses a fixed context length:

```text
BLOCK_SIZE = 32
```

Training examples are constructed using shifted input and target sequences.

For example:

```text
Input:
First Citizen:

Target:
irst Citizen: ...
```

At every position, the model learns:

```text
P(next token | previous tokens)
```

This is the fundamental autoregressive objective used by decoder-only language models.

---

# Transformer Components

## Token and Position Embeddings

Each token ID is mapped to a learnable vector.

```text
Token ID
   ↓
Token Embedding
   ↓
64-dimensional representation
```

Because Transformers do not inherently understand sequence order, learnable positional embeddings are added to the token representations.

```text
Input Representation
=
Token Embedding
+
Position Embedding
```

This provides the model with both token identity and positional information.

---

## Causal Self-Attention

The attention mechanism uses three learned projections:

```text
Query (Q)
Key   (K)
Value (V)
```

For an input representation `X`:

```text
Q = XWQ
K = XWK
V = XWV
```

Scaled dot-product attention is then calculated as:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

The scaling factor helps maintain stable gradients when the dimensionality of the attention representations increases.

---

## Causal Masking

Because the model generates text autoregressively, each token must only attend to information that is already available.

The attention pattern is therefore causal:

```text
Token 1 → Token 1

Token 2 → Token 1, Token 2

Token 3 → Token 1, Token 2, Token 3

Token 4 → Token 1, Token 2, Token 3, Token 4
```

Conceptually, the attention mask has a lower-triangular structure:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This prevents future-token information from leaking into the prediction process.

---

## Multi-Head Attention

The model contains:

```text
4 attention heads
```

With an embedding dimension of 64:

```text
64 / 4 = 16 dimensions per head
```

Each attention head learns its own representation of relationships between tokens.

The individual head outputs are concatenated and projected back into the model dimension:

```text
Head 1 ─┐
Head 2 ─┤
Head 3 ─┼──→ Concatenate → Linear Projection
Head 4 ─┘
```

Multi-head attention allows the model to learn different types of relationships simultaneously.

---

## Feed-Forward Network

Each Transformer block contains a position-wise feed-forward network:

```text
64
 ↓
256
 ↓
GELU
 ↓
64
```

The hidden dimension is four times the model dimension:

```text
4 × 64 = 256
```

The feed-forward network provides additional nonlinear transformation capacity after the attention operation.

---

## Residual Connections

Residual connections are used around both the attention and feed-forward sublayers:

```python
x = x + attention(...)
```

and:

```python
x = x + feed_forward(...)
```

These connections help preserve information and improve optimization in deep neural networks.

---

## Layer Normalization

The Transformer blocks use **Pre-LayerNorm**:

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Self-Attention
  │
  ▼
Residual Add
  │
  ▼
LayerNorm
  │
  ▼
Feed-Forward
  │
  ▼
Residual Add
```

This places normalization before the attention and feed-forward operations.

---

# Language Model Head

After passing through the Transformer blocks, the final representation is normalized and projected into the vocabulary space.

```text
Transformer Output
       ↓
Final LayerNorm
       ↓
Linear Projection
       ↓
Vocabulary Logits
```

The resulting tensor has the shape:

```text
[batch_size, sequence_length, vocabulary_size]
```

For the current configuration:

```text
[4, 32, 65]
```

Each sequence position therefore produces a score for every character in the vocabulary.

---

# Training

The model is trained using **next-token prediction** with cross-entropy loss.

The model receives:

```text
[B, T]
```

token IDs and produces:

```text
[B, T, V]
```

logits, where:

* `B` = batch size
* `T` = sequence length
* `V` = vocabulary size

The logits and targets are flattened before calculating cross-entropy:

```text
Logits:
[B, T, V]
      ↓
[B × T, V]

Targets:
[B, T]
      ↓
[B × T]
```

The loss measures how accurately the model predicts the next character at every position.

---

# Optimization

Training uses the **AdamW optimizer**.

The training pipeline includes:

* Forward propagation
* Cross-entropy loss calculation
* Backpropagation
* Parameter updates
* Training-loss monitoring
* Validation-loss monitoring
* Periodic evaluation
* Best-model tracking
* Checkpoint saving

### Training Progress

Representative training results:

```text
Step     0 | Train Loss: 4.2899 | Val Loss: 4.2924
Step   500 | Train Loss: 2.4381 | Val Loss: 2.4545
Step  1000 | Train Loss: 2.2637 | Val Loss: 2.2696
Step  1500 | Train Loss: 2.1265 | Val Loss: 2.1561
Step  2000 | Train Loss: 2.0436 | Val Loss: 2.0856
Step  2500 | Train Loss: 1.9954 | Val Loss: 2.0252
Step  3000 | Train Loss: 1.9212 | Val Loss: 2.0086
Step  3500 | Train Loss: 1.8897 | Val Loss: 1.9700
Step  4000 | Train Loss: 1.8394 | Val Loss: 1.9368
Step  4500 | Train Loss: 1.8076 | Val Loss: 1.9159
```

Best recorded validation loss:

```text
1.9159
```

---

# Evaluation

The test set is kept completely separate from training and validation.

Final recorded test performance:

```text
Test Loss: 1.9475
```

This provides an independent measurement of how well the trained model generalizes to unseen text from the same corpus.

---

# Text Generation

After training, the model can generate text autoregressively.

A generation prompt can be provided, for example:

```text
First Citizen:
```

The generation process follows:

```text
Prompt
  ↓
Tokenize
  ↓
Model Forward Pass
  ↓
Predict Next Token
  ↓
Sample Token
  ↓
Append Token
  ↓
Update Context
  ↓
Repeat
```

The implementation supports:

* Temperature sampling
* Top-K sampling
* Top-P / nucleus sampling
* Context-window truncation

Generation can be started with:

```bash
python generate.py
```

The resulting output demonstrates that the model has learned statistical patterns from the Shakespeare corpus.

Because the model contains only approximately 160K parameters, generated text is expected to be limited in coherence and long-range consistency.

---

# Dataset

The model is trained on Shakespearean dialogue.

Dataset distribution:

| Split      | Characters |
| ---------- | ---------: |
| Training   |  1,003,854 |
| Validation |     55,770 |
| Test       |     55,770 |
| Total      |  1,115,394 |

Vocabulary size:

```text
65 characters
```

Dataset files:

```text
data/
├── train.csv
├── validation.csv
└── test.csv
```

Each dataset contains a `text` column.

---

# Project Structure

```text
GPT-from-scratch/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── checkpoints/
│   ├── .gitkeep
│   └── gpt_checkpoint.pt
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── test_dataset.py
│   └── evaluate.py
│
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── tokenizer.py
│   └── transformer.py
│
├── generate.py
├── train.py
├── LICENSE
└── README.md
```

---

# Installation

## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Git

The project is designed to run on CPU and does not require a dedicated GPU.

## Clone the Repository

```bash
git clone https://github.com/MujahidAyaz/GPT-from-scratch.git
cd GPT-from-scratch
```

## Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install torch numpy
```

---

# Usage

## 1. Prepare the Dataset

```bash
python -m scripts.prepare_data
```

This prepares the training, validation, and test data required by the project.

---

## 2. Train the Model

```bash
python train.py
```

The training pipeline:

1. Loads the dataset.
2. Builds the character vocabulary.
3. Initializes the tokenizer.
4. Creates the GPT-style Transformer.
5. Generates training batches.
6. Performs forward passes.
7. Calculates cross-entropy loss.
8. Performs backpropagation.
9. Updates parameters using AdamW.
10. Evaluates validation loss.
11. Saves model checkpoints.

---

## 3. Evaluate the Model

```bash
python -m scripts.evaluate
```

Example:

```text
============================================================
TEST SET EVALUATION
============================================================
Test Loss: 1.9475
============================================================
```

---

## 4. Generate Text

```bash
python generate.py
```

The model will load the trained checkpoint and generate text autoregressively.

---

# Technologies

### Languages & Frameworks

* Python
* PyTorch
* NumPy

### Development Tools

* Git
* GitHub

### Deep Learning Concepts

The implementation covers:

* Character tokenization
* Token embeddings
* Positional embeddings
* Query / Key / Value projections
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* GELU activation
* Feed-forward networks
* Residual connections
* Layer normalization
* Transformer blocks
* Language modeling
* Cross-entropy loss
* Backpropagation
* AdamW
* Model checkpointing
* Autoregressive generation
* Temperature sampling
* Top-K sampling
* Top-P sampling

---

# Why Build GPT From Scratch?

The purpose of this project is to understand the internal mechanics of Transformer-based language models rather than simply use an existing implementation.

Instead of treating:

```python
model.generate(...)
```

as a black box, this project exposes the major stages involved in language modeling:

```text
Text
 ↓
Token IDs
 ↓
Embeddings
 ↓
Q / K / V
 ↓
Attention Scores
 ↓
Causal Mask
 ↓
Attention Weights
 ↓
Weighted Values
 ↓
Multi-Head Attention
 ↓
Feed-Forward Network
 ↓
Residual Connections
 ↓
LayerNorm
 ↓
Logits
 ↓
Cross-Entropy
 ↓
Backpropagation
 ↓
Parameter Updates
 ↓
Autoregressive Generation
```

This provides a practical foundation for understanding the architectures used by modern Transformer-based LLMs.

---

# Is This a Real GPT?

This project implements the **core architecture and training objective of a GPT-style decoder-only Transformer**.

However, it is not intended to be comparable to production-scale GPT systems.

Modern large language models use substantially larger:

* Datasets
* Token vocabularies
* Context windows
* Embedding dimensions
* Transformer depths
* Parameter counts
* Training budgets
* Compute resources
* Data pipelines
* Evaluation systems
* Alignment and post-training methods

This project deliberately uses a small architecture so that the entire system can be understood, modified, trained, and inspected on accessible hardware.

---

# Limitations

This is an educational implementation rather than a production language model.

Current limitations include:

* Character-level tokenization
* Small vocabulary
* 32-token context window
* Approximately 160K parameters
* CPU-oriented training
* Limited model capacity
* Limited training corpus
* No distributed training
* No large-scale pretraining infrastructure
* No instruction tuning
* No RLHF
* No preference optimization
* No retrieval-augmented generation
* No production inference server

Therefore, generated text should not be interpreted as factual, instruction-following, or semantically reliable.

The model primarily learns the statistical patterns and stylistic characteristics of the training corpus.

---

# Future Improvements

## Model Architecture

* Increase model dimensionality
* Add more Transformer layers
* Increase attention heads
* Increase context length
* Add dropout
* Implement weight tying
* Improve initialization
* Experiment with regularization

## Tokenization

Replace character-level tokenization with modern subword approaches such as:

* Byte Pair Encoding (BPE)
* WordPiece
* SentencePiece
* GPT-style subword tokenization

## Training

* Learning-rate warmup
* Cosine learning-rate scheduling
* Gradient clipping
* Mixed-precision training
* Gradient accumulation
* Larger batch sizes
* GPU acceleration
* More efficient data loading
* Larger-scale datasets

## Modern LLM Pipeline

A natural progression from this project would be:

```text
Transformer Pretraining
        ↓
Instruction Fine-Tuning
        ↓
Preference Optimization
        ↓
Evaluation
        ↓
Inference
        ↓
Deployment
```

---

# Learning Outcomes

This project provides hands-on implementation experience with:

* How text becomes token IDs
* How embeddings represent tokens
* Why positional information is necessary
* How Query, Key, and Value interact
* How scaled dot-product attention works
* Why causal masking is required
* How multi-head attention works
* Why Transformers use residual connections
* Why LayerNorm is used
* How decoder-only Transformers perform next-token prediction
* How cross-entropy measures prediction error
* How gradients update model parameters
* How AdamW optimizes the network
* How checkpoints preserve model state
* How autoregressive generation works
* How temperature affects sampling
* How Top-K sampling works
* How Top-P sampling works
* How train/validation/test evaluation is performed

---

# Project Status

**Completed — Educational GPT From Scratch**

The current implementation successfully:

* Implements a decoder-only Transformer
* Implements causal multi-head self-attention
* Trains on a real text corpus
* Performs next-character prediction
* Tracks training and validation loss
* Saves model checkpoints
* Evaluates on an unseen test set
* Generates text autoregressively
* Supports temperature, Top-K, and Top-P sampling

The project is intentionally compact so that the complete language-modeling pipeline can be understood and trained using accessible hardware.

---

# Author

## Mujahid Ayaz

Software Engineering → Data Science → Machine Learning → Deep Learning → Generative AI

**GitHub:** [MujahidAyaz](https://github.com/MujahidAyaz)

---

# License

This project is released under the license included in the repository.

See [`LICENSE`](LICENSE) for the complete license terms.
