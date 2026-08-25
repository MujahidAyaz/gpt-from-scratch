import torch
import torch.nn as nn


# ============================================================
# Self-Attention Head
# ============================================================

class SelfAttentionHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        head_size: int,
        block_size: int
    ):
        super().__init__()

        # ----------------------------------------------------
        # Key, Query, Value projections
        # ----------------------------------------------------

        self.key = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        # ----------------------------------------------------
        # Causal mask
        #
        # Prevents a token from attending to future tokens.
        #
        # Example:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # ----------------------------------------------------

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x -> [batch_size, sequence_length, embedding_dim]

        Output:
            output -> [batch_size, sequence_length, head_size]
        """

        _, T, _ = x.shape

        # ----------------------------------------------------
        # Create Q, K, V
        # ----------------------------------------------------

        key = self.key(x)

        query = self.query(x)

        value = self.value(x)

        # ----------------------------------------------------
        # Attention scores
        #
        # Q @ K^T
        # ----------------------------------------------------

        scores = query @ key.transpose(-2, -1)

        # ----------------------------------------------------
        # Scale attention scores
        #
        # Prevents extremely large values before softmax.
        # ----------------------------------------------------

        scores = scores / (
            key.size(-1) ** 0.5
        )

        # ----------------------------------------------------
        # Causal masking
        #
        # Future tokens receive -infinity.
        # After softmax they become zero.
        # ----------------------------------------------------

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Convert scores into probabilities
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted sum of values
        # ----------------------------------------------------

        output = attention_weights @ value

        return output


# ============================================================
# Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        head_size = embedding_dim // num_heads

        # ----------------------------------------------------
        # Create multiple independent attention heads
        # ----------------------------------------------------

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_dim=embedding_dim,
                    head_size=head_size,
                    block_size=block_size
                )
                for _ in range(num_heads)
            ]
        )

        # ----------------------------------------------------
        # Combine information from all heads
        # ----------------------------------------------------

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            [batch_size, sequence_length, embedding_dim]

        Output:
            [batch_size, sequence_length, embedding_dim]
        """

        # Run every attention head
        outputs = [
            head(x)
            for head in self.heads
        ]

        # Concatenate heads
        out = torch.cat(
            outputs,
            dim=-1
        )

        # Final linear projection
        out = self.projection(out)

        return out