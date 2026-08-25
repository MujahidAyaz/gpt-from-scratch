import torch
import torch.nn as nn

from src.attention import MultiHeadAttention

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                4 * embedding_dim
            ),

            nn.GELU(),

            nn.Linear(
                4 * embedding_dim,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.network(x)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_dim
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(
            embedding_dim
        )

        self.ln2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.ln1(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x