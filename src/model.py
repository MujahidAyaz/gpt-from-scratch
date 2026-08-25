import torch
import torch.nn as nn
from src.transformer import TransformerBlock

class TokenAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim
    ):
        super().__init__()

        # Learnable token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Learnable position embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

    def forward(self, x):
        """
        x shape:
            [batch_size, block_size]

        returns:
            [batch_size, block_size, embedding_dim]
        """

        B, T = x.shape

        # Convert token IDs → vectors
        token_embeddings = self.token_embedding(x)

        # Create position IDs:
        # [0, 1, 2, ..., T-1]
        positions = torch.arange(
            T,
            device=x.device
        )

        # Convert positions → vectors
        position_embeddings = self.position_embedding(
            positions
        )

        # Add token meaning + position information
        embeddings = (
            token_embeddings
            + position_embeddings
        )

        return embeddings
    
class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_dim,
        num_heads,
        num_layers
    ):
        super().__init__()

        # Token + position embeddings
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim
        )

        # Stack Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(
            embedding_dim
        )

        # Language model head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx):

        # Token + position embeddings
        x = self.embedding(idx)

        # Transformer blocks
        x = self.blocks(x)

        # Final LayerNorm
        x = self.ln_f(x)

        # Convert representations into vocabulary logits
        logits = self.lm_head(x)

        return logits