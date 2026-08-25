import torch
import torch.nn as nn


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()

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

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        # -------------------------
        # Create Q, K, V
        # -------------------------

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # -------------------------
        # Attention scores
        # -------------------------

        scores = query @ key.transpose(-2, -1)

        # -------------------------
        # Scale
        # -------------------------

        scores = scores / (key.size(-1) ** 0.5)

        # -------------------------
        # Causal mask
        # -------------------------

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # -------------------------
        # Convert scores → weights
        # -------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # Weighted values
        # -------------------------

        output = attention_weights @ value

        return query, key, value, output, attention_weights



class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        head_size = embedding_dim // num_heads

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

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        outputs = []

        for head in self.heads:

            _, _, _, output, _ = head(x)

            outputs.append(output)

        # Concatenate heads
        out = torch.cat(
            outputs,
            dim=-1
        )

        # Final linear projection
        out = self.projection(out)

        return out