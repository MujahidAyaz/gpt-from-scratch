import torch


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu"
):
    """
    Randomly sample a batch of input/target sequences.

    X contains the current sequence.
    Y contains the same sequence shifted by one character.
    """

    # Random starting positions
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    # Target sequences shifted by one position
    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    # Move tensors to device
    x = x.to(device)
    y = y.to(device)

    return x, y