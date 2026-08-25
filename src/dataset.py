import torch


def create_sequences(data, block_size):
    """
    Create input-target pairs for next-token prediction.
    """

    inputs = []
    targets = []

    for i in range(len(data) - block_size):
        inputs.append(data[i:i + block_size])
        targets.append(data[i + 1:i + block_size + 1])

    X = torch.stack(inputs)
    Y = torch.stack(targets)

    return X, Y


def get_batch(X, Y, batch_size):
    """
    Randomly sample a batch of training examples.
    """

    indices = torch.randint(
        low=0,
        high=len(X),
        size=(batch_size,)
    )

    x_batch = X[indices]
    y_batch = Y[indices]

    return x_batch, y_batch