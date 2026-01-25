"""Loss functions for trajectory prediction."""

import torch


def ade_loss(pred, target):
    """
    Average Displacement Error loss.

    Args:
        pred: Predicted positions (batch, seq_len, 3)
        target: Target positions (batch, seq_len, 3)

    Returns:
        Mean L2 distance across all predictions
    """
    error = pred - target  # shape: (batch, seq_len, 3)
    dist = torch.norm(error, dim=2)  # L2 norm across x, y, z -> shape: (batch, seq_len)
    return dist.mean()
