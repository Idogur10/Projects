"""Dataset classes for trajectory prediction."""

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory data."""

    def __init__(self, inputs, targets):
        """
        Args:
            inputs: Input sequences array
            targets: Target sequences array
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )
