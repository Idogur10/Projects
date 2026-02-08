"""Data preprocessing functions."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def downsample_data(data, factor):
    """
    Downsample trajectory data by taking every factor-th sample.

    Args:
        data: (N, seq_len, dim) array of trajectory data
        factor: Downsampling factor (e.g., 10 for 100Hz -> 10Hz)

    Returns:
        Downsampled data (N, seq_len // factor, dim)
    """
    return data[:, ::factor, :]


def preprocessing(data, vel_scaler=None, pos_mean=None):
    """
    Preprocess trajectory data with position centering and velocity scaling.

    Args:
        data: (N, seq_len, 13) array of trajectory data
        vel_scaler: Optional fitted StandardScaler for validation/test mode
        pos_mean: Optional position mean for validation/test mode

    Returns:
        Training mode: (scaled_data, vel_scaler, pos_mean)
        Validation/Test mode: scaled_data
    """
    flat_data = data.reshape(-1, 13)

    # Indices
    pos_indices = [0, 1, 2]  # X, Y, Z
    vel_indices = list(range(3, 9))  # Vx, Vy, Vz, Ax, Ay, Az

    # --- TRAINING MODE ---
    if vel_scaler is None:
        # 1. Position: CENTER ONLY
        pos_mean = np.mean(flat_data[:, pos_indices], axis=0)  # Shape (3,)

        # Center the data
        flat_data[:, pos_indices] = flat_data[:, pos_indices] - pos_mean

        # 2. Velocity: Standard Scaling
        vel_scaler = StandardScaler()
        flat_data[:, vel_indices] = vel_scaler.fit_transform(flat_data[:, vel_indices])

        scaled_data = flat_data.reshape(data.shape)
        # Return MEAN and VEL_SCALER
        return scaled_data, vel_scaler, pos_mean

    # --- VALIDATION/TEST MODE ---
    else:
        # Center using Training Mean
        flat_data[:, pos_indices] = flat_data[:, pos_indices] - pos_mean

        # Scale Velocity
        flat_data[:, vel_indices] = vel_scaler.transform(flat_data[:, vel_indices])

        scaled_data = flat_data.reshape(data.shape)
        return scaled_data
