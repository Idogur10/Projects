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


def preprocessing(data, vel_scaler=None, pos_mean=None, pos_std=None, extra_scaler=None):
    """
    Preprocess trajectory data with position standardization and velocity scaling.
    Supports 13 features (original), 14 (+ curvature), or more.

    Extra columns beyond the base 13 (indices 13, 14, 15, ...) are all
    standardized together with a single StandardScaler.

    Args:
        data: (N, seq_len, features) array of trajectory data
        vel_scaler: Optional fitted StandardScaler for validation/test mode
        pos_mean: Optional position mean for validation/test mode
        pos_std: Optional position std for validation/test mode
        extra_scaler: Optional fitted StandardScaler for extra features (validation/test mode)

    Returns:
        Training mode: (scaled_data, vel_scaler, pos_mean, pos_std) or
                       (scaled_data, vel_scaler, pos_mean, pos_std, extra_scaler) if features > 13
        Validation/Test mode: scaled_data
    """
    n_features = data.shape[-1]
    flat_data = data.reshape(-1, n_features)

    # Indices
    pos_indices = [0, 1, 2]  # X, Y, Z
    vel_indices = list(range(3, 9))  # Vx, Vy, Vz, Ax, Ay, Az
    extra_indices = list(range(13, n_features))  # curvature, alpha, log_beta, etc.

    # --- TRAINING MODE ---
    if vel_scaler is None:
        # 1. Position: STANDARDIZE (center + scale)
        pos_mean = np.mean(flat_data[:, pos_indices], axis=0)  # Shape (3,)
        pos_std = np.std(flat_data[:, pos_indices], axis=0)    # Shape (3,)

        # Standardize position
        flat_data[:, pos_indices] = (flat_data[:, pos_indices] - pos_mean) / pos_std

        # 2. Velocity + Acceleration: Standard Scaling
        vel_scaler = StandardScaler()
        flat_data[:, vel_indices] = vel_scaler.fit_transform(flat_data[:, vel_indices])

        # 3. Extra features: Standard Scaling (if present)
        if extra_indices:
            extra_scaler = StandardScaler()
            flat_data[:, extra_indices] = extra_scaler.fit_transform(flat_data[:, extra_indices])

        scaled_data = flat_data.reshape(data.shape)
        if extra_indices:
            return scaled_data, vel_scaler, pos_mean, pos_std, extra_scaler
        return scaled_data, vel_scaler, pos_mean, pos_std

    # --- VALIDATION/TEST MODE ---
    else:
        # Standardize position using Training stats
        flat_data[:, pos_indices] = (flat_data[:, pos_indices] - pos_mean) / pos_std

        # Scale Velocity + Acceleration
        flat_data[:, vel_indices] = vel_scaler.transform(flat_data[:, vel_indices])

        # Scale Extra features (if present)
        if extra_indices and extra_scaler is not None:
            flat_data[:, extra_indices] = extra_scaler.transform(flat_data[:, extra_indices])

        scaled_data = flat_data.reshape(data.shape)
        return scaled_data
