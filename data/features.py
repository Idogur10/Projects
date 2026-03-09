"""Feature engineering functions for trajectory data."""

import numpy as np


def compute_log_curvature(data):
    """
    Compute log-curvature for each timestep and append as a new feature.

    Curvature: kappa = |v x a| / |v|^3
    Log transform: log(1 + kappa)

    Args:
        data: (N, seq_len, features) array with vel at cols 3:6, acc at cols 6:9

    Returns:
        data with log-curvature appended: (N, seq_len, features + 1)
    """
    vel = data[:, :, 3:6]
    acc = data[:, :, 6:9]

    cross_va = np.cross(vel, acc)
    cross_va_norm = np.linalg.norm(cross_va, axis=2)
    vel_norm = np.linalg.norm(vel, axis=2)

    eps = 1e-10
    curvature = cross_va_norm / (vel_norm ** 3 + eps)
    log_curvature = np.log1p(curvature)

    log_curv_expanded = log_curvature[:, :, np.newaxis]
    return np.concatenate([data, log_curv_expanded], axis=2)


def compute_curv_torsion_features(data, delta_t):
    """
    Compute log-curvature and log-|torsion| and append as new features.

    Curvature:  kappa = |v x a| / |v|^3
    Torsion:    tau   = (v x a) . j / |v x a|^2

    Args:
        data: (N, seq_len, 13) array
        delta_t: time step for jerk computation

    Returns:
        (N, seq_len, 15) array
    """
    vel = data[:, :, 3:6]
    acc = data[:, :, 6:9]

    # Jerk via numerical differentiation
    jerk = np.gradient(acc, delta_t, axis=1)

    # Speed, cross product
    speed = np.linalg.norm(vel, axis=2)
    cross_va = np.cross(vel, acc)
    cross_va_norm = np.linalg.norm(cross_va, axis=2)

    eps = 1e-10

    # Curvature
    curvature = cross_va_norm / (speed ** 3 + eps)
    log_curvature = np.log1p(curvature)

    # Torsion
    cross_dot_jerk = np.sum(cross_va * jerk, axis=2)
    torsion = cross_dot_jerk / (cross_va_norm ** 2 + eps)
    log_abs_torsion = np.log1p(np.abs(torsion))

    # Append
    log_curv_exp = log_curvature[:, :, np.newaxis]
    log_tors_exp = log_abs_torsion[:, :, np.newaxis]

    return np.concatenate([data, log_curv_exp, log_tors_exp], axis=2)
