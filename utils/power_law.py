"""Power law parameter extraction for motor invariant analysis."""

import numpy as np


def get_parameters_per_sample(input_vel, delta_time, eps=1e-8, start_index=0):
    """
    Extract power law parameters (α, β) for each trajectory.

    Fits the motor invariant relationship: V = α * κ^β
    where V is velocity magnitude and κ is curvature.

    Uses log-linear regression: log(V) = log(α) + β * log(κ)

    Args:
        input_vel: Velocity data (N, seq_len, 3)
        delta_time: Time step between samples
        eps: Small constant for numerical stability
        start_index: Optional starting index for windowing

    Returns:
        alphas: Array of α parameters per trajectory (N,)
        betas: Array of β parameters per trajectory (N,)
    """
    # Extract velocity window if needed
    input_vel = input_vel[:, start_index:, :]

    # Compute acceleration via finite differences
    accel = np.gradient(input_vel, delta_time, axis=1)

    # Compute velocity magnitude and squared norms
    v_norm = np.linalg.norm(input_vel, axis=-1)
    v_sq_norm = v_norm ** 2
    a_sq_norm = np.sum(accel ** 2, axis=2)
    v_dot_a_sq = np.sum(input_vel * accel, axis=2) ** 2

    # Compute curvature: κ = ||v × a|| / ||v||³
    kappa_num = np.sqrt(np.maximum(v_sq_norm * a_sq_norm - v_dot_a_sq, 0))
    kappa = kappa_num / (v_sq_norm ** 1.5 + eps)

    Alpha, Beta = [], []

    # Fit power law for each trajectory
    for i in range(input_vel.shape[0]):
        v_seg = v_norm[i, :] + eps
        k_seg = kappa[i, :] + eps

        # Log-space linear regression
        log_V = np.log(v_seg)
        log_K = np.log(k_seg)

        # Fit: log(V) = intercept + beta * log(K)
        A = np.column_stack([np.ones_like(log_K), log_K])
        try:
            X, _, _, _ = np.linalg.lstsq(A, log_V, rcond=None)
            Alpha.append(np.exp(X[0]))  # α = exp(intercept)
            Beta.append(X[1])            # β = slope
        except np.linalg.LinAlgError:
            Alpha.append(np.nan)
            Beta.append(np.nan)

    return np.array(Alpha), np.array(Beta)
