"""B-spline utilities for trajectory approximation."""

import numpy as np
import torch
from scipy.interpolate import BSpline


def initialize_bspline_matrices(H, K, degree=3, device='cuda'):
    """
    Initialize B-spline basis matrices for position, velocity, acceleration, and jerk.

    Args:
        H: Number of evaluation points (horizon)
        K: Number of control points
        degree: B-spline degree (default 3 for cubic)
        device: torch device

    Returns:
        B_matrices: tuple of (B, B_dot, B_ddot, B_dddot) torch tensors
        B_pinv: Pseudo-inverse of B for control point initialization
    """
    # Setup clamped knot vector
    n_knots = K + degree + 1
    knots = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_knots - 2 * degree),
        np.ones(degree)
    ])

    # Evaluation points uniformly distributed in [0, 1]
    s_vals = np.linspace(0, 1, H)

    # Build basis matrices using identity coefficients
    coeffs = np.eye(K)
    spline = BSpline(knots, coeffs, degree)

    # Compute B-spline basis and derivatives at evaluation points
    B = torch.from_numpy(spline(s_vals)).float().to(device)
    B_dot = torch.from_numpy(spline.derivative(1)(s_vals)).float().to(device)
    B_ddot = torch.from_numpy(spline.derivative(2)(s_vals)).float().to(device)
    B_dddot = torch.from_numpy(spline.derivative(3)(s_vals)).float().to(device)

    # Pseudo-inverse for control point initialization
    B_pinv = torch.linalg.pinv(B)

    return (B, B_dot, B_ddot, B_dddot), B_pinv
