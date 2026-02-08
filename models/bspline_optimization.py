"""B-spline trajectory optimization with motor invariant constraints."""

import torch
import torch.nn as nn


class ModelCopt(nn.Module):
    """
    Inner optimization model for bi-level trajectory fitting.

    For each trajectory, optimizes control points C_hat to minimize:
        L_inner = λ₁ * ||P(s) - R_U||² + λ₂ * ||V - α*κ^β||² + λ₃ * ||J||²

    Where:
        P(s) = B @ C_hat          (position)
        V(s) = B_dot @ C_hat      (velocity)
        κ(s) = curvature
        J(s) = B_dddot @ C_hat    (jerk)

    This implements differentiable gradient descent to maintain gradient flow
    from outer loss through inner optimization back to hyperparameters.
    """

    def __init__(self, B_matrices, B_pinv, H, K, device='cuda'):
        """
        Initialize ModelCopt with B-spline basis matrices.

        Args:
            B_matrices: Tuple of (B, B_dot, B_ddot, B_dddot) basis matrices
            B_pinv: Pseudo-inverse of B for initialization
            H: Number of evaluation points (horizon)
            K: Number of control points
            device: torch device
        """
        super(ModelCopt, self).__init__()
        self.device = device
        self.H = H  # number of evaluation points (horizon)
        self.K = K  # number of control points

        # Register B matrices as buffers (not trainable)
        self.register_buffer('B', B_matrices[0])           # (H, K)
        self.register_buffer('B_dot', B_matrices[1])       # (H, K)
        self.register_buffer('B_ddot', B_matrices[2])      # (H, K)
        self.register_buffer('B_dddot', B_matrices[3])     # (H, K)
        self.register_buffer('B_pinv', B_pinv)             # (K, H)

    def compute_curvature(self, vel, acc):
        """
        Compute curvature κ from velocity and acceleration.
        κ = ||v × a|| / ||v||³

        Args:
            vel: Velocity tensor (batch, H, 3)
            acc: Acceleration tensor (batch, H, 3)

        Returns:
            Curvature tensor (batch, H)
        """
        eps = 1e-8
        v_norm = torch.norm(vel, dim=-1, keepdim=True) + eps  # (batch, H, 1)

        # Cross product for 3D: κ = ||v × a|| / ||v||³
        cross = torch.cross(vel, acc, dim=-1)  # (batch, H, 3)
        cross_norm = torch.norm(cross, dim=-1, keepdim=True)  # (batch, H, 1)

        kappa = cross_norm / (v_norm ** 3 + eps)
        return kappa.squeeze(-1)  # (batch, H)

    def forward(self, R_U, alpha, beta, lambdas, inner_steps, inner_lr, return_inner_losses=False):
        """
        Run inner optimization loop - each trajectory is optimized independently.

        Args:
            R_U: Target trajectory positions (batch, H, 3)
            alpha: Power law alpha per sample (batch,)
            beta: Power law beta per sample (batch,)
            lambdas: (λ₁, λ₂, λ₃) - per-timestep weights (H,)
            inner_steps: Number of inner optimization steps
            inner_lr: Learning rate for inner optimization
            return_inner_losses: If True, return (C_hat_all, inner_losses) else just C_hat_all

        Returns:
            C_hat_all: Optimized control points (batch, K, 3)
            inner_losses (optional): Final inner loss per trajectory (batch,)
        """
        batch_size = R_U.shape[0]
        l1, l2, l3 = lambdas

        # Store optimized control points and inner losses for each trajectory
        C_hat_list = []
        inner_losses = []

        # Optimize each trajectory independently
        for i in range(batch_size):
            # Get single trajectory data
            R_U_i = R_U[i:i+1]      # (1, H, 3)
            alpha_i = alpha[i]      # scalar
            beta_i = beta[i]        # scalar

            # Initialize C_hat_i = B_pinv @ R_U_i
            C_hat_i = torch.matmul(self.B_pinv, R_U_i)  # (1, K, 3)
            C_hat_i = C_hat_i.clone().requires_grad_(True)  # Enable gradients for optimization

            # Inner optimization loop for trajectory i
            # IMPORTANT: Keep gradients flowing through inner loop for bi-level optimization
            for step in range(inner_steps):
                # Forward pass: compute spline outputs
                P_i = torch.einsum('hk,bkd->bhd', self.B, C_hat_i)           # (1, H, 3)
                V_i = torch.einsum('hk,bkd->bhd', self.B_dot, C_hat_i)       # (1, H, 3)
                A_i = torch.einsum('hk,bkd->bhd', self.B_ddot, C_hat_i)      # (1, H, 3)
                J_i = torch.einsum('hk,bkd->bhd', self.B_dddot, C_hat_i)     # (1, H, 3)

                # Compute curvature for trajectory i
                kappa_i = self.compute_curvature(V_i, A_i).squeeze(0)  # (H,)

                # Compute velocity magnitude
                v_norm_i = torch.norm(V_i, dim=-1).squeeze(0)  # (H,)

                # === Loss terms for trajectory i ===
                # 1. Position loss: ||P(s) - R_U||²
                pos_loss_i = torch.sum((P_i - R_U_i) ** 2, dim=-1).squeeze(0)  # (H,)

                # 2. Power law loss: ||v - α * κ^β||²
                # Clamp kappa to avoid numerical issues with negative exponents
                # Use log-space for stability: κ^β = exp(β * log(κ))
                kappa_i_safe = torch.clamp(kappa_i, min=1e-4, max=1e4)
                log_kappa = torch.log(kappa_i_safe)
                log_power = beta_i * log_kappa
                # Clamp the exponent to prevent overflow
                log_power = torch.clamp(log_power, min=-10, max=10)
                kappa_beta = torch.exp(log_power)
                power_law_target_i = alpha_i * kappa_beta
                # Clamp target to reasonable range
                power_law_target_i = torch.clamp(power_law_target_i, min=1e-6, max=1e6)
                power_law_loss_i = (v_norm_i - power_law_target_i) ** 2  # (H,)

                # 3. Jerk loss: ||J||²
                jerk_loss_i = torch.sum(J_i ** 2, dim=-1).squeeze(0)  # (H,)

                # Combine losses with per-timestep lambdas
                # L_inner = Σ_h [λ₁_h * pos_loss_h + λ₂_h * power_law_loss_h + λ₃_h * jerk_loss_h]
                loss_i = torch.sum(l1 * pos_loss_i + l2 * power_law_loss_i + l3 * jerk_loss_i)

                # Compute gradient of loss_i w.r.t C_hat_i
                grad_C_hat_i = torch.autograd.grad(
                    loss_i,
                    C_hat_i,
                    create_graph=True,  # Keep graph for outer optimization
                    retain_graph=True
                )[0]

                # Differentiable gradient descent update (NO torch.no_grad())
                C_hat_i = C_hat_i - inner_lr * grad_C_hat_i

            # Store optimized control points (still connected to lambdas via gradient graph)
            C_hat_list.append(C_hat_i)

            # Store final inner loss for this trajectory (for monitoring MAML progress)
            if return_inner_losses:
                inner_losses.append(loss_i.item())

        # Stack all optimized control points
        C_hat_all = torch.cat(C_hat_list, dim=0)  # (batch, K, 3)

        if return_inner_losses:
            return C_hat_all, inner_losses
        else:
            return C_hat_all
