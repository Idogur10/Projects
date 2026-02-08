import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

from config import (
    HIDDEN_DIM, N_EPOCHS, LEARNING_RATE,
    RANDOM_SEED, DIRECTORY, WINDOW_SIZE, HORIZON, DELTA_T, OUTPUT_SIZE,
    W_POS_ADE, W_VEL_MSE, W_ACC_MSE, PATIENCE,
    DOWNSAMPLE_FACTOR, ORIGINAL_WINDOW_SIZE, ORIGINAL_HORIZON
)
from data import TrajectoryDataset, preprocessing
from models import Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM
from utils import ade_loss, evaluate_at_timestamps, plot_stepwise_errors, plot_val_trajs_3d_and_xyz

BATCH_SIZE = 5  # Reduced for debugging (working version)
INNER_LR = 1e-3
INNER_STEPS = 3


class ModelCopt(nn.Module):
    """
    Inner optimization model (Part 1 of pseudo-code).

    For each trajectory, optimizes control points C_hat to minimize:
        L_inner = λ₁ * ||P(s) - R_U||² + λ₂ * ||V - α*κ^β||² + λ₃ * ||J||²

    Where:
        P(s) = B @ C_hat          (position)
        V(s) = B_dot @ C_hat      (velocity)
        κ(s) = curvature
        J(s) = B_dddot @ C_hat    (jerk)
    """

    def __init__(self, B_matrices, B_pinv, H, K, device='cuda'):
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
        """
        eps = 1e-8
        v_norm = torch.norm(vel, dim=-1, keepdim=True) + eps  # (batch, H, 1)

        # For 3D: κ = ||v × a|| / ||v||³
        # Cross product
        cross = torch.cross(vel, acc, dim=-1)  # (batch, H, 3)
        cross_norm = torch.norm(cross, dim=-1, keepdim=True)  # (batch, H, 1)

        kappa = cross_norm / (v_norm ** 3 + eps)
        return kappa.squeeze(-1)  # (batch, H)

    def forward(self, R_U, alpha, beta, lambdas, inner_steps=INNER_STEPS, inner_lr=INNER_LR):
        """
        Run inner optimization loop - each trajectory is optimized independently.

        Args:
            R_U: Target trajectory positions (batch, H, 3)
            alpha: Power law alpha per sample (batch,)
            beta: Power law beta per sample (batch,)
            lambdas: (λ₁, λ₂, λ₃) - per-timestep weights (H,)
            inner_steps: Number of inner optimization steps
            inner_lr: Learning rate for inner optimization

        Returns:
            C_hat_all: Optimized control points (batch, K, 3)
        """
        batch_size = R_U.shape[0]
        l1, l2, l3 = lambdas

        # Store optimized control points for each trajectory
        C_hat_list = []

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

        # Stack all optimized control points
        C_hat_all = torch.cat(C_hat_list, dim=0)  # (batch, K, 3)
        return C_hat_all


def initialize_bspline_matrices(H, K, degree=3, device='cuda'):
    """
    Initialize B-spline basis matrices.

    Args:
        H: Number of evaluation points (horizon)
        K: Number of control points
        degree: B-spline degree (default 3 for cubic)
        device: torch device

    Returns:
        B_matrices: tuple of (B, B_dot, B_ddot, B_dddot)
        B_pinv: Pseudo-inverse of B
    """
    # Setup clamped knot vector
    n_knots = K + degree + 1
    knots = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_knots - 2 * degree),
        np.ones(degree)
    ])

    # Evaluation points
    s_vals = np.linspace(0, 1, H)

    # Build basis matrices
    coeffs = np.eye(K)
    spline = BSpline(knots, coeffs, degree)

    B = torch.from_numpy(spline(s_vals)).float().to(device)
    B_dot = torch.from_numpy(spline.derivative(1)(s_vals)).float().to(device)
    B_ddot = torch.from_numpy(spline.derivative(2)(s_vals)).float().to(device)
    B_dddot = torch.from_numpy(spline.derivative(3)(s_vals)).float().to(device)

    # Pseudo-inverse for initialization
    B_pinv = torch.linalg.pinv(B)

    return (B, B_dot, B_ddot, B_dddot), B_pinv

def get_parameters_per_sample(input_vel, delta_time, eps=1e-8, start_index=0):
    """
    Extract power law parameters (α, β) for each trajectory.
    Fits: V = α * κ^β using log-linear regression.
    """
    input_vel = input_vel[:, start_index:, :]
    accel = np.gradient(input_vel, delta_time, axis=1)

    v_norm = np.linalg.norm(input_vel, axis=-1)
    v_sq_norm = v_norm ** 2
    a_sq_norm = np.sum(accel ** 2, axis=2)
    v_dot_a_sq = np.sum(input_vel * accel, axis=2) ** 2

    # Curvature
    kappa_num = np.sqrt(np.maximum(v_sq_norm * a_sq_norm - v_dot_a_sq, 0))
    kappa = kappa_num / (v_sq_norm ** 1.5 + eps)

    Alpha, Beta = [], []

    for i in range(input_vel.shape[0]):
        v_seg = v_norm[i, :] + eps
        k_seg = kappa[i, :] + eps

        log_V = np.log(v_seg)
        log_K = np.log(k_seg)

        A = np.column_stack([np.ones_like(log_K), log_K])
        try:
            X, _, _, _ = np.linalg.lstsq(A, log_V, rcond=None)
            Alpha.append(np.exp(X[0]))
            Beta.append(X[1])
        except np.linalg.LinAlgError:
            Alpha.append(np.nan)
            Beta.append(np.nan)

    return np.array(Alpha), np.array(Beta)


def downsample_data(data, factor):
    """Downsample trajectory data by taking every factor-th sample."""
    return data[:, ::factor, :]


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Downsampling: 100Hz -> {100 // DOWNSAMPLE_FACTOR}Hz (factor={DOWNSAMPLE_FACTOR})")
    print(f"Horizon: {HORIZON} timesteps, Delta_t: {DELTA_T}s\n")

    # === Load and preprocess data ===
    filename = "No_B_legnth_250_step_size50vert_hor_WithoutB.npy"
    file_path = os.path.join(DIRECTORY, filename)
    train_data = np.load(file_path)
    train_data = downsample_data(train_data, DOWNSAMPLE_FACTOR)

    # Ground truth trajectory (positions only)
    R_TRUE = train_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3]
    R_TRUE_tensor = torch.tensor(R_TRUE, dtype=torch.float32).to(device)

    # Prepare inputs
    train_inputs_raw = train_data[:, :WINDOW_SIZE, :]
    train_inputs_raw, vel_scaler, pos_mean = preprocessing(train_inputs_raw)

    # === Load LSTM model (matching train.py) ===
    encoder = Encoder_LSTM(input_size=13, hidden_size=HIDDEN_DIM, num_layers=1)
    decoder = Decoder_LSTM(input_size=9, hidden_size=HIDDEN_DIM, num_layers=1, output_size=3)
    lstm_model = Seq2SeqLSTM(
        encoder=encoder,
        decoder=decoder,
        horizon=HORIZON,
        delta_t=DELTA_T,
        pos_mean=pos_mean,
        scaler_vel=vel_scaler
    ).to(device)
    weights_path = 'C:/Users/idogu/OneDrive/Documents/Master/data/model/Seq2SeqLSTM_Hid96_ds10.pth'
    lstm_model.load_state_dict(torch.load(weights_path, map_location=device))
    lstm_model.eval()

    # === Get LSTM predictions (R_U) ===
    with torch.no_grad():
        input_tensor = torch.tensor(train_inputs_raw[:BATCH_SIZE], dtype=torch.float32).to(device)
        R_U = lstm_model(input_tensor)[:, :, :3]  # positions only (batch, H, 3)

    # === Extract power law parameters ===
    velocity_input = train_data[:BATCH_SIZE, WINDOW_SIZE:WINDOW_SIZE + HORIZON, 3:6]
    alphas, betas = get_parameters_per_sample(velocity_input, DELTA_T)

    # Check for NaN/inf in extracted parameters
    print(f"Alpha range: [{np.nanmin(alphas):.4f}, {np.nanmax(alphas):.4f}], mean: {np.nanmean(alphas):.4f}")
    print(f"Beta range: [{np.nanmin(betas):.4f}, {np.nanmax(betas):.4f}], mean: {np.nanmean(betas):.4f}")
    print(f"NaN count - alphas: {np.sum(np.isnan(alphas))}, betas: {np.sum(np.isnan(betas))}")

    # Replace NaN with default values
    alphas = np.nan_to_num(alphas, nan=1.0)
    betas = np.nan_to_num(betas, nan=-0.33)

    alphas_tensor = torch.tensor(alphas, dtype=torch.float32).to(device)
    betas_tensor = torch.tensor(betas, dtype=torch.float32).to(device)

    # === Initialize B-spline matrices ===
    H = HORIZON  # evaluation points (5 timesteps)
    K = 8        # control points (increased to test jerk optimization)
    B_matrices, B_pinv = initialize_bspline_matrices(H, K, degree=3, device=device)

    # Ground truth control points
    C_TRUE = torch.matmul(B_pinv, R_TRUE_tensor[:BATCH_SIZE])  # (batch, K, 3)

    # === Initialize ModelCopt ===
    model_copt = ModelCopt(B_matrices, B_pinv, H, K, device=device).to(device)

    # === Initialize hyperparameters (λ₁, λ₂, λ₃) ===
    # Using log-space for positivity constraint: exp(log_l) = l
    # λ₁ = 500, λ₂ = 1000, λ₃ = 0.1 (increased to test jerk)
    import math
    log_l1 = torch.full((H,), math.log(500), device=device, requires_grad=True)    # position weight
    log_l2 = torch.full((H,), math.log(1000), device=device, requires_grad=True)   # power law weight
    log_l3 = torch.full((H,), math.log(0.1), device=device, requires_grad=True)    # jerk weight (increased from 1e-7)

    optimizer_lambda = optim.Adam([log_l1, log_l2, log_l3], lr=1e-4)

    # === Outer optimization loop ===
    print("=" * 50)
    print("STARTING BI-LEVEL OPTIMIZATION")
    print("=" * 50)

    n_epochs = 50
    for epoch in range(n_epochs):
        optimizer_lambda.zero_grad()

        # Get current lambdas (exponentiate for positivity)
        l1 = torch.exp(log_l1)
        l2 = torch.exp(log_l2)
        l3 = torch.exp(log_l3)
        lambdas = (l1, l2, l3)

        # === Inner loop: optimize C_hat for each trajectory ===
        C_hat = model_copt(R_U, alphas_tensor, betas_tensor, lambdas,
                          inner_steps=INNER_STEPS, inner_lr=INNER_LR)

        # === Outer loss: trajectory-level loss (not control point loss) ===
        # Evaluate trajectories at the B-spline points
        P_true = torch.einsum('hk,bkd->bhd', model_copt.B, C_TRUE)   # (batch, H, 3)
        P_hat = torch.einsum('hk,bkd->bhd', model_copt.B, C_hat)     # (batch, H, 3)

        # ADE loss (better than MSE for trajectories)
        loss_outer = torch.mean(torch.norm(P_true - P_hat, dim=-1))  # Average displacement error

        # Backprop to update lambdas
        loss_outer.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([log_l1, log_l2, log_l3], 1.0)

        optimizer_lambda.step()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Outer Loss: {loss_outer.item():.6f} | "
                  f"λ₁: {l1.mean().item():.4f}, λ₂: {l2.mean().item():.4f}, λ₃: {l3.mean().item():.2e}")

    # === Final results ===
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Final λ₁ (position):   {torch.exp(log_l1).detach().cpu().numpy()}")
    print(f"Final λ₂ (power law):  {torch.exp(log_l2).detach().cpu().numpy()}")
    print(f"Final λ₃ (jerk):       {torch.exp(log_l3).detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
