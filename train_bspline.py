"""
Bi-level optimization for B-spline trajectory fitting with motor invariant constraints.

This script implements a meta-learning approach where:
- Inner loop: For each trajectory, optimize control points C to minimize position error,
  power law error, and jerk, weighted by hyperparameters λ₁, λ₂, λ₃
- Outer loop: Optimize hyperparameters λ to minimize trajectory reconstruction error
  across all trajectories

The optimization maintains differentiable gradient flow from outer loss through inner
optimization steps back to hyperparameters using create_graph=True.
"""

import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from config import (
    HIDDEN_DIM, RANDOM_SEED, DIRECTORY, WINDOW_SIZE, HORIZON, DELTA_T,
    DOWNSAMPLE_FACTOR, BSPLINE_BATCH_SIZE, BSPLINE_INNER_STEPS, BSPLINE_INNER_LR,
    BSPLINE_OUTER_LR, BSPLINE_N_EPOCHS, BSPLINE_K, BSPLINE_DEGREE,
    LAMBDA_1_INIT, LAMBDA_2_INIT, LAMBDA_3_INIT,
    LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, LR_MIN
)
from data import preprocessing, downsample_data
from models import Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM, ModelCopt
from utils import initialize_bspline_matrices, get_parameters_per_sample, plot_multiple_trajectories, plot_inner_losses


def main():
    # === Setup ===
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Downsampling: 100Hz -> {100 // DOWNSAMPLE_FACTOR}Hz (factor={DOWNSAMPLE_FACTOR})")
    print(f"Horizon: {HORIZON} timesteps, Delta_t: {DELTA_T}s")
    print(f"Bi-level optimization config:")
    print(f"  Batch size: {BSPLINE_BATCH_SIZE}")
    print(f"  Inner steps: {BSPLINE_INNER_STEPS}, Inner LR: {BSPLINE_INNER_LR}")
    print(f"  Outer LR: {BSPLINE_OUTER_LR}, Epochs: {BSPLINE_N_EPOCHS}")
    print(f"  Control points (K): {BSPLINE_K}, Degree: {BSPLINE_DEGREE}\n")

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

    # === Load LSTM model ===
    print("Loading pre-trained LSTM model...")
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
        input_tensor = torch.tensor(train_inputs_raw[:BSPLINE_BATCH_SIZE], dtype=torch.float32).to(device)
        R_U = lstm_model(input_tensor)[:, :, :3]  # positions only (batch, H, 3)

    # === Extract power law parameters ===
    print("Extracting power law parameters (alpha, beta) per trajectory...")
    velocity_input = train_data[:BSPLINE_BATCH_SIZE, WINDOW_SIZE:WINDOW_SIZE + HORIZON, 3:6]
    alphas, betas = get_parameters_per_sample(velocity_input, DELTA_T)

    # Validate and fix NaN/inf
    print(f"Alpha range: [{np.nanmin(alphas):.4f}, {np.nanmax(alphas):.4f}], mean: {np.nanmean(alphas):.4f}")
    print(f"Beta range: [{np.nanmin(betas):.4f}, {np.nanmax(betas):.4f}], mean: {np.nanmean(betas):.4f}")
    print(f"NaN count - alphas: {np.sum(np.isnan(alphas))}, betas: {np.sum(np.isnan(betas))}")

    # Replace NaN with default values
    alphas = np.nan_to_num(alphas, nan=1.0)
    betas = np.nan_to_num(betas, nan=-0.33)

    alphas_tensor = torch.tensor(alphas, dtype=torch.float32).to(device)
    betas_tensor = torch.tensor(betas, dtype=torch.float32).to(device)

    # === Initialize B-spline matrices ===
    print(f"Initializing B-spline matrices (H={HORIZON}, K={BSPLINE_K}, degree={BSPLINE_DEGREE})...")
    B_matrices, B_pinv = initialize_bspline_matrices(HORIZON, BSPLINE_K, BSPLINE_DEGREE, device=device)

    # Ground truth control points
    C_TRUE = torch.matmul(B_pinv, R_TRUE_tensor[:BSPLINE_BATCH_SIZE])  # (batch, K, 3)

    # === Initialize ModelCopt ===
    model_copt = ModelCopt(B_matrices, B_pinv, HORIZON, BSPLINE_K, device=device).to(device)

    # === Initialize hyperparameters (λ₁, λ₂, λ₃) ===
    # Using log-space for positivity constraint: exp(log_l) = l
    log_l1 = torch.full((HORIZON,), math.log(LAMBDA_1_INIT), device=device, requires_grad=True)
    log_l2 = torch.full((HORIZON,), math.log(LAMBDA_2_INIT), device=device, requires_grad=True)
    log_l3 = torch.full((HORIZON,), math.log(LAMBDA_3_INIT), device=device, requires_grad=True)

    optimizer_lambda = optim.Adam([log_l1, log_l2, log_l3], lr=BSPLINE_OUTER_LR)

    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_lambda,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_MIN,
        verbose=True
    )

    # === Outer optimization loop ===
    print("\n" + "=" * 70)
    print("STARTING BI-LEVEL OPTIMIZATION (MAML-like Meta-Learning)")
    print("=" * 70)
    print("\nGoal: Find hyperparameters (lambda1, lambda2, lambda3) that enable")
    print("      good B-spline approximations across ALL trajectories")
    print("      - Inner loop: Fit B-splines per trajectory using current lambdas")
    print("      - Outer loop: Optimize lambdas to minimize trajectory error")
    print("=" * 70)

    # Loss tracking
    loss_history = []
    best_loss = float('inf')
    no_improve_count = 0
    patience = 20  # Stop if no improvement for 20 epochs

    for epoch in range(BSPLINE_N_EPOCHS):
        optimizer_lambda.zero_grad()

        # Get current lambdas (exponentiate for positivity)
        l1 = torch.exp(log_l1)
        l2 = torch.exp(log_l2)
        l3 = torch.exp(log_l3)
        lambdas = (l1, l2, l3)

        # === Inner loop: optimize C_hat for each trajectory ===
        # Return inner losses every 50 epochs for monitoring
        return_losses = ((epoch + 1) % 50 == 0)
        if return_losses:
            C_hat, inner_losses_list = model_copt(
                R_U, alphas_tensor, betas_tensor, lambdas,
                inner_steps=BSPLINE_INNER_STEPS,
                inner_lr=BSPLINE_INNER_LR,
                return_inner_losses=True
            )
        else:
            C_hat = model_copt(
                R_U, alphas_tensor, betas_tensor, lambdas,
                inner_steps=BSPLINE_INNER_STEPS,
                inner_lr=BSPLINE_INNER_LR,
                return_inner_losses=False
            )

        # === Outer loss: trajectory-level loss (not control point loss) ===
        # Evaluate trajectories at the B-spline points
        P_true = torch.einsum('hk,bkd->bhd', model_copt.B, C_TRUE)   # (batch, H, 3)
        P_hat = torch.einsum('hk,bkd->bhd', model_copt.B, C_hat)     # (batch, H, 3)

        # ADE loss (Average Displacement Error)
        loss_outer = torch.mean(torch.norm(P_true - P_hat, dim=-1))

        # Backprop to update lambdas
        loss_outer.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([log_l1, log_l2, log_l3], 1.0)

        optimizer_lambda.step()

        # Track loss
        current_loss = loss_outer.item()
        loss_history.append(current_loss)

        # Update learning rate based on loss
        scheduler.step(current_loss)

        # Check for improvement
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer_lambda.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{BSPLINE_N_EPOCHS} | "
                  f"Loss: {current_loss:.6f} (best: {best_loss:.6f}) | "
                  f"LR: {current_lr:.2e} | "
                  f"L1: {l1.mean().item():.4f}, "
                  f"L2: {l2.mean().item():.4f}, "
                  f"L3: {l3.mean().item():.2e}")

        # Print lambda vectors and inner losses every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"  L1 vector: {l1.detach().cpu().numpy()}")
            print(f"  L2 vector: {l2.detach().cpu().numpy()}")
            print(f"  L3 vector: {l3.detach().cpu().numpy()}")
            if return_losses:
                inner_losses_np = np.array(inner_losses_list)
                print(f"  Inner losses (B-spline fit per traj): mean={inner_losses_np.mean():.2f}, "
                      f"std={inner_losses_np.std():.2f}, range=[{inner_losses_np.min():.2f}, {inner_losses_np.max():.2f}]")

        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}: no improvement for {patience} epochs")
            break

    # === Final results ===
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    final_l1 = torch.exp(log_l1).detach().cpu().numpy()
    final_l2 = torch.exp(log_l2).detach().cpu().numpy()
    final_l3 = torch.exp(log_l3).detach().cpu().numpy()

    print(f"\nFinal hyperparameters (per timestep):")
    print(f"  lambda1 (position):  mean={final_l1.mean():.4f}, std={final_l1.std():.4f}")
    print(f"  lambda2 (power law): mean={final_l2.mean():.4f}, std={final_l2.std():.4f}")
    print(f"  lambda3 (jerk):      mean={final_l3.mean():.2e}, std={final_l3.std():.2e}")

    print(f"\nFinal loss: {loss_outer.item():.6f} mm (best: {best_loss:.6f} mm)")
    print(f"Average trajectory error: {loss_outer.item():.2f} mm = {loss_outer.item()/1000:.3f} meters")
    print(f"Total epochs run: {len(loss_history)}")

    # Loss improvement summary
    if len(loss_history) > 1:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"\nLoss improvement: {initial_loss:.6f} -> {final_loss:.6f} ({improvement:.1f}% decrease)")

    # === Visualization ===
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Compute final fitted trajectories with the optimized lambdas
    # Run one more forward pass to get C_hat and inner losses
    final_lambdas = (torch.exp(log_l1.detach()), torch.exp(log_l2.detach()), torch.exp(log_l3.detach()))
    C_hat_final, final_inner_losses = model_copt(
        R_U, alphas_tensor, betas_tensor, final_lambdas,
        inner_steps=BSPLINE_INNER_STEPS,
        inner_lr=BSPLINE_INNER_LR,
        return_inner_losses=True
    )

    # Compute fitted trajectories R_C = B @ C_hat
    R_C = torch.einsum('hk,bkd->bhd', model_copt.B, C_hat_final.detach())

    # Convert to numpy for plotting
    R_U_np = R_U.detach().cpu().numpy()
    R_TRUE_np = R_TRUE_tensor[:BSPLINE_BATCH_SIZE].detach().cpu().numpy()
    R_C_np = R_C.detach().cpu().numpy()

    print(f"\nFinal inner losses (B-spline approximation quality per trajectory):")
    final_inner_losses_np = np.array(final_inner_losses)
    for i, loss_i in enumerate(final_inner_losses):
        print(f"  Trajectory {i}: {loss_i:.2f}")
    print(f"  Mean: {final_inner_losses_np.mean():.2f}, Std: {final_inner_losses_np.std():.2f}")

    # Plot inner losses
    fig_losses = plot_inner_losses(final_inner_losses, epoch=len(loss_history))
    plt.savefig('inner_losses_final.png', dpi=150, bbox_inches='tight')
    print("\nSaved inner_losses_final.png")
    plt.close(fig_losses)

    # Plot trajectory comparisons for 3 examples
    print("\nPlotting trajectory comparisons (R_U vs R_TRUE vs R_C)...")
    plot_multiple_trajectories(R_U_np, R_TRUE_np, R_C_np, num_examples=3)

    print("\n" + "=" * 70)
    print("COMPLETE! Check the following files:")
    print("  - trajectory_comparison_0.png")
    print("  - trajectory_comparison_1.png")
    print("  - trajectory_comparison_2.png")
    print("  - trajectory_comparison_summary.png")
    print("  - inner_losses_final.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
