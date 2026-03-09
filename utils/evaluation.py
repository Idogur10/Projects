"""Evaluation functions for trajectory prediction."""

import numpy as np
import torch


def evaluate_avg_mae_rmse(model, data_loader, device):
    """
    Calculates average MAE and RMSE over all trajectories and all timesteps.
    Positions are converted from meters to millimeters.

    Args:
        model: Trained model
        data_loader: DataLoader
        device: torch device

    Returns:
        dict with 'mae' and 'rmse' per axis and total
    """
    model.eval()
    all_abs_errors = []
    all_sq_errors = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch, teacher_forcing_ratio=0.0)

            # Convert to mm
            pred_pos_mm = pred[:, :, :3] * 1000
            true_pos_mm = y_batch[:, :, :3] * 1000

            diff = pred_pos_mm - true_pos_mm

            # Absolute error: (batch, horizon, 3)
            all_abs_errors.append(torch.abs(diff).cpu())
            # Squared error: (batch, horizon, 3)
            all_sq_errors.append((diff ** 2).cpu())

    # Concatenate all batches: (N, horizon, 3)
    all_abs_errors = torch.cat(all_abs_errors, dim=0)
    all_sq_errors = torch.cat(all_sq_errors, dim=0)

    # Average over all samples and all timesteps: (3,)
    mae_per_axis = all_abs_errors.mean(dim=(0, 1)).numpy()
    mse_per_axis = all_sq_errors.mean(dim=(0, 1)).numpy()
    rmse_per_axis = np.sqrt(mse_per_axis)

    # Total MAE/RMSE (average across axes)
    mae_total = mae_per_axis.mean()
    rmse_total = np.sqrt(mse_per_axis.mean())

    # Print results
    print(f"\n{'='*50}")
    print(f"Average MAE / RMSE over all trajectories (mm)")
    print(f"{'='*50}")
    axes_labels = ['X', 'Y', 'Z']
    print(f"{'Axis':<6} | {'MAE (mm)':<14} | {'RMSE (mm)':<12}")
    print(f"{'-'*40}")
    for i, ax in enumerate(axes_labels):
        print(f"{ax:<6} | {mae_per_axis[i]:<14.4f} | {rmse_per_axis[i]:<12.4f}")
    print(f"{'-'*40}")
    print(f"{'Total':<6} | {mae_total:<14.4f} | {rmse_total:<12.4f}")
    print(f"{'='*50}")

    return {
        'mae_per_axis': mae_per_axis,
        'rmse_per_axis': rmse_per_axis,
        'mae_total': mae_total,
        'rmse_total': rmse_total,
    }


def evaluate_at_timestamps(model, valid_loader, device, steps=[10, 20, 30, 40, 50]):
    """
    Calculates axis-wise MAE, RMSE, and total Euclidean Distance.
    Input data is in Meters -> Output is converted to Millimeters (mm).

    Args:
        model: Trained model
        valid_loader: DataLoader for validation data
        device: torch device
        steps: List of timesteps to evaluate at
    """
    model.eval()
    indices = [s - 1 for s in steps]

    # Store results for each step
    metrics = {
        s: {'mae_axes': [], 'mse_axes': [], 'l2_dist': []}
        for s in steps
    }

    with torch.no_grad():
        for x_val, y_val in valid_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pred = model(x_val, y_val, teacher_forcing_ratio=0.0)

            # CONVERSION: Meters to Millimeters
            pred_pos_mm = pred[:, :, :3] * 1000
            true_pos_mm = y_val[:, :, :3] * 1000

            for s, idx in zip(steps, indices):
                p_s = pred_pos_mm[:, idx, :]
                t_s = true_pos_mm[:, idx, :]

                diff = p_s - t_s

                # 1. MAE per axis
                mae_axes = torch.abs(diff).mean(dim=0).cpu().numpy()

                # 2. MSE per axis (Needed to calculate RMSE later)
                mse_axes = torch.pow(diff, 2).mean(dim=0).cpu().numpy()

                # 3. Total Euclidean Distance (L2)
                l2_dist = torch.norm(diff, p=2, dim=1).mean().item()

                metrics[s]['mae_axes'].append(mae_axes)
                metrics[s]['mse_axes'].append(mse_axes)
                metrics[s]['l2_dist'].append(l2_dist)

    # --- Print Header ---
    print(f"\n{'Step':<5} | {'Axis':<2} | {'MAE (mm)':<10} | {'RMSE (mm)':<10} | {'L2 Dist (mm)':<12}")
    print("-" * 60)

    for s in steps:
        # Average the metrics across all batches
        avg_mae = np.mean(metrics[s]['mae_axes'], axis=0)
        avg_mse = np.mean(metrics[s]['mse_axes'], axis=0)

        # CALCULATE RMSE: The square root of the average MSE
        avg_rmse = np.sqrt(avg_mse)

        avg_l2 = np.mean(metrics[s]['l2_dist'])

        axes_labels = ['X', 'Y', 'Z']
        for i in range(3):
            l2_str = f"{avg_l2:.4f}" if i == 1 else ""
            step_str = f"{s}" if i == 0 else ""

            print(f"{step_str:<5} | {axes_labels[i]:<2} | {avg_mae[i]:<10.4f} | {avg_rmse[i]:<10.4f} | {l2_str:<12}")
        print("-" * 60)
