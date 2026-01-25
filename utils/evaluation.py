"""Evaluation functions for trajectory prediction."""

import numpy as np
import torch


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
