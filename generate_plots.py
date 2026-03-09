"""Generate and save evaluation plots from pre-trained models."""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (
    HIDDEN_DIM, BATCH_SIZE, WINDOW_SIZE, HORIZON, DELTA_T, OUTPUT_SIZE,
    DIRECTORY, DOWNSAMPLE_FACTOR, RANDOM_SEED
)
from data import TrajectoryDataset, preprocessing, compute_curv_torsion_features
from models import Seq2SeqDaVinciNet, Seq2SeqDaVinciNetVelAcc
from utils import ade_loss, evaluate_at_timestamps, evaluate_avg_mae_rmse

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(SAVE_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def downsample_data(data, factor):
    return data[:, ::factor, :]


def _set_axes_equal_3d(ax):
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    max_range = max(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]) / 2.0
    ax.set_xlim3d([xmid - max_range, xmid + max_range])
    ax.set_ylim3d([ymid - max_range, ymid + max_range])
    ax.set_zlim3d([zmid - max_range, zmid + max_range])


def save_stepwise_errors(pred_pos, target_pos, prefix=""):
    """Save MAE/RMSE and Euclidean error plots."""
    diff = pred_pos - target_pos
    diff_mm = diff * 1000

    H = pred_pos.shape[1]
    t = np.arange(1, H + 1)

    mae_t = np.mean(np.abs(diff_mm), axis=0)
    rmse_t = np.sqrt(np.mean(diff_mm ** 2, axis=0))

    dist = np.linalg.norm(diff_mm, axis=2)
    mean_eucl = np.mean(dist, axis=0)
    std_eucl = np.std(dist, axis=0)

    # Plot 1: MAE & RMSE
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']

    for j in range(3):
        axs[0].plot(t, mae_t[:, j], color=colors[j], label=labels[j], linewidth=2)
    axs[0].set_ylabel("MAE [mm]")
    axs[0].set_title(f"Mean Absolute Error per Step ({prefix})")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    for j in range(3):
        axs[1].plot(t, rmse_t[:, j], color=colors[j], label=labels[j], linewidth=2)
    axs[1].set_ylabel("RMSE [mm]")
    axs[1].set_xlabel("Prediction Step (100ms each)")
    axs[1].set_title("Root Mean Square Error per Step")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    fname = os.path.join(SAVE_DIR, f'{prefix.lower().replace(" ", "_")}_mae_rmse.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # Plot 2: Euclidean Error
    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(t, mean_eucl, 'k-', linewidth=2, label="Mean Euclidean Dist")
    plt.fill_between(t, np.maximum(0, mean_eucl - std_eucl),
                     mean_eucl + std_eucl, color='gray', alpha=0.3, label="\u00b1 1 Std. Dev.")
    plt.xlabel("Prediction Step (100ms each)")
    plt.ylabel("Euclidean Error [mm]")
    plt.title(f"Trajectory Error Evolution ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fname2 = os.path.join(SAVE_DIR, f'{prefix.lower().replace(" ", "_")}_euclidean_error.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname2}")


def save_trajectory_plots(model, test_loader, test_data, prefix="", num_examples=3):
    """Save 3D trajectory and XYZ component plots."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        x_val, y_val = next(iter(test_loader))
        x_val, y_val = x_val.to(device), y_val.to(device)
        full_pred = model(x_val, y_val, teacher_forcing_ratio=0.0)
        pred_pos = full_pred[:, :, :3]

    pred_pos_np = pred_pos.detach().cpu().numpy()
    valid_np = np.asarray(test_data)
    B = pred_pos_np.shape[0]
    indices = np.linspace(0, B - 1, num_examples, dtype=int)

    # 3D Plot
    K = len(indices)
    fig3d = plt.figure(figsize=(6 * K, 6))
    axs3d = [fig3d.add_subplot(1, K, j + 1, projection='3d') for j in range(K)]

    for j, i in enumerate(indices):
        ax3d = axs3d[j]
        hist_pos = valid_np[i, :WINDOW_SIZE, :3] * 1000
        true_full = valid_np[i, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3] * 1000
        pred_future = pred_pos_np[i, :HORIZON, :] * 1000
        anchor = hist_pos[-1]

        ax3d.scatter(hist_pos[:, 0], hist_pos[:, 1], hist_pos[:, 2],
                     s=12, c='k', marker='o', alpha=0.6, label='History')
        ax3d.scatter(true_full[:, 0], true_full[:, 1], true_full[:, 2],
                     s=20, c='rebeccapurple', marker='o', label='True Future')
        ax3d.scatter(pred_future[:, 0], pred_future[:, 1], pred_future[:, 2],
                     s=20, c='deeppink', marker='^', label='Pred Future')
        ax3d.scatter(anchor[0], anchor[1], anchor[2],
                     s=60, c='teal', marker='D', edgecolor='white', linewidth=1.5, label='Anchor')
        ax3d.set_title(f"Trajectory {i} (mm)")
        ax3d.set_xlabel("X [mm]")
        ax3d.set_ylabel("Y [mm]")
        ax3d.set_zlabel("Z [mm]")
        try:
            _set_axes_equal_3d(ax3d)
        except:
            pass
        if j == 0:
            ax3d.legend(loc='best', fontsize=8)

    plt.suptitle(f'Test Trajectories - {prefix} [mm]', fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(SAVE_DIR, f'{prefix.lower().replace(" ", "_")}_3d_trajectories.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    # XYZ Component Plot (first trajectory only)
    i = indices[0]
    hist_pos = valid_np[i, :WINDOW_SIZE, :3] * 1000
    true_full = valid_np[i, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3] * 1000
    pred_future = pred_pos_np[i, :HORIZON, :] * 1000
    anchor = hist_pos[-1]

    t_hist = np.arange(WINDOW_SIZE)
    t_fut = np.arange(WINDOW_SIZE, WINDOW_SIZE + HORIZON)
    t_anchor = np.array([WINDOW_SIZE - 1])

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axis_labels = ['X', 'Y', 'Z']

    for ax, dim, lab in zip(axs, range(3), axis_labels):
        ax.scatter(t_hist, hist_pos[:, dim], s=12, c='k', label='History' if dim == 0 else "")
        ax.scatter(t_anchor, [anchor[dim]], s=40, c='teal', edgecolor='k', marker='D',
                   label='Anchor' if dim == 0 else "")
        ax.scatter(t_fut, true_full[:, dim], s=12, c='rebeccapurple', label='True' if dim == 0 else "")
        ax.scatter(t_fut, pred_future[:, dim], s=12, c='deeppink', marker='^', label='Pred' if dim == 0 else "")
        ax.set_ylabel(f'{lab} [mm]')
        ax.grid(True)

    axs[-1].set_xlabel('Time Step (10Hz)')
    axs[0].legend(ncol=4, fontsize=9)
    fig.suptitle(f'Trajectory Components - {prefix} [mm]', fontweight='bold')
    plt.tight_layout()
    fname2 = os.path.join(SAVE_DIR, f'{prefix.lower().replace(" ", "_")}_xyz_components.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname2}")


def evaluate_model(model_name, model, test_loader, test_data, prefix):
    """Run full evaluation and save plots."""
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    # Print metrics
    evaluate_at_timestamps(model, test_loader, device, steps=[1, 2, 3, 4, 5])
    results = evaluate_avg_mae_rmse(model, test_loader, device)

    # Generate predictions for plots
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_v, y_v in test_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            out_v = model(x_v, teacher_forcing_ratio=0.0)
            all_preds.append(out_v[:, :, :3].cpu().numpy())
            all_targets.append(y_v[:, :, :3].cpu().numpy())

    final_preds = np.concatenate(all_preds, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)

    # Save plots
    save_stepwise_errors(final_preds, final_targets, prefix=prefix)
    save_trajectory_plots(model, test_loader, test_data, prefix=prefix)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # MODEL 1: Baseline DaVinciNet (13-dim input, Verlet integration)
    # =========================================================================
    print("\n" + "="*60)
    print("Loading Baseline DaVinciNet model...")
    print("="*60)

    # Load and preprocess training data (to get scalers)
    train_data = np.load(os.path.join(DIRECTORY, "No_B_legnth_250_step_size50vert_hor_WithoutB.npy"))
    train_data = downsample_data(train_data, DOWNSAMPLE_FACTOR)
    train_inputs = train_data[:, :WINDOW_SIZE, :]
    train_inputs, vel_scaler, pos_mean, pos_std = preprocessing(train_inputs)

    # Build and load model
    model_baseline = Seq2SeqDaVinciNet(
        input_size=13, hidden_size=HIDDEN_DIM, seq_len=WINDOW_SIZE,
        horizon=HORIZON, delta_t=DELTA_T, pos_mean=pos_mean, pos_std=pos_std,
        scaler_vel=vel_scaler
    ).to(device)

    baseline_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    if not os.path.exists(baseline_path):
        print(f"WARNING: {baseline_path} not found, skipping baseline model.")
        baseline_results = None
    else:
        model_baseline.load_state_dict(torch.load(baseline_path, map_location=device))

        # Load test data
        test_data_base = np.load(os.path.join(DIRECTORY, "No_U_legnth_250_step_size50vert_horA_test.npy"))
        test_data_base = downsample_data(test_data_base, DOWNSAMPLE_FACTOR)

        test_inputs = test_data_base[:, :WINDOW_SIZE, :]
        test_inputs = preprocessing(test_inputs, vel_scaler, pos_mean, pos_std)
        test_targets = test_data_base[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3]

        test_set = TrajectoryDataset(test_inputs, test_targets)
        test_loader = DataLoader(test_set, batch_size=len(test_set))

        baseline_results = evaluate_model("Baseline DaVinciNet", model_baseline,
                                          test_loader, test_data_base, "Baseline")

    # =========================================================================
    # MODEL 2: DaVinciNet + Curvature/Torsion (15-dim input)
    # =========================================================================
    print("\n" + "="*60)
    print("Loading DaVinciNet + Curvature model...")
    print("="*60)

    # Reload train data with curvature features
    train_data_c = np.load(os.path.join(DIRECTORY, "No_B_legnth_250_step_size50vert_hor_WithoutB.npy"))
    train_data_c = downsample_data(train_data_c, DOWNSAMPLE_FACTOR)
    train_data_c = compute_curv_torsion_features(train_data_c, DELTA_T)
    train_inputs_c = train_data_c[:, :WINDOW_SIZE, :]
    train_inputs_c, vel_scaler_c, pos_mean_c, pos_std_c, curv_scaler_c = preprocessing(train_inputs_c)

    model_curv = Seq2SeqDaVinciNet(
        input_size=15, hidden_size=HIDDEN_DIM, seq_len=WINDOW_SIZE,
        horizon=HORIZON, delta_t=DELTA_T, pos_mean=pos_mean_c, pos_std=pos_std_c,
        scaler_vel=vel_scaler_c
    ).to(device)

    curv_path = os.path.join(os.path.dirname(__file__), 'best_model_curvature.pth')
    model_curv.load_state_dict(torch.load(curv_path, map_location=device))

    # Test data with curvature
    test_data_curv = np.load(os.path.join(DIRECTORY, "No_U_legnth_250_step_size50vert_horA_test.npy"))
    test_data_curv = downsample_data(test_data_curv, DOWNSAMPLE_FACTOR)
    test_data_curv = compute_curv_torsion_features(test_data_curv, DELTA_T)

    test_inputs_c = test_data_curv[:, :WINDOW_SIZE, :]
    test_inputs_c = preprocessing(test_inputs_c, vel_scaler_c, pos_mean_c, pos_std_c, curv_scaler_c)
    test_targets_c = test_data_curv[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3]

    test_set_c = TrajectoryDataset(test_inputs_c, test_targets_c)
    test_loader_c = DataLoader(test_set_c, batch_size=len(test_set_c))

    curv_results = evaluate_model("DaVinciNet + Curvature/Torsion", model_curv,
                                  test_loader_c, test_data_curv, "Curvature")

    # =========================================================================
    # MODEL 3: DaVinciNet VelAcc (3rd-order Taylor + Jerk)
    # =========================================================================
    print("\n" + "="*60)
    print("Loading DaVinciNet VelAcc model...")
    print("="*60)

    model_velacc = Seq2SeqDaVinciNetVelAcc(
        input_size=15, hidden_size=HIDDEN_DIM, seq_len=WINDOW_SIZE,
        horizon=HORIZON, delta_t=DELTA_T, pos_mean=pos_mean_c, pos_std=pos_std_c,
        scaler_vel=vel_scaler_c
    ).to(device)

    velacc_path = os.path.join(os.path.dirname(__file__), 'best_model_vel_acc.pth')
    model_velacc.load_state_dict(torch.load(velacc_path, map_location=device))

    test_loader_va = DataLoader(test_set_c, batch_size=len(test_set_c))

    velacc_results = evaluate_model("DaVinciNet VelAcc (3rd-order Taylor)", model_velacc,
                                    test_loader_va, test_data_curv, "VelAcc")

    # =========================================================================
    # COMPARISON PLOT
    # =========================================================================
    print("\n" + "="*60)
    print("Generating comparison plot...")
    print("="*60)

    # Collect all results for comparison
    model_names = []
    mae_vals = []
    rmse_vals = []

    if baseline_results:
        model_names.append("Baseline\n(Verlet)")
        mae_vals.append(baseline_results['mae_total'])
        rmse_vals.append(baseline_results['rmse_total'])

    model_names.append("+ Curvature\n& Torsion")
    mae_vals.append(curv_results['mae_total'])
    rmse_vals.append(curv_results['rmse_total'])

    model_names.append("VelAcc\n(3rd-order Taylor)")
    mae_vals.append(velacc_results['mae_total'])
    rmse_vals.append(velacc_results['rmse_total'])

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, mae_vals, width, label='MAE (mm)', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, rmse_vals, width, label='RMSE (mm)', color='coral', edgecolor='black')

    ax.set_ylabel('Error (mm)', fontweight='bold')
    ax.set_title('Model Comparison - Test Set Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fname = os.path.join(SAVE_DIR, 'model_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

    print(f"\nAll plots saved to: {SAVE_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
