"""Visualization functions for trajectory prediction."""

import numpy as np
import matplotlib.pyplot as plt
import torch


def _set_axes_equal_3d(ax):
    """Set equal aspect ratio for 3D plots."""
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xmid, ymid, zmid = np.mean(xs), np.mean(ys), np.mean(zs)
    max_range = max(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]) / 2.0
    ax.set_xlim3d([xmid - max_range, xmid + max_range])
    ax.set_ylim3d([ymid - max_range, ymid + max_range])
    ax.set_zlim3d([zmid - max_range, zmid + max_range])


def plot_stepwise_errors(pred_pos, target_pos, title_suffix=""):
    """
    Plots error metrics over the prediction horizon.

    Args:
        pred_pos: (Batch, Horizon, 3) numpy array [Meters]
        target_pos: (Batch, Horizon, 3) numpy array [Meters]
        title_suffix: Optional suffix for plot titles
    """
    # 1. Setup Data - Convert to mm
    diff = pred_pos - target_pos
    diff_mm = diff * 1000

    # Time steps (1 to Horizon)
    H = pred_pos.shape[1]
    t = np.arange(1, H + 1)

    # 2. Calculate Metrics per Step (averaging over batch)
    mae_t = np.mean(np.abs(diff_mm), axis=0)
    rmse_t = np.sqrt(np.mean(diff_mm ** 2, axis=0))

    # Euclidean Error (Scalar distance per step)
    dist = np.linalg.norm(diff_mm, axis=2)
    mean_eucl = np.mean(dist, axis=0)
    std_eucl = np.std(dist, axis=0)

    # --- PLOT 1: Component-wise Error (MAE & RMSE) ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']

    # MAE
    for j in range(3):
        axs[0].plot(t, mae_t[:, j], color=colors[j], label=labels[j], linewidth=2)
    axs[0].set_ylabel("MAE [mm]")
    axs[0].set_title(f"Mean Absolute Error per Step {title_suffix}")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # RMSE
    for j in range(3):
        axs[1].plot(t, rmse_t[:, j], color=colors[j], label=labels[j], linewidth=2)
    axs[1].set_ylabel("RMSE [mm]")
    axs[1].set_xlabel("Prediction Step (Horizon)")
    axs[1].set_title("Root Mean Square Error per Step")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Total Euclidean Distance with Std Dev ---
    fig2 = plt.figure(figsize=(8, 5))

    plt.plot(t, mean_eucl, 'k-', linewidth=2, label="Mean Euclidean Dist")
    plt.fill_between(t,
                     np.maximum(0, mean_eucl - std_eucl),
                     mean_eucl + std_eucl,
                     color='gray', alpha=0.3, label="± 1 Std. Dev.")

    plt.xlabel("Prediction Step")
    plt.ylabel("Euclidean Error [mm]")
    plt.title(f"Trajectory Error Evolution {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_val_trajs_3d_and_xyz(model, valid_loader, valid_data, window_size, horizon,
                               device, num_examples=3, indices=None, save=False,
                               outdir="plots", seed=0):
    """
    Plot 3D trajectories and XYZ components over time.

    Args:
        model: Trained model
        valid_loader: DataLoader for validation data
        valid_data: Raw validation data array
        window_size: Input window size
        horizon: Prediction horizon
        device: torch device
        num_examples: Number of examples to plot
        indices: Specific indices to plot (optional)
        save: Whether to save plots
        outdir: Output directory for saved plots
        seed: Random seed for reproducibility
    """
    # --- FONT CONFIGURATION ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'

    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Fetch one batch
    with torch.no_grad():
        x_val, y_val = next(iter(valid_loader))
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        full_pred = model(x_val, y_val, teacher_forcing_ratio=0.0)
        pred_pos = full_pred[:, :, :3]

    pred_pos_np = pred_pos.detach().cpu().numpy()
    valid_np = np.asarray(valid_data)
    B = pred_pos_np.shape[0]

    # 2. Select Indices
    if indices is not None:
        indices = [i for i in indices if i < B]
        if not indices:
            print(f"Warning: Provided indices out of range for batch size {B}. Using default.")
            indices = np.linspace(0, B - 1, num_examples, dtype=int)
    else:
        indices = np.linspace(0, B - 1, num_examples, dtype=int)

    # 3. 3D Plot Loop
    K = len(indices)
    fig3d = plt.figure(figsize=(6 * K, 6))
    axs3d = [fig3d.add_subplot(1, K, j + 1, projection='3d') for j in range(K)]

    for j, i in enumerate(indices):
        ax3d = axs3d[j]

        # --- CONVERT TO MM ---
        hist_pos = valid_np[i, :window_size, :3] * 1000
        true_full = valid_np[i, window_size: window_size + horizon, :3] * 1000
        pred_future = pred_pos_np[i, :horizon, :] * 1000
        anchor = hist_pos[-1]

        # --- PLOT 3D ---
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

    plt.suptitle('Validation Trajectories [mm]', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 4. XYZ vs Time Plot
    for i in indices:
        hist_pos = valid_np[i, :window_size, :3] * 1000
        true_full = valid_np[i, window_size: window_size + horizon, :3] * 1000
        pred_future = pred_pos_np[i, :horizon, :] * 1000
        anchor = hist_pos[-1]

        t_hist = np.arange(window_size)
        t_fut = np.arange(window_size, window_size + horizon)
        t_anchor = np.array([window_size - 1])

        fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        labels = ['X', 'Y', 'Z']

        for ax, dim, lab in zip(axs, range(3), labels):
            ax.scatter(t_hist, hist_pos[:, dim], s=12, c='k', label='History' if dim == 0 else "")
            ax.scatter(t_anchor, [anchor[dim]], s=40, c='teal', edgecolor='k', marker='D',
                       label='Anchor' if dim == 0 else "")
            ax.scatter(t_fut, true_full[:, dim], s=12, c='rebeccapurple', label='True' if dim == 0 else "")
            ax.scatter(t_fut, pred_future[:, dim], s=12, c='deeppink', marker='^', label='Pred' if dim == 0 else "")
            ax.set_ylabel(f'{lab} [mm]')
            ax.grid(True)

        axs[-1].set_xlabel('Time Step')
        axs[0].legend(ncol=4, fontsize=9)
        fig.suptitle(f'Trajectory {i} Components [mm]', fontweight='bold')
        plt.tight_layout()
        plt.show()
