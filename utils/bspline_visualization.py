"""Visualization utilities for B-spline trajectory fitting."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_comparison(R_U, R_TRUE, R_C, idx, title_suffix=""):
    """
    Plot comparison of LSTM prediction, ground truth, and B-spline fit for a single trajectory.

    Args:
        R_U: LSTM predictions (H, 3)
        R_TRUE: Ground truth (H, 3)
        R_C: B-spline fitted trajectory (H, 3)
        idx: Trajectory index
        title_suffix: Optional suffix for plot title
    """
    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(R_TRUE[:, 0], R_TRUE[:, 1], R_TRUE[:, 2],
             'go-', linewidth=2, markersize=8, label='R_TRUE (Ground Truth)', alpha=0.7)
    ax1.plot(R_U[:, 0], R_U[:, 1], R_U[:, 2],
             'bs-', linewidth=2, markersize=6, label='R_U (LSTM)', alpha=0.7)
    ax1.plot(R_C[:, 0], R_C[:, 1], R_C[:, 2],
             'r^-', linewidth=2, markersize=6, label='R_C (B-spline Fit)', alpha=0.7)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Trajectory {idx} - 3D View{title_suffix}')
    ax1.legend()
    ax1.grid(True)

    # X-Y projection
    ax2 = fig.add_subplot(132)
    ax2.plot(R_TRUE[:, 0], R_TRUE[:, 1], 'go-', linewidth=2, markersize=8, label='R_TRUE', alpha=0.7)
    ax2.plot(R_U[:, 0], R_U[:, 1], 'bs-', linewidth=2, markersize=6, label='R_U (LSTM)', alpha=0.7)
    ax2.plot(R_C[:, 0], R_C[:, 1], 'r^-', linewidth=2, markersize=6, label='R_C (B-spline)', alpha=0.7)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title(f'X-Y Projection{title_suffix}')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Time series (all dimensions)
    ax3 = fig.add_subplot(133)
    timesteps = np.arange(len(R_TRUE))

    # Plot each dimension separately
    for dim, dim_name, color in [(0, 'X', 'r'), (1, 'Y', 'g'), (2, 'Z', 'b')]:
        ax3.plot(timesteps, R_TRUE[:, dim], f'{color}o-', linewidth=2,
                label=f'TRUE {dim_name}', alpha=0.5, markersize=6)
        ax3.plot(timesteps, R_U[:, dim], f'{color}s--', linewidth=1.5,
                label=f'LSTM {dim_name}', alpha=0.7, markersize=4)
        ax3.plot(timesteps, R_C[:, dim], f'{color}^:', linewidth=1.5,
                label=f'B-spline {dim_name}', alpha=0.7, markersize=4)

    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Position (mm)')
    ax3.set_title(f'Time Series{title_suffix}')
    ax3.legend(ncol=3, fontsize=8)
    ax3.grid(True)

    plt.tight_layout()
    return fig


def plot_multiple_trajectories(R_U_batch, R_TRUE_batch, R_C_batch, num_examples=3):
    """
    Plot comparison for multiple trajectories.

    Args:
        R_U_batch: LSTM predictions (batch, H, 3)
        R_TRUE_batch: Ground truth (batch, H, 3)
        R_C_batch: B-spline fitted trajectories (batch, H, 3)
        num_examples: Number of trajectories to plot
    """
    num_examples = min(num_examples, len(R_U_batch))

    for i in range(num_examples):
        fig = plot_trajectory_comparison(
            R_U_batch[i],
            R_TRUE_batch[i],
            R_C_batch[i],
            idx=i
        )
        plt.savefig(f'trajectory_comparison_{i}.png', dpi=150, bbox_inches='tight')
        print(f"Saved trajectory_comparison_{i}.png")

    # Also create a summary figure with all 3 in one
    fig = plt.figure(figsize=(18, 5 * num_examples))

    for i in range(num_examples):
        # 3D plot for trajectory i
        ax = fig.add_subplot(num_examples, 3, i * 3 + 1, projection='3d')
        ax.plot(R_TRUE_batch[i, :, 0], R_TRUE_batch[i, :, 1], R_TRUE_batch[i, :, 2],
                'go-', linewidth=2, markersize=8, label='Ground Truth', alpha=0.7)
        ax.plot(R_U_batch[i, :, 0], R_U_batch[i, :, 1], R_U_batch[i, :, 2],
                'bs-', linewidth=2, markersize=6, label='LSTM', alpha=0.7)
        ax.plot(R_C_batch[i, :, 0], R_C_batch[i, :, 1], R_C_batch[i, :, 2],
                'r^-', linewidth=2, markersize=6, label='B-spline', alpha=0.7)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Trajectory {i} - 3D')
        if i == 0:
            ax.legend()
        ax.grid(True)

        # X-Y projection
        ax = fig.add_subplot(num_examples, 3, i * 3 + 2)
        ax.plot(R_TRUE_batch[i, :, 0], R_TRUE_batch[i, :, 1], 'go-', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(R_U_batch[i, :, 0], R_U_batch[i, :, 1], 'bs-', linewidth=2, markersize=6, alpha=0.7)
        ax.plot(R_C_batch[i, :, 0], R_C_batch[i, :, 1], 'r^-', linewidth=2, markersize=6, alpha=0.7)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'X-Y Projection')
        ax.grid(True)
        ax.axis('equal')

        # Time series
        ax = fig.add_subplot(num_examples, 3, i * 3 + 3)
        timesteps = np.arange(R_TRUE_batch.shape[1])
        for dim, dim_name, color in [(0, 'X', 'r'), (1, 'Y', 'g'), (2, 'Z', 'b')]:
            ax.plot(timesteps, R_TRUE_batch[i, :, dim], f'{color}o-', linewidth=2, alpha=0.5, markersize=6)
            ax.plot(timesteps, R_U_batch[i, :, dim], f'{color}s--', linewidth=1.5, alpha=0.7, markersize=4)
            ax.plot(timesteps, R_C_batch[i, :, dim], f'{color}^:', linewidth=1.5, alpha=0.7, markersize=4)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position (mm)')
        ax.set_title(f'Time Series')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_comparison_summary.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory_comparison_summary.png")

    return fig


def plot_inner_losses(inner_losses_per_trajectory, epoch):
    """
    Plot the inner loop (B-spline approximation) losses for each trajectory.

    Args:
        inner_losses_per_trajectory: List of losses per trajectory (batch,)
        epoch: Current epoch number
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(inner_losses_per_trajectory)), inner_losses_per_trajectory,
           color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Inner Loop Loss (B-spline Approximation Error)')
    ax.set_title(f'Per-Trajectory B-spline Fitting Loss at Epoch {epoch}')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_loss = np.mean(inner_losses_per_trajectory)
    ax.axhline(mean_loss, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_loss:.2f}')
    ax.legend()

    plt.tight_layout()
    return fig
