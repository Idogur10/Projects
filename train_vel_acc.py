"""Training script with vel+acc decoder (3rd-order Taylor + jerk regularization)."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    HIDDEN_DIM, BATCH_SIZE, N_EPOCHS, LEARNING_RATE,
    RANDOM_SEED, DIRECTORY, WINDOW_SIZE, HORIZON, DELTA_T, OUTPUT_SIZE,
    W_POS_ADE, W_VEL_MSE, W_ACC_MSE, PATIENCE,
    DOWNSAMPLE_FACTOR, ORIGINAL_WINDOW_SIZE, ORIGINAL_HORIZON
)
from data import TrajectoryDataset, preprocessing, compute_curv_torsion_features
from models import Seq2SeqDaVinciNetVelAcc
from utils import ade_loss, evaluate_at_timestamps, evaluate_avg_mae_rmse, plot_stepwise_errors, plot_val_trajs_3d_and_xyz

INPUT_SIZE = 15  # 13 original + log-curvature + log-|torsion|
W_JERK = 1.0     # Jerk regularization weight


def downsample_data(data, factor):
    """Downsample trajectory data by taking every factor-th sample."""
    return data[:, ::factor, :]


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 60)
    print("Training: VEL+ACC DECODER (3rd-order Taylor + Jerk Reg.)")
    print("=" * 60)
    print(f"Downsampling: 100Hz -> {100 // DOWNSAMPLE_FACTOR}Hz (factor={DOWNSAMPLE_FACTOR})")
    print(f"Input window: {ORIGINAL_WINDOW_SIZE} -> {WINDOW_SIZE} timesteps")
    print(f"Horizon: {ORIGINAL_HORIZON} -> {HORIZON} timesteps")
    print(f"Delta_t: 0.01s -> {DELTA_T}s")
    print(f"Loss weights: pos={W_POS_ADE}, vel={W_VEL_MSE}, acc={W_ACC_MSE}, jerk={W_JERK}")
    print()

    # === Load Training Data ===
    filename = "No_B_legnth_250_step_size50vert_hor_WithoutB.npy"
    file_path = os.path.join(DIRECTORY, filename)
    train_data = np.load(file_path)
    print(f"Original train data shape: {train_data.shape}")

    # Downsample the data
    train_data = downsample_data(train_data, DOWNSAMPLE_FACTOR)
    print(f"Downsampled train data shape: {train_data.shape}")

    # Compute curvature + torsion features
    train_data = compute_curv_torsion_features(train_data, DELTA_T)
    print(f"After adding curvature+torsion: {train_data.shape}")

    # Prepare input features and targets
    train_inputs_raw = train_data[:, :WINDOW_SIZE, :]
    train_inputs_raw, vel_scaler, pos_mean, pos_std, curv_scaler = preprocessing(train_inputs_raw)

    # Targets: pos + vel, then compute acc via gradient
    train_targets_raw = train_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :OUTPUT_SIZE - 3]
    train_targets = train_targets_raw
    acceleration_target = np.gradient(train_targets[:, :, 3:6], DELTA_T, axis=1)
    train_targets = np.concatenate([train_targets, acceleration_target], axis=-1)

    print(f"Train inputs shape: {train_inputs_raw.shape}")
    print(f"Train targets shape: {train_targets.shape}")

    train_set = TrajectoryDataset(train_inputs_raw, train_targets)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # === Load Validation Data ===
    filename = "No_B_legnth_250_step_size50vert_hor_only_B.npy"
    file_path = os.path.join(DIRECTORY, filename)
    valid_data = np.load(file_path)

    # Downsample and add curvature
    valid_data = downsample_data(valid_data, DOWNSAMPLE_FACTOR)
    valid_data = compute_curv_torsion_features(valid_data, DELTA_T)

    valid_inputs_raw = valid_data[:, :WINDOW_SIZE, :]
    valid_inputs_raw = preprocessing(valid_inputs_raw, vel_scaler, pos_mean, pos_std, curv_scaler)

    # Targets: pos + vel + acc
    valid_targets_raw = valid_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :OUTPUT_SIZE - 3]
    acceleration_target = np.gradient(valid_targets_raw[:, :, 3:6], DELTA_T, axis=1)
    valid_targets = np.concatenate([valid_targets_raw, acceleration_target], axis=-1)

    valid_set = TrajectoryDataset(valid_inputs_raw, valid_targets)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=True, pin_memory=True)

    # === Setup Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    # === Build Model ===
    model = Seq2SeqDaVinciNetVelAcc(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_DIM,
        seq_len=WINDOW_SIZE,
        horizon=HORIZON,
        delta_t=DELTA_T,
        pos_mean=pos_mean,
        pos_std=pos_std,
        scaler_vel=vel_scaler
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = ade_loss
    vel_loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # === Training Loop ===
    train_losses = []
    valid_losses = []
    trigger_times = 0
    best_loss = float('inf')

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_total_loss = 0
        comp_losses = {'pos': 0.0, 'vel': 0.0, 'acc': 0.0, 'jerk': 0.0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x, teacher_forcing_ratio=0.0)
            # out shape: (batch, horizon, 12) = (pos, vel, acc, jerk)

            # Calculate loss components
            l_ade = criterion(out[:, :, :3], y[:, :, :3])
            l_vel = vel_loss(out[:, :, 3:6], y[:, :, 3:6])
            l_acc = vel_loss(out[:, :, 6:9], y[:, :, 6:9])

            # Jerk regularization: penalize large jerk values
            l_jerk = (out[:, :, 9:12] ** 2).sum(dim=(1, 2)).mean()

            # Apply weights
            w_pos = W_POS_ADE * l_ade
            w_vel = W_VEL_MSE * l_vel
            w_acc = W_ACC_MSE * l_acc
            w_jerk = W_JERK * l_jerk

            loss = w_pos + w_vel + w_acc + w_jerk

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            batch_size = x.size(0)
            epoch_total_loss += loss.item() * batch_size
            comp_losses['pos'] += w_pos.item() * batch_size
            comp_losses['vel'] += w_vel.item() * batch_size
            comp_losses['acc'] += w_acc.item() * batch_size
            comp_losses['jerk'] += w_jerk.item() * batch_size

        # Print epoch statistics
        n = len(train_set)
        avg_total = epoch_total_loss / n

        print(f"\nEpoch {epoch + 1} Summary (Total Loss: {avg_total:.6f})")
        print(f"  Pos Contribution:  {comp_losses['pos'] / n:.6f} ({(comp_losses['pos'] / n / avg_total) * 100:.1f}%)")
        print(f"  Vel Contribution:  {comp_losses['vel'] / n:.6f} ({(comp_losses['vel'] / n / avg_total) * 100:.1f}%)")
        print(f"  Acc Contribution:  {comp_losses['acc'] / n:.6f} ({(comp_losses['acc'] / n / avg_total) * 100:.1f}%)")
        print(f"  Jerk Contribution: {comp_losses['jerk'] / n:.6f} ({(comp_losses['jerk'] / n / avg_total) * 100:.1f}%)")

        # === Validation Loop ===
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val, teacher_forcing_ratio=0.0)
                val_ade = criterion(out_val[:, :, :3], y_val[:, :, :3])
                val_vel = vel_loss(out_val[:, :, 3:6], y_val[:, :, 3:6])
                val_acc = vel_loss(out_val[:, :, 6:9], y_val[:, :, 6:9])
                val_jerk = (out_val[:, :, 9:12] ** 2).sum(dim=(1, 2)).mean()
                loss_val = (W_POS_ADE * val_ade) + (W_VEL_MSE * val_vel) + (W_ACC_MSE * val_acc) + (W_JERK * val_jerk)
                valid_loss = loss_val.item()

        valid_losses.append(valid_loss)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        evaluate_at_timestamps(model, valid_loader, device, steps=[1, 2, 3, 4, 5])
        evaluate_avg_mae_rmse(model, valid_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_total:.6f}, Val Loss: {valid_loss:.6f}, LR: {current_lr:.6e}")
        train_losses.append(avg_total)

        # Early Stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model_vel_acc.pth')
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    # === Load Best Model ===
    print("\nLoading best model from best_model_vel_acc.pth...")
    model.load_state_dict(torch.load('best_model_vel_acc.pth', map_location=device))
    model.eval()

    # === Final Evaluation ===
    print("**********************************")
    print("Checking results on the trainset")
    print("**********************************")
    evaluate_at_timestamps(model, train_loader, device, steps=[1, 2, 3, 4, 5])
    evaluate_avg_mae_rmse(model, train_loader, device)

    print("**********************************")
    print("Checking results on the validset")
    print("**********************************")
    evaluate_at_timestamps(model, valid_loader, device, steps=[1, 2, 3, 4, 5])
    evaluate_avg_mae_rmse(model, valid_loader, device)

    # Save best model to final path
    model_save_path = os.path.join(DIRECTORY, 'model', f'DaVinciNet_VelAcc_Hid{HIDDEN_DIM}_ds{DOWNSAMPLE_FACTOR}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Plot training curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch (VelAcc Decoder + Jerk Reg.)')
    plt.legend(['Train Loss', 'Val Loss'])
    plt.grid()
    plt.show()

    # === Test Set Evaluation ===
    print("\nGenerating Final Visualization Plots...")
    filename = "No_U_legnth_250_step_size50vert_horA_test.npy"
    file_path = os.path.join(DIRECTORY, filename)
    test_data = np.load(file_path)

    # Downsample and add curvature
    test_data = downsample_data(test_data, DOWNSAMPLE_FACTOR)
    test_data = compute_curv_torsion_features(test_data, DELTA_T)
    model.eval()

    test_inputs_raw = test_data[:, :WINDOW_SIZE, :]
    test_inputs_raw = preprocessing(test_inputs_raw, vel_scaler, pos_mean, pos_std, curv_scaler)
    test_targets_raw = test_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3]
    test_targets = test_targets_raw

    test_set = TrajectoryDataset(test_inputs_raw, test_targets)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_v, y_v in test_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            out_v = model(x_v, teacher_forcing_ratio=0.0)
            all_preds.append(out_v[:, :, :3].cpu().numpy())
            all_targets.append(y_v[:, :, :3].cpu().numpy())

    final_preds_np = np.concatenate(all_preds, axis=0)
    final_targets_np = np.concatenate(all_targets, axis=0)

    evaluate_at_timestamps(model, test_loader, device, steps=[1, 2, 3, 4, 5])
    evaluate_avg_mae_rmse(model, test_loader, device)

    # Plot error metrics
    plot_stepwise_errors(final_preds_np, final_targets_np, title_suffix=" (Test Set - 10Hz, VelAcc+Jerk)")

    # Plot 3D trajectories
    plot_val_trajs_3d_and_xyz(
        model=model,
        valid_loader=test_loader,
        valid_data=test_data,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        device=device,
        num_examples=3
    )


if __name__ == "__main__":
    main()
