"""Main training script for trajectory prediction model."""

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
from data import TrajectoryDataset, preprocessing
from models import Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM
from utils import ade_loss, evaluate_at_timestamps, plot_stepwise_errors, plot_val_trajs_3d_and_xyz


def downsample_data(data, factor):
    """
    Downsample trajectory data by taking every factor-th sample.

    Args:
        data: (N, seq_len, features) array
        factor: Downsampling factor (e.g., 10 for 100Hz -> 10Hz)

    Returns:
        Downsampled data (N, seq_len // factor, features)
    """
    return data[:, ::factor, :]


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Downsampling: 100Hz -> {100 // DOWNSAMPLE_FACTOR}Hz (factor={DOWNSAMPLE_FACTOR})")
    print(f"Input window: {ORIGINAL_WINDOW_SIZE} -> {WINDOW_SIZE} timesteps")
    print(f"Horizon: {ORIGINAL_HORIZON} -> {HORIZON} timesteps")
    print(f"Delta_t: 0.01s -> {DELTA_T}s")
    print()

    # === Load Training Data ===
    filename = "No_B_legnth_250_step_size50vert_hor_WithoutB.npy"
    file_path = os.path.join(DIRECTORY, filename)
    train_data = np.load(file_path)
    print(f"Original train data shape: {train_data.shape}")

    # Downsample the data
    train_data = downsample_data(train_data, DOWNSAMPLE_FACTOR)
    print(f"Downsampled train data shape: {train_data.shape}")

    # Prepare input features and targets
    train_inputs_raw = train_data[:, :WINDOW_SIZE, :]
    train_inputs_raw, vel_scaler, pos_mean = preprocessing(train_inputs_raw)
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

    # Downsample
    valid_data = downsample_data(valid_data, DOWNSAMPLE_FACTOR)

    valid_inputs_raw = valid_data[:, :WINDOW_SIZE, :]
    valid_inputs_raw = preprocessing(valid_inputs_raw, vel_scaler, pos_mean)
    valid_targets_raw = valid_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :OUTPUT_SIZE]
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
    encoder = Encoder_LSTM(input_size=13, hidden_size=HIDDEN_DIM, num_layers=1)
    decoder = Decoder_LSTM(input_size=9, hidden_size=HIDDEN_DIM, num_layers=1, output_size=3)

    model = Seq2SeqLSTM(
        encoder=encoder,
        decoder=decoder,
        horizon=HORIZON,
        delta_t=DELTA_T,
        pos_mean=pos_mean,
        scaler_vel=vel_scaler
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = ade_loss
    vel_loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # === Training Loop ===
    train_losses = []
    valid_losses = []
    trigger_times = 0
    best_loss = float('inf')

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_total_loss = 0
        comp_losses = {'pos': 0.0, 'vel': 0.0, 'acc': 0.0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            teacher_forcing_ratio = max(0.0, 1 - (epoch / 100))
            out = model(x, y, teacher_forcing_ratio=teacher_forcing_ratio)

            # Calculate loss components
            l_ade = criterion(out[:, :, :3], y[:, :, :3])
            l_vel = vel_loss(out[:, :, 3:6], y[:, :, 3:6])
            l_acc = vel_loss(out[:, :, 6:9], y[:, :, 6:9])

            # Apply weights
            w_pos = W_POS_ADE * l_ade
            w_vel = W_VEL_MSE * l_vel
            w_acc = W_ACC_MSE * l_acc

            loss = w_pos + w_vel + w_acc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            batch_size = x.size(0)
            epoch_total_loss += loss.item() * batch_size
            comp_losses['pos'] += w_pos.item() * batch_size
            comp_losses['vel'] += w_vel.item() * batch_size
            comp_losses['acc'] += w_acc.item() * batch_size

        # Print epoch statistics
        n = len(train_set)
        avg_total = epoch_total_loss / n

        print(f"\nEpoch {epoch + 1} Summary (Total Loss: {avg_total:.6f})")
        print(f"  Pos Contribution:  {comp_losses['pos'] / n:.6f} ({(comp_losses['pos'] / n / avg_total) * 100:.1f}%)")
        print(f"  Vel Contribution:  {comp_losses['vel'] / n:.6f} ({(comp_losses['vel'] / n / avg_total) * 100:.1f}%)")
        print(f"  Acc Contribution:  {comp_losses['acc'] / n:.6f} ({(comp_losses['acc'] / n / avg_total) * 100:.1f}%)")

        # === Validation Loop ===
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val, y_val, teacher_forcing_ratio=0.0)
                val_ade = criterion(out_val[:, :, :3], y_val[:, :, :3])
                val_vel = vel_loss(out_val[:, :, 3:6], y_val[:, :, 3:6])
                val_acc = vel_loss(out_val[:, :, 6:9], y_val[:, :, 6:9])
                loss_val = (W_POS_ADE * val_ade) + (W_VEL_MSE * val_vel) + (W_ACC_MSE * val_acc)
                valid_loss = loss_val.item()

        valid_losses.append(valid_loss)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        evaluate_at_timestamps(model, valid_loader, device, steps=[1, 2, 3, 4, 5])

        print(f"Epoch {epoch + 1}, Train Loss: {avg_total:.6f}, Val Loss: {valid_loss:.6f}, LR: {current_lr:.6e}")
        train_losses.append(avg_total)

        # Early Stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    # === Final Evaluation ===
    print("**********************************")
    print("Checking results on the trainset")
    print("**********************************")
    evaluate_at_timestamps(model, train_loader, device, steps=[1, 2, 3, 4, 5])

    # Save model
    model_save_path = os.path.join(DIRECTORY, 'model', f'Seq2SeqLSTM_Hid{HIDDEN_DIM}_ds{DOWNSAMPLE_FACTOR}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Plot training curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend(['Train Loss', 'Val Loss'])
    plt.grid()
    plt.show()

    # === Test Set Evaluation ===
    print("\nGenerating Final Visualization Plots...")
    filename = "No_U_legnth_250_step_size50vert_horA_test.npy"
    file_path = os.path.join(DIRECTORY, filename)
    test_data = np.load(file_path)

    # Downsample test data
    test_data = downsample_data(test_data, DOWNSAMPLE_FACTOR)
    model.eval()

    test_inputs_raw = test_data[:, :WINDOW_SIZE, :]
    test_inputs_raw = preprocessing(test_inputs_raw, vel_scaler, pos_mean)
    test_targets_raw = test_data[:, WINDOW_SIZE:WINDOW_SIZE + HORIZON, :3]
    test_targets = test_targets_raw

    test_set = TrajectoryDataset(test_inputs_raw, test_targets)
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_v, y_v in test_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            out_v = model(x_v, y_v, teacher_forcing_ratio=0.0)
            all_preds.append(out_v[:, :, :3].cpu().numpy())
            all_targets.append(y_v[:, :, :3].cpu().numpy())

    final_preds_np = np.concatenate(all_preds, axis=0)
    final_targets_np = np.concatenate(all_targets, axis=0)

    # Plot error metrics
    plot_stepwise_errors(final_preds_np, final_targets_np, title_suffix=" (Test Set - 10Hz)")

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
    evaluate_at_timestamps(model, test_loader, device, steps=[1, 2, 3, 4, 5])


if __name__ == "__main__":
    main()
