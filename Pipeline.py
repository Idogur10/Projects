
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    HIDDEN_DIM, N_EPOCHS, LEARNING_RATE,
    RANDOM_SEED, DIRECTORY, WINDOW_SIZE, HORIZON, DELTA_T, OUTPUT_SIZE,
    W_POS_ADE, W_VEL_MSE, W_ACC_MSE, PATIENCE,
    DOWNSAMPLE_FACTOR, ORIGINAL_WINDOW_SIZE, ORIGINAL_HORIZON
)
from data import TrajectoryDataset, preprocessing
from models import Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM
from utils import ade_loss, evaluate_at_timestamps, plot_stepwise_errors, plot_val_trajs_3d_and_xyz
""" parametrs relvant here

"""
BATCH_SIZE=30
delta_t=0.1

def initialize_constant_matrices(H, K, degree=3):
    # 1. Setup Knot Vector (Clamped)
    # Total knots = n_control_points + degree + 1
    n_knots = K + degree + 1
    knots = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_knots - 2 * degree),
        np.ones(degree)
    ])

    # 2. Setup evaluation points (parameter s)
    s_vals = np.linspace(0, 1, H)

    # 3. Build Matrices
    # Each column of the matrix represents one basis function
    coeffs = np.eye(K)
    spline = BSpline(knots, coeffs, degree)

    # Convert to Torch Tensors
    B = torch.from_numpy(spline(s_vals)).float()              # (H, K)
    B_dot = torch.from_numpy(spline.derivative(1)(s_vals)).float()  # (H, K)
    B_ddot = torch.from_numpy(spline.derivative(2)(s_vals)).float() # (H, K)
    B_dddot = torch.from_numpy(spline.derivative(3)(s_vals)).float() # (H, K)

    return B, B_dot, B_ddot, B_dddot

def get_parameters_per_sample(input_vel, delta_time, eps, start_index):
    """
    Returns lists of alpha and beta for each sample in the batch.
    Uses log-linear regression to fit the Power Law: V = alpha * K^beta * T^lambda
    """
    # 1. Slice and calculate derivatives
    input_vel = input_vel[:, start_index:, :]
    accel = np.gradient(input_vel, delta_time, axis=1)


    v_norm = np.linalg.norm(input_vel, axis=-1)
    v_sq_norm = v_norm ** 2
    a_sq_norm = np.sum(accel ** 2, axis=2)
    v_dot_a_sq = np.sum(input_vel * accel, axis=2) ** 2

    # Curvature (Kappa)
    kappa_num = np.sqrt(np.maximum(v_sq_norm * a_sq_norm - v_dot_a_sq, 0))
    kappa = kappa_num / (v_sq_norm ** 1.5 + eps)


    Alpha, Beta = [], []

    for i in range(input_vel.shape[0]):
        # We take absolute values because log is undefined for negative/zero
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
            Alpha.append(np.nan);
            Beta.append(np.nan);
            

    return Alpha, Beta

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
    # Loading train data to extract the relevant scalers for position and velocity
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
    
    #loading the test data
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
    #Finding the alpha,beta parameters.
    velocity_input = valid_inputs_raw[:BATCH_SIZE, :, 3:6]  # Assuming velocity is in the last 3 channels
    alphas_ref, betas_ref = get_parameters_per_sample(velocity_input, delta_t, eps=1e-8, start_index=0)
    alphas_np = np.asarray(alphas_ref, dtype=np.float64)
    betas_np = np.asarray(betas_ref, dtype=np.float64)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_alphas = torch.tensor(alphas_ref, dtype=torch.float32).to(device)
    target_betas = torch.tensor(betas_ref, dtype=torch.float32).to(device)
    

    # Model
    encoder = Encoder_LSTM(valid_inputs_raw.shape[-1], HIDDEN_DIM)
    decoder = Decoder_LSTM(OUTPUT_SIZE + 3, HIDDEN_DIM)
    model = Seq2SeqLSTM(encoder, decoder, HORIZON, DEVICE='cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load('C:/Users/idogu/OneDrive/Documents/Master/data/model/Seq2SeqLSTM_Hid96_ds10.pth',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(weights)

    # Evaluation
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(train_inputs_raw[:Batch_size, :, :], dtype=torch.float32).to(device)
        R_u = model(input_tensor)
    # Creating the B matrices for the first
    H, K_ctrl = 5, 3
    B, B_dot, B_ddot, B_dddot = initialize_constant_matrices(H, K_ctrl)

if __name__ == "__main__":
    main()   