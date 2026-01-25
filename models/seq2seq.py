"""Sequence-to-sequence models for trajectory prediction."""

import numpy as np
import torch
import torch.nn as nn


class Encoder_GRU(nn.Module):
    """GRU-based encoder for trajectory sequences."""

    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, h = self.gru(x)
        return h


class Decoder_GRU(nn.Module):
    """GRU-based decoder for trajectory prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, pre_hidden):
        output, h = self.gru(x, pre_hidden)
        prediction = self.fc(output)
        prediction = prediction.squeeze(1)
        return prediction, h


class Seq2SeqGRU(nn.Module):
    """Sequence-to-sequence model with physics-based trajectory prediction."""

    def __init__(self, encoder, decoder, horizon, delta_t, pos_mean, scaler_vel):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.horizon = horizon
        self.delta_t = delta_t

        # Register Buffers for Normalization
        self.register_buffer('scaler_mean', torch.tensor(scaler_vel.mean_[:3], dtype=torch.float32))
        self.register_buffer('scaler_std', torch.tensor(scaler_vel.scale_[:3], dtype=torch.float32))
        self.register_buffer('pos_mean', torch.tensor(pos_mean, dtype=torch.float32))

    def forward(self, input_batch, target_seq=None, teacher_forcing_ratio=0.5):
        # 1. ENCODE
        hidden = self.encoder(input_batch)

        # 2. INITIALIZE PHYSICS STATE (Real World Units)
        current_pos = input_batch[:, -1, :3] + self.pos_mean
        current_vel = (input_batch[:, -1, 3:6] * self.scaler_std) + self.scaler_mean

        # Calculate initial acceleration (Real World)
        pre_vel = (input_batch[:, -2, 3:6] * self.scaler_std) + self.scaler_mean
        current_acc = (current_vel - pre_vel) / self.delta_t

        path_vec = []

        for t in range(self.horizon):
            # --- A. PREPARE NETWORK INPUT (Normalize!) ---
            # 1. Normalize Position (Center it)
            norm_pos = current_pos - self.pos_mean

            # 2. Normalize Velocity (Standard Scaler)
            norm_vel = (current_vel - self.scaler_mean) / self.scaler_std

            # 3. Normalize Acceleration
            norm_acc = current_acc / self.scaler_std

            # Concatenate Normalized Inputs for the Decoder
            input_decoder = torch.cat((norm_pos, norm_vel, norm_acc), dim=1)

            # --- B. NETWORK PREDICTION ---
            pred_raw, hidden = self.decoder(input_decoder.unsqueeze(1), hidden)

            # Un-Normalize Prediction to get Real World Acceleration (m/s^2)
            pred_acc_real = pred_raw * self.scaler_std

            # --- C. PHYSICS UPDATE (Velocity Verlet) ---
            # 1. Position Update
            pos_predicted = current_pos + (current_vel * self.delta_t) + (0.5 * current_acc * self.delta_t ** 2)

            # 2. Velocity Update
            current_vel = current_vel + 0.5 * (current_acc + pred_acc_real) * self.delta_t

            # 3. Acceleration Update
            current_acc = pred_acc_real

            # --- D. STORE & LOOP ---
            output_save = torch.cat((pos_predicted, current_vel, current_acc), dim=1)
            path_vec.append(output_save.unsqueeze(1))

            # --- E. NEXT STEP STATE ---
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                # Teacher Forcing: Reset Physics State to Ground Truth
                current_pos = target_seq[:, t, :3]
                current_vel = target_seq[:, t, 3:6]
                current_acc = target_seq[:, t, 6:9]
            else:
                # Auto-regressive: Use our calculated position
                current_pos = pos_predicted

        return torch.cat(path_vec, dim=1)
