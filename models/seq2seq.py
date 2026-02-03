"""Sequence-to-sequence models for trajectory prediction."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_LSTM(nn.Module):
    """LSTM-based encoder for trajectory sequences."""

    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        return (h, c)  # Return both hidden and cell state


class Decoder_LSTM(nn.Module):
    """LSTM-based decoder for trajectory prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Decoder_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        # hidden_state is a tuple (h, c)
        output, (h, c) = self.lstm(x, hidden_state)
        prediction = self.fc(output)
        prediction = prediction.squeeze(1)
        return prediction, (h, c)


class Seq2SeqLSTM(nn.Module):
    """Sequence-to-sequence model with physics-based trajectory prediction using LSTM."""

    def __init__(self, encoder, decoder, horizon, delta_t, pos_mean, scaler_vel):
        super(Seq2SeqLSTM, self).__init__()
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
        hidden = self.encoder(input_batch)  # Returns (h, c) tuple

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


# =============================================================================
# DaVinciNet-style Architecture (Input Attention Encoder + Temporal Attention Decoder)
# Reference: "daVinciNet: Joint Prediction of Motion and Surgical State" (Qin et al., 2020)
# =============================================================================

class KinematicsEncoder(nn.Module):
    """
    LSTM Encoder with Input Attention mechanism (DaVinciNet/DA-RNN style).

    At each time step, learns attention weights for each input feature based on
    the previous hidden state and cell state. This allows the encoder to
    adaptively focus on the most relevant input features.

    Equations (from DA-RNN paper):
        α_t = softmax(tanh(W_e[h_{t-1}; s_{t-1}] + V_e * x^i))
        x̃_t = Σ α_i * x_i_t
    """

    def __init__(self, input_size, hidden_size, seq_len):
        super(KinematicsEncoder, self).__init__()
        self.input_size = input_size  # Number of input features
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # LSTM cell for step-by-step processing with attention
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Input Attention parameters
        # W_e: maps [h_{t-1}; s_{t-1}] to attention space
        self.W_e = nn.Linear(2 * hidden_size, seq_len, bias=False)
        # V_e: maps each input series x^i (length T) to attention space
        self.V_e = nn.Linear(seq_len, seq_len, bias=False)
        # v_e: reduces to scalar attention score
        self.v_e = nn.Linear(seq_len, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            all_hidden: All hidden states (batch, seq_len, hidden_size)
            (h, c): Final hidden and cell states (batch, hidden_size)
        """
        batch_size = x.size(0)
        device = x.device

        # Transpose for attention: (batch, input_size, seq_len)
        x_transposed = x.permute(0, 2, 1)

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        all_hidden = []

        for t in range(self.seq_len):
            # Compute input attention weights
            # Concatenate h and c: (batch, 2*hidden)
            h_c = torch.cat([h, c], dim=1)

            # W_e * [h; c]: (batch, seq_len)
            attn_hidden = self.W_e(h_c)

            # V_e * x^i for all features: (batch, input_size, seq_len)
            attn_input = self.V_e(x_transposed)

            # Combine and apply tanh: (batch, input_size, seq_len)
            attn_combined = torch.tanh(attn_hidden.unsqueeze(1) + attn_input)

            # Get scalar scores: (batch, input_size)
            e = self.v_e(attn_combined).squeeze(-1)

            # Softmax for attention weights
            alpha = F.softmax(e, dim=1)

            # Apply attention to current timestep input
            x_t = x[:, t, :]  # (batch, input_size)
            x_tilde = alpha * x_t  # Weighted input

            # LSTM step
            h, c = self.lstm_cell(x_tilde, (h, c))
            all_hidden.append(h.unsqueeze(1))

        # Stack all hidden states: (batch, seq_len, hidden)
        all_hidden = torch.cat(all_hidden, dim=1)

        return all_hidden, (h, c)


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism for the decoder (DaVinciNet style).

    Allows the decoder to selectively attend to relevant encoder hidden states
    across all time steps when making predictions.

    Equations:
        β_t = softmax(tanh(W_d[d_{t-1}; c_{t-1}] + V_d * h_j))
        context = Σ β_j * h_j
    """

    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(TemporalAttention, self).__init__()

        # W_d: maps decoder state [d; c] to attention space
        self.W_d = nn.Linear(2 * decoder_hidden_size, encoder_hidden_size, bias=False)
        # V_d: maps encoder hidden states to attention space
        self.V_d = nn.Linear(encoder_hidden_size, encoder_hidden_size, bias=False)
        # v_d: reduces to scalar
        self.v_d = nn.Linear(encoder_hidden_size, 1, bias=False)

    def forward(self, encoder_hidden, decoder_h, decoder_c):
        """
        Args:
            encoder_hidden: All encoder hidden states (batch, enc_seq_len, enc_hidden)
            decoder_h: Decoder hidden state (batch, dec_hidden)
            decoder_c: Decoder cell state (batch, dec_hidden)

        Returns:
            context: Weighted context vector (batch, enc_hidden)
            beta: Attention weights (batch, enc_seq_len)
        """
        # Concatenate decoder states: (batch, 2*dec_hidden)
        d_c = torch.cat([decoder_h, decoder_c], dim=1)

        # W_d * [d; c]: (batch, enc_hidden)
        attn_decoder = self.W_d(d_c)

        # V_d * h_j for all encoder states: (batch, enc_seq_len, enc_hidden)
        attn_encoder = self.V_d(encoder_hidden)

        # Combine: (batch, enc_seq_len, enc_hidden)
        attn_combined = torch.tanh(attn_decoder.unsqueeze(1) + attn_encoder)

        # Scalar scores: (batch, enc_seq_len)
        l = self.v_d(attn_combined).squeeze(-1)

        # Attention weights
        beta = F.softmax(l, dim=1)

        # Context vector: weighted sum of encoder hidden states
        context = torch.bmm(beta.unsqueeze(1), encoder_hidden).squeeze(1)

        return context, beta


class Seq2SeqDaVinciNet(nn.Module):
    """
    DaVinciNet-style Seq2Seq model with Verlet integration.

    Architecture:
    - Encoder: LSTM with Input Attention (focuses on relevant input features)
    - Decoder: LSTM with Temporal Attention (attends to encoder hidden states)
    - Physics: Velocity Verlet integration for trajectory prediction

    The decoder predicts acceleration, which is used in Verlet integration
    to compute physically consistent position and velocity updates.
    """

    def __init__(self, input_size, hidden_size, seq_len, horizon, delta_t, pos_mean, scaler_vel):
        super(Seq2SeqDaVinciNet, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.horizon = horizon
        self.delta_t = delta_t

        # Encoder with Input Attention
        self.encoder = KinematicsEncoder(input_size, hidden_size, seq_len)

        # Temporal Attention
        self.temporal_attention = TemporalAttention(hidden_size, hidden_size)

        # Decoder LSTM cell
        # Input: physics state (9) + context vector (hidden_size)
        self.decoder_cell = nn.LSTMCell(9 + hidden_size, hidden_size)

        # Output layer: predicts acceleration (3D)
        self.fc_out = nn.Linear(hidden_size + hidden_size, 3)  # decoder_hidden + context

        # Register Buffers for Normalization
        self.register_buffer('scaler_mean', torch.tensor(scaler_vel.mean_[:3], dtype=torch.float32))
        self.register_buffer('scaler_std', torch.tensor(scaler_vel.scale_[:3], dtype=torch.float32))
        self.register_buffer('pos_mean', torch.tensor(pos_mean, dtype=torch.float32))

    def forward(self, input_batch, target_seq=None, teacher_forcing_ratio=0.5):
        batch_size = input_batch.size(0)

        # 1. ENCODE with Input Attention
        encoder_hidden, (h_enc, c_enc) = self.encoder(input_batch)
        # encoder_hidden: (batch, seq_len, hidden_size)

        # Initialize decoder states from encoder final state
        d = h_enc
        c = c_enc

        # 2. INITIALIZE PHYSICS STATE (Real World Units)
        current_pos = input_batch[:, -1, :3] + self.pos_mean
        current_vel = (input_batch[:, -1, 3:6] * self.scaler_std) + self.scaler_mean

        # Calculate initial acceleration
        pre_vel = (input_batch[:, -2, 3:6] * self.scaler_std) + self.scaler_mean
        current_acc = (current_vel - pre_vel) / self.delta_t

        path_vec = []

        for t in range(self.horizon):
            # --- A. TEMPORAL ATTENTION ---
            context, attn_weights = self.temporal_attention(encoder_hidden, d, c)

            # --- B. PREPARE DECODER INPUT ---
            # Normalize physics state
            norm_pos = current_pos - self.pos_mean
            norm_vel = (current_vel - self.scaler_mean) / self.scaler_std
            norm_acc = current_acc / self.scaler_std

            # Concatenate physics state with context
            physics_state = torch.cat((norm_pos, norm_vel, norm_acc), dim=1)
            decoder_input = torch.cat((physics_state, context), dim=1)

            # --- C. DECODER LSTM STEP ---
            d, c = self.decoder_cell(decoder_input, (d, c))

            # --- D. PREDICT ACCELERATION ---
            decoder_context = torch.cat((d, context), dim=1)
            pred_raw = self.fc_out(decoder_context)

            # Un-normalize to get real world acceleration
            pred_acc_real = pred_raw * self.scaler_std

            # --- E. PHYSICS UPDATE (Velocity Verlet) ---
            pos_predicted = current_pos + (current_vel * self.delta_t) + (0.5 * current_acc * self.delta_t ** 2)
            current_vel = current_vel + 0.5 * (current_acc + pred_acc_real) * self.delta_t
            current_acc = pred_acc_real

            # --- F. STORE ---
            output_save = torch.cat((pos_predicted, current_vel, current_acc), dim=1)
            path_vec.append(output_save.unsqueeze(1))

            # --- G. TEACHER FORCING ---
            if target_seq is not None and np.random.rand() < teacher_forcing_ratio:
                current_pos = target_seq[:, t, :3]
                current_vel = target_seq[:, t, 3:6]
                current_acc = target_seq[:, t, 6:9]
            else:
                current_pos = pos_predicted

        return torch.cat(path_vec, dim=1)


# Aliases for backward compatibility
Encoder_GRU = Encoder_LSTM
Decoder_GRU = Decoder_LSTM
Seq2SeqGRU = Seq2SeqLSTM
