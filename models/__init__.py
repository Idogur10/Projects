"""Models module for trajectory prediction."""

from .seq2seq import (
    Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM,
    # DaVinciNet-style architecture
    KinematicsEncoder, TemporalAttention, Seq2SeqDaVinciNet,
    Encoder_GRU, Decoder_GRU, Seq2SeqGRU  # Aliases for backward compatibility
)

__all__ = [
    'Encoder_LSTM', 'Decoder_LSTM', 'Seq2SeqLSTM',
    'KinematicsEncoder', 'TemporalAttention', 'Seq2SeqDaVinciNet',
    'Encoder_GRU', 'Decoder_GRU', 'Seq2SeqGRU'
]
