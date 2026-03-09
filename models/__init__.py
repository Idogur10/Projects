"""Models module for trajectory prediction."""

from .seq2seq import (
    Encoder_LSTM, Decoder_LSTM, Seq2SeqLSTM,
    # DaVinciNet-style architecture
    KinematicsEncoder, TemporalAttention, Seq2SeqDaVinciNet,
    Seq2SeqDaVinciNetVelAcc,
    Encoder_GRU, Decoder_GRU, Seq2SeqGRU  # Aliases for backward compatibility
)
from .bspline_optimization import ModelCopt

__all__ = [
    'Encoder_LSTM', 'Decoder_LSTM', 'Seq2SeqLSTM',
    'KinematicsEncoder', 'TemporalAttention', 'Seq2SeqDaVinciNet',
    'Seq2SeqDaVinciNetVelAcc',
    'Encoder_GRU', 'Decoder_GRU', 'Seq2SeqGRU',
    'ModelCopt'
]
