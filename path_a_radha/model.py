"""Path A: Radha et al. 2019 LSTM architecture.

Architecture (Radha 2019, §3.3.4):
  Input  →  Dense(8)  →  LSTM(32)  →  Dense(3)  →  [rel_SBP, rel_DBP, rel_MAP]

All targets are *relative* BP (daily mean subtracted).
MAP output is included only to improve convergence (Su et al. 2018 approach
cited in Radha §3.3 paragraph on loss function).

Input: (batch, seq_len, n_features=174)
  - seq_len = number of 5-min windows per day (varies; LSTM processes as sequence)
  - n_features = 174 handcrafted PPG features (Radha's 195 minus 21 activity;
    see ARCHITECTURE.md §1 and OPEN_QUESTIONS.md Q9 — MIMIC training data
    has no accelerometer, so activity features are dropped from the LSTM input
    and used only by sleep/wake and motion-rejection logic at LEAP inference)

Loss: MSE * |SBP - mean_SBP_training| amplification (Radha §3.3 paragraph 4)

Do not modify this until OPEN_QUESTIONS.md items are resolved and
ARCHITECTURE.md is finalized (Gate 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RadhaLSTM(nn.Module):
    """32-cell LSTM for relative BP trend estimation.

    Radha et al. 2019, §3.3.4 and Table 2 (best configuration).
    Architecture:
      - Dense hidden layer: 8 perceptrons (fixed at best MLP result)
      - LSTM layer: 32 cells
      - Output layer: 3 values (relative SBP, DBP, MAP)
    """

    def __init__(
        self,
        n_features: int = 174,
        lstm_cells: int = 32,
        dense_hidden: int = 8,
        activation: str = "relu",   # "tanh" is first fallback if val loss unstable (Q7)
        dropout: float = 0.2,       # Tier 1 — regularize ~7K-param model on ~50 subjects
        bidirectional: bool = True, # Tier 1.5 — offline reports use future context
    ) -> None:
        super().__init__()
        self.dense_in = nn.Linear(n_features, dense_hidden)
        _acts = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "elu": nn.ELU()}
        if activation not in _acts:
            raise ValueError(f"Unknown activation: {activation!r}. Choose from {list(_acts)}")
        self.act = _acts[activation]
        self.dropout_dense = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=dense_hidden,
            hidden_size=lstm_cells,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout_lstm = nn.Dropout(dropout)
        # Bidirectional LSTM concatenates forward + backward hidden states → 2x cells
        output_in_features = lstm_cells * (2 if bidirectional else 1)
        self.output = nn.Linear(output_in_features, 3)  # [rel_SBP, rel_DBP, rel_MAP]
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → (batch, seq_len, 3)."""
        h = self.act(self.dense_in(x))
        h = self.dropout_dense(h)
        lstm_out, _ = self.lstm(h)
        lstm_out = self.dropout_lstm(lstm_out)
        return self.output(lstm_out)


def enable_mc_dropout(model: nn.Module) -> None:
    """Re-enable dropout layers only — call after model.eval() to support
    Monte Carlo (MC) dropout inference (Gal & Ghahramani 2016).

    Standard `model.eval()` disables dropout (deterministic output). For
    uncertainty quantification we want dropout active at inference: multiple
    forward passes with different masks → distribution of predictions →
    prediction mean + std.

    All other layers (LSTM, Linear, etc.) stay in eval mode (no gradient,
    BatchNorm uses running stats).
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def amplified_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sbp_mean_train: float,
) -> torch.Tensor:
    """MSE loss amplified by deviation of SBP from training mean.

    Radha et al. 2019 §3.3 paragraph 4:
    'the loss was amplified by the absolute difference of the SBP value
    from the mean SBP in the training set.'

    pred and target shape: (batch, seq_len, 3) — index 0 is SBP.
    """
    mse = (pred - target) ** 2
    sbp_dev = torch.abs(target[..., 0] - sbp_mean_train).unsqueeze(-1)
    amplified = mse * (1.0 + sbp_dev)
    return amplified.mean()
