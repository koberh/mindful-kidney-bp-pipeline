"""Phase 5 training script — in-domain retraining on Aurora-BP ambulatory data.

Trains a fresh Radha-LSTM on the Aurora-BP ambulatory .npz files produced by
aurora_bp_adapter.py.  Uses the same architecture and hyperparameters as Phase 3
(the deployed model) so results are directly comparable.

Key differences from train.py / train_phase4.py:
  - Data source: Aurora-BP ambulatory (wrist PPG + oscillometric cuff)
    instead of MIMIC-III ICU (fingertip PPG + arterial line)
  - Short windows: each .npz row = 1 Aurora-BP 15-second snippet
    (vs ~5-min windows from MIMIC).  The LSTM sequence length per day is
    ~18-22 (cuff readings/day) rather than ~288 (5-min slots in 24h).
  - Targets stored as absolute BP in the .npz; ParticipantDayDataset
    relativizes them (same as build_day_npz convention).
  - Output directory: benchmark/results/path_a_phase5

Usage:
    # 1. Run aurora_bp_adapter.py first to generate the .npz files
    python -m path_a_radha.aurora_bp_adapter \\
        --tsv  data/aurora_bp/measurements_oscillometric.tsv \\
        --zip  data/aurora_bp/measurements_oscillometric.zip \\
        --output-dir data/processed/path_a_aurora

    # 2. Train Phase 5
    python -m path_a_radha.train_phase5

    # 3. Check results
    #    benchmark/results/path_a_phase5/best_model.pt
    #    benchmark/results/path_a_phase5/per_subject_metrics.json
"""
from __future__ import annotations

import logging
from pathlib import Path

from path_a_radha.train import TrainConfig, train

logger = logging.getLogger(__name__)


def phase5_config() -> TrainConfig:
    """Phase 5 training config — mirrors Phase 3 architecture on Aurora-BP data."""
    return TrainConfig(
        # Architecture — identical to Phase 3 so comparison is clean
        n_features=176,          # 174 PPG + 2 position features
        lstm_cells=16,
        dense_hidden=8,
        dense_activation="relu",
        dropout=0.4,
        bidirectional=True,
        grad_clip_norm=5.0,
        feat_zero_std_eps=1e-3,
        feat_jitter_std=0.10,
        # Training
        lr=1e-3,
        weight_decay=1e-3,
        batch_size_days=8,
        val_fraction=0.20,
        early_stop_patience=50,
        max_epochs=2000,
        seed=42,
        sqi_threshold=0.3,
        min_windows_per_day=3,
        # Data — Aurora-BP ambulatory .npz files
        data_dir=Path("data/processed/path_a_aurora"),
        output_dir=Path("benchmark/results/path_a_phase5"),
        mlflow_experiment="path_a_phase5",
        mlflow_tracking_uri="benchmark/results/mlruns",
        device="cpu",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = phase5_config()
    logger.info("Phase 5 config: %s", cfg)
    ckpt = train(cfg)
    logger.info("Phase 5 training complete. Checkpoint: %s", ckpt)
