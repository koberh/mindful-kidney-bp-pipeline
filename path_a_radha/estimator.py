"""Path A BPEstimator — wraps RadhaLSTM for the common eval protocol.

Loads a trained checkpoint and implements the BPEstimator.predict() interface.

Usage example:
    est = RadhaEstimator(
        checkpoint_path="benchmark/results/path_a_first_run/best_model.pt",
        calibration_sbp_mean=120.0,  # participant's historical mean SBP
        calibration_dbp_mean=80.0,
    )
    result = est.predict(ppg_window, accel_window, fs=25)
    # result["sbp"], result["dbp"], result["confidence"]

Calibration note:
    RadhaLSTM predicts *relative* BP (deviation from daily mean).  Adding
    calibration means converts predictions to absolute mmHg.  If no
    calibration is provided, predictions are returned as relative values
    (appropriate for nocturnal-dip and trend-correlation metrics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.preprocessing import (
    NormMethod,
    PreprocessConfig,
    SQIMethod,
    bandpass_filter,
    compute_sqi,
    normalize_window,
)
from path_a_radha.features import (
    compute_rest_probabilities,
    extract_features,
)
from path_a_radha.model import RadhaLSTM, enable_mc_dropout
from path_a_radha.train import TrainConfig


class RadhaEstimator:
    """BPEstimator wrapping a trained RadhaLSTM checkpoint.

    Implements the BPEstimator protocol defined in common/eval/__init__.py.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        calibration_sbp_mean: float = 0.0,
        calibration_dbp_mean: float = 0.0,
        device: str = "cpu",
        preprocess_config: PreprocessConfig | None = None,
        n_mc_samples: int = 30,
    ) -> None:
        """
        Args:
            checkpoint_path:       path to .pt checkpoint saved by train.py
            calibration_sbp_mean:  participant's mean SBP (mmHg); added to relative predictions
            calibration_dbp_mean:  participant's mean DBP (mmHg)
            device:                "cpu" or "cuda"
            preprocess_config:     overrides default (25 Hz, z-score, template SQI)
            n_mc_samples:          Monte Carlo dropout samples for uncertainty
                                   quantification (Gal & Ghahramani 2016). 0
                                   disables MC sampling (deterministic single
                                   forward pass). Default 30 — output includes
                                   per-window sbp_std/dbp_std reflecting model
                                   epistemic uncertainty.
        """
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config_dict = ckpt.get("config", {})

        # Tier 1 preprocessing artifacts (saved by train.py). Apply in order:
        # 1. impute NaN with feat_medians, 2. select kept_indices, 3. z-score.
        self.feat_medians = ckpt.get("feat_medians")
        self.kept_indices = ckpt.get("kept_indices")
        self.feat_mean = ckpt.get("feat_mean")
        self.feat_std = ckpt.get("feat_std")
        actual_n_features = int(ckpt.get("actual_n_features", config_dict.get("n_features", 174)))
        for name, arr in [
            ("feat_medians", self.feat_medians), ("kept_indices", self.kept_indices),
            ("feat_mean", self.feat_mean), ("feat_std", self.feat_std),
        ]:
            if arr is None:
                raise RuntimeError(
                    f"Checkpoint missing {name!r}. Older checkpoints from before "
                    "Tier 1 preprocessing was added are not compatible — retrain "
                    "with the current train.py."
                )

        # Reconstruct model with the post-drop feature dimensionality
        self.model = RadhaLSTM(
            n_features=actual_n_features,
            lstm_cells=config_dict.get("lstm_cells", 32),
            dense_hidden=config_dict.get("dense_hidden", 8),
            activation=config_dict.get("dense_activation", "relu"),
            dropout=config_dict.get("dropout", 0.2),
            bidirectional=config_dict.get("bidirectional", True),  # Tier 1.5
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.calibration_sbp = calibration_sbp_mean
        self.calibration_dbp = calibration_dbp_mean
        self.n_mc_samples = int(n_mc_samples)

        self.cfg = preprocess_config or PreprocessConfig(
            fs=25,
            bp_low_hz=0.5,
            bp_high_hz=4.0,
            bp_order=4,
            sqi_method=SQIMethod.TEMPLATE,
        )

    @torch.no_grad()
    def predict(
        self,
        ppg: np.ndarray,
        accel: np.ndarray,
        fs: int = 25,
        window_starts_min: np.ndarray | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Predict SBP and DBP from one or more PPG windows.

        Args:
            ppg:   (W,) for a single window, or (T, W) for T windows
            accel: (W, 3) for a single window, or (T, W, 3) for T windows
            fs:    sampling rate in Hz (default 25)
            window_starts_min: optional (T,) minutes since recording start
                for each window. REQUIRED when the model was trained with
                position features (Tier 1.5 onward; feat_medians.shape[0] >= 176).
                For 174-feature models from before Tier 1.5, leave as None.

        Returns dict with:
            sbp:        (T,) predicted SBP in mmHg (absolute if calibration set; else relative)
            dbp:        (T,) predicted DBP in mmHg
            sbp_std:    (T,) MC dropout uncertainty for SBP in mmHg. Larger
                        values = model less confident at this window.
            dbp_std:    (T,) MC dropout uncertainty for DBP in mmHg.
            confidence: (T,) SQI value per window (0–1) — SIGNAL quality
                        (separate from MC uncertainty above which is MODEL
                        confidence). A trustworthy estimate has high SQI
                        AND low sbp_std/dbp_std.
            relative:   (T, 3) raw relative prediction means [rel_SBP, rel_DBP, rel_MAP]
            relative_std: (T, 3) MC dropout std for the three outputs
            n_mc_samples: how many MC forward passes contributed
        """
        # Normalize input to batch form
        if ppg.ndim == 1:
            ppg   = ppg[np.newaxis, :]
            accel = accel[np.newaxis, :]

        T = ppg.shape[0]
        # Extract full Radha 195-feature vector per window; slice off the 21
        # activity features [0:21] before passing to the LSTM (OPEN_QUESTIONS Q9).
        # The activity features are still computed because features.py emits them
        # together; they are consumed elsewhere (sleep/wake, motion rejection).
        feature_rows = np.empty((T, 195), dtype=np.float32)
        sqi_vals     = np.empty(T, dtype=np.float32)

        # Per-window preprocessing + feature extraction
        enmo_per_win = np.maximum(0.0, np.linalg.norm(accel, axis=2) - 1.0).mean(axis=1)
        rest_probs   = compute_rest_probabilities(enmo_per_win)

        for i in range(T):
            ppg_raw = ppg[i].astype(np.float64)
            try:
                ppg_filt = bandpass_filter(ppg_raw, fs, self.cfg.bp_low_hz,
                                           self.cfg.bp_high_hz, self.cfg.bp_order)
            except ValueError:
                ppg_filt = ppg_raw.copy()

            sqi_vals[i] = compute_sqi(ppg_filt, self.cfg.sqi_method, fs=fs)
            ppg_norm = normalize_window(ppg_filt, NormMethod.ZSCORE)
            feature_rows[i] = extract_features(
                ppg_norm, accel[i], fs=fs, rest_prob=float(rest_probs[i])
            )

        # LSTM inference: treat T windows as a single-day sequence (batch=1).
        # Apply the same Tier 1+1.5 preprocessing pipeline that train.py used:
        #   1. Drop activity features [0:21] (no MIMIC training signal for them)
        #   2. Append position features if model expects them (Tier 1.5)
        #   3. Impute NaN with training-cohort medians
        #   4. Select features kept after training-time zero-std drop
        #   5. Z-score with training-time mean/std
        feats_ppg = feature_rows[:, 21:]                                # (T, 174)
        expected_dim = int(self.feat_medians.shape[0])
        if expected_dim == 174:
            feats_raw = feats_ppg
        elif expected_dim == 176:
            # Append 2 position features: time_since_start_min, gap_to_prev_min
            # mimic_adapter writes these as the LAST two columns of the feature vector.
            if window_starts_min is None:
                # Fall back to assuming 5-min uniform spacing starting at 0
                window_starts_min = np.arange(T, dtype=np.float32) * 5.0
            else:
                window_starts_min = np.asarray(window_starts_min, dtype=np.float32)
                if window_starts_min.shape != (T,):
                    raise ValueError(
                        f"window_starts_min must be shape ({T},); got {window_starts_min.shape}"
                    )
            gap_to_prev = np.empty(T, dtype=np.float32)
            gap_to_prev[0] = 0.0
            if T > 1:
                gap_to_prev[1:] = np.diff(window_starts_min)
            position_feats = np.stack([window_starts_min, gap_to_prev], axis=1)  # (T, 2)
            feats_raw = np.concatenate([feats_ppg, position_feats], axis=1)      # (T, 176)
        else:
            raise RuntimeError(
                f"Unexpected feat_medians shape: {expected_dim}. "
                "Estimator only supports 174-feature (pre-Tier-1.5) or 176-feature "
                "(Tier 1.5+) models."
            )
        nan_mask = np.isnan(feats_raw)
        if nan_mask.any():
            idx_col = np.tile(np.arange(feats_raw.shape[1]), (feats_raw.shape[0], 1))
            feats_raw = np.where(nan_mask, self.feat_medians[idx_col], feats_raw)
        feats_kept = feats_raw[:, self.kept_indices]                    # (T, actual_n_features)
        feats_normed = (feats_kept - self.feat_mean) / self.feat_std    # z-score
        x = torch.from_numpy(feats_normed.astype(np.float32)).unsqueeze(0).to(self.device)

        # Monte Carlo dropout for uncertainty quantification (Gal & Ghahramani 2016).
        # `model.eval()` was set in __init__; enable_mc_dropout flips only the
        # Dropout layers back on so each forward pass uses a different mask.
        if self.n_mc_samples > 0:
            self.model.eval()
            enable_mc_dropout(self.model)
            mc_preds = []
            for _ in range(self.n_mc_samples):
                mc_preds.append(self.model(x).squeeze(0).cpu().numpy())  # (T, 3)
            mc_preds = np.stack(mc_preds)            # (N, T, 3)
            rel_pred = mc_preds.mean(axis=0)         # (T, 3)
            rel_std = mc_preds.std(axis=0)           # (T, 3)
        else:
            rel_pred = self.model(x).squeeze(0).cpu().numpy()
            rel_std = np.zeros_like(rel_pred)

        sbp = rel_pred[:, 0] + self.calibration_sbp
        dbp = rel_pred[:, 1] + self.calibration_dbp

        return {
            "sbp":          sbp,
            "dbp":          dbp,
            "sbp_std":      rel_std[:, 0],
            "dbp_std":      rel_std[:, 1],
            "confidence":   sqi_vals,
            "relative":     rel_pred,
            "relative_std": rel_std,
            "n_mc_samples": self.n_mc_samples,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **kwargs,
    ) -> "RadhaEstimator":
        """Convenience constructor — same as __init__."""
        return cls(checkpoint_path, **kwargs)
