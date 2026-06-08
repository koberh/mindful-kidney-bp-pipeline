"""Path A training script — Radha et al. 2019 LSTM.

Training protocol (Radha 2019 §3.2.6 and §3.3):
- Batch size: 32 participant-days (one day = one variable-length LSTM sequence)
- 80/20 subject-disjoint train/val split
- Early stopping: patience=20 epochs without val-loss improvement
- Optimizer: Adam lr=1e-3, weight_decay=1e-4 (Q6 decision)
- Targets: relative SBP, DBP, MAP (per-day mean subtracted per participant)
- MAP included to improve convergence (Su et al. 2018, cited in Radha §3.3)

Data format (data/processed/path_a/<pid>_<date>.npz):
    features  : (T, 174) float32   — one feature vector per 5-min cuff-centered window
                                       (Radha 195 minus 21 activity; see OPEN_QUESTIONS Q9)
    targets   : (T, 3) float32     — [SBP_mmhg, DBP_mmhg, MAP_mmhg], NaN if no cuff nearby
    sqi       : (T,) float32       — SQI per window (template method)
    participant_id: str
    date      : str (YYYY-MM-DD)

See build_day_npz() for how to produce these from a LEAPRecording + cuff readings.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from path_a_radha.model import RadhaLSTM, amplified_mse_loss

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Architecture (Radha §3.3.4)
    n_features: int = 176               # 174 PPG + 2 position features (Tier 1.5)
    lstm_cells: int = 16                # Phase 3 — halved 32→16 to reduce overfitting capacity
    dense_hidden: int = 8
    dense_activation: str = "relu"      # Q7 decision; first fallback is "tanh"
    dropout: float = 0.4                # Phase 3 — bumped 0.2→0.4 (Phase 2 overfit by epoch 20)
    bidirectional: bool = True          # Tier 1.5 — offline reports use future context
    grad_clip_norm: float = 5.0         # Bumped 1.0→5.0 after Phase 1 found loss-scale dampened learning
    feat_zero_std_eps: float = 1e-3     # Bumped 1e-6→1e-3 to drop near-constant features (raw-idx 152 Monte-Moreno had std=3.86e-4 → blew up under z-score)
    feat_jitter_std: float = 0.10       # Phase 3 — bumped 0.05→0.10 for more augmentation
    lr_plateau_factor: float = 0.5      # Tier 1.5 — ReduceLROnPlateau scheduler
    lr_plateau_patience: int = 15       # Bumped 5→15 — don't drop LR too aggressively before model has chance
    lr_min: float = 1e-6
    sqi_threshold: float = 0.3          # Lowered from 0.5 — SQI weighting in loss handles soft-quality
    # Training (Radha §3.2.6) — adjusted after Phase 1 found 376 total weight updates was insufficient
    batch_size_days: int = 8            # Bumped down from 32 → 5-6 batches/epoch instead of ~2
    val_fraction: float = 0.20          # subject-level split
    early_stop_patience: int = 50       # Bumped 20→50 — give model chance to find real learning signal
    max_epochs: int = 2000              # Bumped 500→2000 — early stopping still primary; this is the budget ceiling
    seed: int = 42
    # Optimizer (Q6 decision)
    lr: float = 1e-3
    weight_decay: float = 1e-3          # Phase 3 — bumped 1e-4→1e-3 for stronger L2 regularization
    # Loss (Radha §3.3 ¶4) — 0.0 when targets are already relative
    sbp_mean_train: float = 0.0
    # Paths
    data_dir: Path = Path("data/processed/path_a")
    output_dir: Path = Path("benchmark/results/path_a_first_run")
    mlflow_experiment: str = "path_a_radha"
    mlflow_tracking_uri: str = "benchmark/results/mlruns"
    device: str = "cpu"
    # Quality filter (sqi_threshold also declared above with Tier 1.5 value)
    min_windows_per_day: int = 3        # skip day-files with too few valid windows
    # Optional explicit val-subject override (Phase 4 head-to-head):
    # If non-empty, these participant_ids are forced into val regardless of
    # the random shuffle. Used to align Phase 4's val cohort with Phase 3's
    # 37-subject val set so the two models can be compared on identical
    # held-out subjects. None / [] -> normal random subject split.
    val_subjects: list[str] | None = None


# ── Dataset ───────────────────────────────────────────────────────────────────

class ParticipantDayDataset(Dataset):
    """One item = one participant-day: (T, 176) features + (T, 3) relative BP
    targets + (T,) SQI weights.

    Loads pre-built .npz files from data_dir. Soft SQI weighting (Tier 1.5):
    the hard `sqi_threshold` still drops truly garbage windows (defaults to a
    very low value now), and the surviving windows propagate their SQI value
    through the data pipeline so the loss can soft-weight by quality.
    Relativizes targets by subtracting the per-day mean (Radha §3.2.5).

    Tier 1.6: optional Gaussian feature-jitter augmentation (training only).
    """

    def __init__(
        self,
        npz_paths: list[Path],
        sqi_threshold: float = 0.3,
        min_windows: int = 3,
        feat_jitter_std: float = 0.0,    # >0 only for training set
    ) -> None:
        self.items: list[dict[str, np.ndarray]] = []
        self.feat_jitter_std = float(feat_jitter_std)
        skipped = 0
        for p in npz_paths:
            data = np.load(p, allow_pickle=True)
            features = data["features"].astype(np.float32)   # (T, 176)
            targets  = data["targets"].astype(np.float32)    # (T, 3)
            sqi      = data["sqi"].astype(np.float32) if "sqi" in data else np.ones(len(features), dtype=np.float32)
            pid      = str(data["participant_id"]) if "participant_id" in data else p.stem.rsplit("_", 1)[0]

            mask = (sqi >= sqi_threshold) & np.all(np.isfinite(targets), axis=1)
            features = features[mask]
            targets  = targets[mask]
            sqi      = sqi[mask]

            if len(features) < min_windows:
                skipped += 1
                continue

            daily_mean = targets.mean(axis=0, keepdims=True)
            rel_targets = targets - daily_mean

            self.items.append({
                "features": features, "targets": rel_targets,
                "sqi": sqi, "participant_id": pid,
            })

        if skipped:
            logger.info("Skipped %d day-files with < %d valid windows", skipped, min_windows)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        feats = item["features"]
        if self.feat_jitter_std > 0.0:
            # Per-call Gaussian noise — different every epoch (Tier 1.6 augmentation)
            feats = feats + np.random.normal(0.0, self.feat_jitter_std, feats.shape).astype(np.float32)
        return (
            torch.from_numpy(feats),
            torch.from_numpy(item["targets"]),
            torch.from_numpy(item["sqi"]),
        )


def collate_days(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length day sequences. Returns (features, targets, sqi, lengths)."""
    features, targets, sqis = zip(*batch)
    lengths = torch.tensor([len(f) for f in features], dtype=torch.long)
    order = lengths.argsort(descending=True)
    lengths = lengths[order]
    features_padded = pad_sequence([features[i] for i in order], batch_first=True)  # (B, T_max, 176)
    targets_padded  = pad_sequence([targets[i]  for i in order], batch_first=True)  # (B, T_max, 3)
    sqi_padded      = pad_sequence([sqis[i]     for i in order], batch_first=True)  # (B, T_max)
    return features_padded, targets_padded, sqi_padded, lengths


# ── Training utilities ────────────────────────────────────────────────────────

def _subject_split(
    npz_paths: list[Path],
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Subject-disjoint train/val split on participant IDs embedded in filenames.

    Expects filenames of the form  <participant_id>_<date>.npz.
    """
    rng = np.random.default_rng(seed)
    # Extract unique participant IDs
    pids = sorted({p.stem.rsplit("_", 1)[0] for p in npz_paths})
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_fraction))
    val_pids = set(pids[:n_val])
    train = [p for p in npz_paths if p.stem.rsplit("_", 1)[0] not in val_pids]
    val   = [p for p in npz_paths if p.stem.rsplit("_", 1)[0] in val_pids]
    return train, val


def _masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sqi: torch.Tensor,
    lengths: torch.Tensor,
    sbp_mean_train: float,
) -> torch.Tensor:
    """Amplified, SQI-weighted MSE loss, masked to valid (non-padding) steps.

    pred, target: (B, T_max, 3)
    sqi:          (B, T_max)
    lengths:      (B,) number of real time steps per sequence

    Tier 1.5: each window's contribution to the loss is weighted by its SQI
    value (clipped to [0.1, 1.0] so low-SQI windows still contribute a little).
    Radha's amplification by |SBP - mean_SBP_training| is preserved.
    """
    B, T_max, _ = pred.shape
    valid_mask = torch.arange(T_max, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)

    sqi_weight = torch.clamp(sqi, min=0.1, max=1.0)                    # (B, T_max)
    mse = (pred - target) ** 2                                          # (B, T_max, 3)
    sbp_dev = torch.abs(target[..., 0] - sbp_mean_train).unsqueeze(-1)  # (B, T_max, 1)
    amplified = mse * (1.0 + sbp_dev)                                   # (B, T_max, 3)
    weighted = amplified * sqi_weight.unsqueeze(-1)                     # (B, T_max, 3)
    weighted = weighted * valid_mask.unsqueeze(-1).float()
    # Normalize by sum of (weights × 3 outputs) over valid steps
    denom = (sqi_weight * valid_mask.float()).sum() * 3
    return weighted.sum() / torch.clamp(denom, min=1.0)


def _evaluate(
    model: RadhaLSTM,
    loader: DataLoader,
    sbp_mean_train: float,
    device: torch.device,
) -> dict[str, float]:
    """Compute val loss, SBP RMSE, DBP RMSE on a dataloader."""
    model.eval()
    total_loss = 0.0
    sbp_sq, dbp_sq, n = 0.0, 0.0, 0
    with torch.no_grad():
        for feat, tgt, sqi, lengths in loader:
            feat, tgt, sqi, lengths = (feat.to(device), tgt.to(device),
                                       sqi.to(device), lengths.to(device))
            pred = model(feat)
            total_loss += _masked_loss(pred, tgt, sqi, lengths, sbp_mean_train).item()
            mask = torch.arange(feat.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
            valid_pred = pred[mask]
            valid_tgt  = tgt[mask]
            sbp_sq += ((valid_pred[:, 0] - valid_tgt[:, 0]) ** 2).sum().item()
            dbp_sq += ((valid_pred[:, 1] - valid_tgt[:, 1]) ** 2).sum().item()
            n += valid_pred.shape[0]

    n = max(n, 1)
    return {
        "val_loss": total_loss / max(len(loader), 1),
        "sbp_rmse": (sbp_sq / n) ** 0.5,
        "dbp_rmse": (dbp_sq / n) ** 0.5,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: TrainConfig) -> Path:
    """Train Radha LSTM from pre-built .npz day-files.

    Returns path to the saved best-model checkpoint.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    # Locate data files
    data_dir = Path(config.data_dir)
    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if config.val_subjects:
        val_set = set(config.val_subjects)
        train_paths = [p for p in npz_paths if p.stem.rsplit("_", 1)[0] not in val_set]
        val_paths = [p for p in npz_paths if p.stem.rsplit("_", 1)[0] in val_set]
        actually_held_out = {p.stem.rsplit("_", 1)[0] for p in val_paths}
        missing = val_set - actually_held_out
        logger.info(
            "Using explicit val_subjects override: %d subjects requested, %d present in data_dir (missing %d)",
            len(val_set), len(actually_held_out), len(missing),
        )
    else:
        train_paths, val_paths = _subject_split(npz_paths, config.val_fraction, config.seed)
    logger.info("Train: %d day-files (%d subjects); Val: %d day-files",
                len(train_paths), len({p.stem.rsplit("_",1)[0] for p in train_paths}),
                len(val_paths))

    train_ds = ParticipantDayDataset(
        train_paths, config.sqi_threshold, config.min_windows_per_day,
        feat_jitter_std=config.feat_jitter_std,   # Tier 1.6 — only on training set
    )
    val_ds   = ParticipantDayDataset(
        val_paths, config.sqi_threshold, config.min_windows_per_day,
        feat_jitter_std=0.0,                      # never jitter val
    )
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after quality filtering")

    # ── Tier 1: cohort-level NaN imputation + drop zero-std features + z-score ──
    # Adapter writes features with NaN preserved (where extraction failed).
    # Compute training-cohort medians, impute NaN in train AND val with those medians,
    # then identify features with zero variance across training and drop them,
    # then z-score remaining features. Save all three things (medians, kept_indices,
    # feat_mean/feat_std) with the checkpoint so estimator.py applies the same
    # transformation at LEAP inference time.
    all_train_raw = np.concatenate(
        [item["features"] for item in train_ds.items], axis=0
    )  # (total_T, 174)
    # 1. Cohort-level NaN imputation (median per feature across training set)
    feat_medians = np.nanmedian(all_train_raw, axis=0).astype(np.float32)  # (174,)
    # If a feature is NaN even after pooling all training data, fall back to 0
    feat_medians = np.where(np.isnan(feat_medians), 0.0, feat_medians).astype(np.float32)
    n_always_nan_features = int(np.isnan(np.nanmedian(all_train_raw, axis=0)).sum())
    for item in (train_ds.items + val_ds.items):
        feats = item["features"]
        nan_mask = np.isnan(feats)
        if nan_mask.any():
            idx_col = np.tile(np.arange(feats.shape[1]), (feats.shape[0], 1))
            item["features"] = np.where(nan_mask, feat_medians[idx_col], feats).astype(np.float32)
    # Recompute pooled features after imputation
    all_train_imputed = np.concatenate(
        [item["features"] for item in train_ds.items], axis=0
    )
    # 2. Identify zero-std features in training data; drop those columns
    full_std = all_train_imputed.std(axis=0)
    kept_indices = np.where(full_std >= config.feat_zero_std_eps)[0].astype(np.int64)
    n_dropped = 174 - len(kept_indices)
    if n_dropped > 0:
        logger.info("Dropping %d zero-std features (kept %d / 174)", n_dropped, len(kept_indices))
        for item in (train_ds.items + val_ds.items):
            item["features"] = item["features"][:, kept_indices]
        all_train_imputed = all_train_imputed[:, kept_indices]
    # 3. Z-score on the surviving features (training-only stats)
    feat_mean = all_train_imputed.mean(axis=0).astype(np.float32)
    feat_std = all_train_imputed.std(axis=0).astype(np.float32)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    for item in (train_ds.items + val_ds.items):
        item["features"] = ((item["features"] - feat_mean) / feat_std).astype(np.float32)
    # Override n_features in the model construction below
    actual_n_features = len(kept_indices)
    logger.info(
        "Tier 1 preprocessing: %d always-NaN features imputed via cohort median; "
        "%d zero-std features dropped; %d features remain. "
        "z-score mean range [%.3g, %.3g], std range [%.3g, %.3g]",
        n_always_nan_features, n_dropped, actual_n_features,
        feat_mean.min(), feat_mean.max(), feat_std.min(), feat_std.max(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size_days, shuffle=True,
        collate_fn=collate_days, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size_days, shuffle=False,
        collate_fn=collate_days, num_workers=0,
    )

    # Model + optimizer — actual_n_features comes from kept_indices after
    # zero-std drop, may be < config.n_features
    model = RadhaLSTM(
        n_features=actual_n_features,
        lstm_cells=config.lstm_cells,
        dense_hidden=config.dense_hidden,
        activation=config.dense_activation,
        dropout=config.dropout,
        bidirectional=config.bidirectional,       # Tier 1.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # Tier 1.5 — drop LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.lr_min,
    )

    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best_model.pt"

    # Git SHA for reproducibility
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=Path(__file__).parent.parent,
                                          text=True).strip()
    except Exception:
        git_sha = "unknown"

    # MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run():
        # Log all config params
        mlflow.log_params({
            "n_features":     config.n_features,
            "actual_n_features": actual_n_features,
            "n_dropped_features": n_dropped,
            "n_always_nan_features": n_always_nan_features,
            "lstm_cells":     config.lstm_cells,
            "dense_hidden":   config.dense_hidden,
            "activation":     config.dense_activation,
            "dropout":        config.dropout,
            "grad_clip_norm": config.grad_clip_norm,
            "batch_size":     config.batch_size_days,
            "lr":             config.lr,
            "weight_decay":   config.weight_decay,
            "seed":           config.seed,
            "git_sha":        git_sha,
            "n_train_days":   len(train_ds),
            "n_val_days":     len(val_ds),
        })

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, config.max_epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for feat, tgt, sqi, lengths in train_loader:
                feat, tgt, sqi, lengths = (feat.to(device), tgt.to(device),
                                            sqi.to(device), lengths.to(device))
                optimizer.zero_grad()
                pred = model(feat)
                loss = _masked_loss(pred, tgt, sqi, lengths, config.sbp_mean_train)
                loss.backward()
                # Tier 1: configurable gradient clipping (LSTM stability)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            # ── Validate ─────────────────────────────────────────────────────
            metrics = _evaluate(model, val_loader, config.sbp_mean_train, device)
            val_loss = metrics["val_loss"]

            # Tier 1.5 — adjust LR if val plateaus
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "sbp_rmse":   metrics["sbp_rmse"],
                    "dbp_rmse":   metrics["dbp_rmse"],
                    "lr":         current_lr,
                },
                step=epoch,
            )

            if epoch % 10 == 0 or epoch <= 5:
                logger.info(
                    "Epoch %3d | train=%.4f | val=%.4f | sbp_rmse=%.2f | dbp_rmse=%.2f",
                    epoch, train_loss, val_loss, metrics["sbp_rmse"], metrics["dbp_rmse"],
                )

            # ── Early stopping ───────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss":         val_loss,
                        "config":           config.__dict__,
                        "git_sha":          git_sha,
                        # Tier 1 preprocessing artifacts (estimator.py must apply
                        # these in order: 1) impute NaN with feat_medians, 2)
                        # select columns by kept_indices, 3) z-score with feat_mean/std)
                        "feat_medians":     feat_medians,
                        "kept_indices":     kept_indices,
                        "feat_mean":        feat_mean,
                        "feat_std":         feat_std,
                        "actual_n_features": actual_n_features,
                    },
                    best_ckpt,
                )
                mlflow.log_artifact(str(best_ckpt))
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    logger.info("Early stopping at epoch %d (patience=%d)",
                                epoch, config.early_stop_patience)
                    break

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("stopped_epoch", epoch)
        logger.info("Training complete. Best val_loss=%.4f, saved to %s", best_val_loss, best_ckpt)

        # ── Tier 1.6: per-subject loss tracking on val ────────────────────
        # Reload best checkpoint and compute per-subject SBP/DBP RMSE on val.
        # Flags subjects whose RMSE is > 2x the cohort median — these are
        # outliers worth eyeballing (e.g. p002636-style hemodynamic instability).
        try:
            best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(best_state["model_state_dict"])
            model.eval()
            per_subject = _per_subject_metrics(model, val_ds, device)
            _log_per_subject_outliers(per_subject, output_dir, mlflow)
        except Exception as exc:  # noqa: BLE001 — never fail training due to logging
            logger.warning("Per-subject loss tracking failed: %s", exc)

    return best_ckpt


def _per_subject_metrics(
    model: RadhaLSTM,
    dataset: ParticipantDayDataset,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compute per-subject SBP / DBP RMSE on a dataset. Returns dict keyed by
    participant_id. Used after training to flag outlier subjects (Tier 1.6)."""
    per_subj_sq_sbp: dict[str, list[float]] = {}
    per_subj_sq_dbp: dict[str, list[float]] = {}
    with torch.no_grad():
        for item in dataset.items:
            pid = str(item.get("participant_id", "unknown"))
            feats = torch.from_numpy(item["features"]).unsqueeze(0).to(device)  # (1, T, F)
            tgt = item["targets"]  # (T, 3)
            pred = model(feats).squeeze(0).cpu().numpy()  # (T, 3)
            per_subj_sq_sbp.setdefault(pid, []).extend(
                ((pred[:, 0] - tgt[:, 0]) ** 2).tolist()
            )
            per_subj_sq_dbp.setdefault(pid, []).extend(
                ((pred[:, 1] - tgt[:, 1]) ** 2).tolist()
            )
    return {
        pid: {
            "sbp_rmse": float(np.sqrt(np.mean(per_subj_sq_sbp[pid]))),
            "dbp_rmse": float(np.sqrt(np.mean(per_subj_sq_dbp[pid]))),
            "n_windows": int(len(per_subj_sq_sbp[pid])),
        }
        for pid in sorted(per_subj_sq_sbp)
    }


def _log_per_subject_outliers(
    per_subject: dict[str, dict[str, float]],
    output_dir: Path,
    mlflow_ctx,
) -> None:
    """Write per-subject metrics to JSON and flag outliers (>2x median RMSE)."""
    import json as _json
    out_path = output_dir / "per_subject_metrics.json"
    with open(out_path, "w") as fh:
        _json.dump(per_subject, fh, indent=2)
    mlflow_ctx.log_artifact(str(out_path))
    if not per_subject:
        return
    sbp_rmses = np.array([m["sbp_rmse"] for m in per_subject.values()])
    median_sbp = float(np.median(sbp_rmses))
    outlier_pids = [
        pid for pid, m in per_subject.items()
        if m["sbp_rmse"] > 2.0 * median_sbp
    ]
    logger.info("Per-subject SBP RMSE: median=%.2f, max=%.2f, outliers (>2x median): %s",
                median_sbp, float(sbp_rmses.max()), outlier_pids)
    mlflow_ctx.log_metric("per_subj_median_sbp_rmse", median_sbp)
    mlflow_ctx.log_metric("per_subj_max_sbp_rmse", float(sbp_rmses.max()))
    mlflow_ctx.log_metric("per_subj_n_outliers", len(outlier_pids))


# ── Feature-cache builder ─────────────────────────────────────────────────────

def build_day_npz(
    recording,                  # LEAPRecording
    cuff_readings: list,        # list[CuffReading]
    output_path: Path,
    preprocess_config=None,
    max_lag_seconds: float = 900.0,
) -> Path:
    """Build a .npz feature-cache for one participant-day.

    For each cuff reading within the recording window, extracts a 5-minute PPG/accel
    window centered on the cuff timestamp, computes 195 features (then drops the
    21 activity features for compatibility with the 174-feature LSTM — see
    OPEN_QUESTIONS Q9), and pairs with [SBP, DBP, MAP] targets.  Saves to output_path.

    Args:
        recording:         LEAPRecording (from load_leap_recording)
        cuff_readings:     list of CuffReading for this participant-day
        output_path:       where to save the .npz
        preprocess_config: PreprocessConfig; defaults to fs=25, 5-min window, z-score
        max_lag_seconds:   discard cuff readings with no PPG data within this lag

    Returns:
        output_path (for chaining)
    """
    from common.io import align_cuff_to_windows
    from common.preprocessing import PreprocessConfig, SQIMethod, compute_sqi
    from path_a_radha.features import compute_rest_probabilities, extract_features

    if preprocess_config is None:
        preprocess_config = PreprocessConfig(
            fs=recording.fs,
            window_seconds=300.0,   # 5 minutes — Radha §3.2.5
            overlap=0.0,            # non-overlapping for cuff-aligned windows
            sqi_method=SQIMethod.TEMPLATE,
        )

    windows = align_cuff_to_windows(
        recording, cuff_readings,
        window_seconds=preprocess_config.window_seconds,
        max_lag_seconds=max_lag_seconds,
    )
    if not windows:
        raise ValueError("No aligned windows found (no cuff readings within the recording window)")

    feature_rows, target_rows, sqi_rows = [], [], []
    fs = recording.fs

    # ENMO per window for rest probability
    enmo_per_win = [
        float(np.maximum(0.0, np.linalg.norm(w.accel, axis=1) - 1.0).mean())
        for w in windows
    ]
    rest_probs = compute_rest_probabilities(np.array(enmo_per_win))

    from common.preprocessing import bandpass_filter, normalize_window, NormMethod
    for i, w in enumerate(windows):
        ppg = w.ppg.astype(np.float64)
        try:
            ppg_filt = bandpass_filter(ppg, fs, 0.5, 4.0)
        except ValueError:
            ppg_filt = ppg
        sqi = compute_sqi(ppg_filt, preprocess_config.sqi_method, fs=fs)
        ppg_norm = normalize_window(ppg_filt, NormMethod.ZSCORE)

        feats = extract_features(ppg_norm, w.accel, fs=fs, rest_prob=float(rest_probs[i]))
        feature_rows.append(feats)
        sqi_rows.append(sqi)

        sbp = w.cuff.sbp_mmhg
        dbp = w.cuff.dbp_mmhg
        map_ = (sbp + 2 * dbp) / 3.0
        target_rows.append([sbp, dbp, map_])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=np.array(feature_rows, dtype=np.float32),
        targets=np.array(target_rows, dtype=np.float32),
        sqi=np.array(sqi_rows, dtype=np.float32),
        participant_id=recording.metadata.participant_id,
    )
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(TrainConfig())
