"""Evaluate a trained Path A checkpoint on its val split.

Reproduces the train/val split (subject-disjoint, seed=42) so we can run
inference on exactly the held-out subjects. Computes:

  - Per-subject Pearson + Spearman correlation of predicted vs actual relative
    SBP/DBP/MAP (PRIMARY metric per CLAUDE.md / PI Garrett Ash)
  - Per-subject RMSE
  - Pooled population MAE/RMSE/Bland-Altman/BHS
  - Trend-direction agreement (consecutive-window sign matches)
  - MC dropout uncertainty bands

Saves:
  - benchmark/results/<run_dir>/eval/per_subject_metrics_v2.json
  - benchmark/results/<run_dir>/eval/pooled_summary.json
  - benchmark/results/<run_dir>/eval/per_subject_timeseries/<pid>.png
  - benchmark/results/<run_dir>/eval/scatter_pred_vs_actual.png
  - benchmark/results/<run_dir>/eval/bland_altman.png

Usage:
    python -m path_a_radha.evaluate \\
        --checkpoint benchmark/results/path_a_first_run/best_model.pt \\
        --data-dir   data/processed/path_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.eval import (
    bhs_grade,
    bland_altman,
    within_person_correlations,
)
from path_a_radha.model import RadhaLSTM, enable_mc_dropout
from path_a_radha.train import (
    ParticipantDayDataset,
    TrainConfig,
    _subject_split,
)


def load_model_and_preprocessing(ckpt_path: Path, device: str = "cpu") -> dict:
    """Load checkpoint and reconstruct model + preprocessing artifacts."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = RadhaLSTM(
        n_features=int(ckpt["actual_n_features"]),
        lstm_cells=cfg.get("lstm_cells", 32),
        dense_hidden=cfg.get("dense_hidden", 8),
        activation=cfg.get("dense_activation", "relu"),
        dropout=cfg.get("dropout", 0.2),
        bidirectional=cfg.get("bidirectional", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "feat_medians": np.asarray(ckpt["feat_medians"]),
        "kept_indices": np.asarray(ckpt["kept_indices"]),
        "feat_mean":    np.asarray(ckpt["feat_mean"]),
        "feat_std":     np.asarray(ckpt["feat_std"]),
        "epoch":        ckpt.get("epoch"),
        "val_loss":     ckpt.get("val_loss"),
    }


def preprocess_features(features_raw: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the training-time preprocessing chain to a (T, 176) feature array:
    impute NaN with cohort medians → select kept_indices → z-score.
    """
    nan_mask = np.isnan(features_raw)
    if nan_mask.any():
        idx_col = np.tile(np.arange(features_raw.shape[1]), (features_raw.shape[0], 1))
        features_raw = np.where(nan_mask, stats["feat_medians"][idx_col], features_raw)
    feats_kept = features_raw[:, stats["kept_indices"]]
    return ((feats_kept - stats["feat_mean"]) / stats["feat_std"]).astype(np.float32)


def predict_with_mc_dropout(
    model: RadhaLSTM,
    features: np.ndarray,
    n_mc: int = 30,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean (T, 3), std (T, 3)) using MC dropout."""
    model.eval()
    enable_mc_dropout(model)
    x = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, F)
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            preds.append(model(x).squeeze(0).cpu().numpy())   # (T, 3)
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def trend_direction_agreement(pred: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of consecutive-window steps where pred and ref move in same direction."""
    if len(pred) < 2:
        return float("nan")
    pred_diff = np.diff(pred.astype(np.float64))
    ref_diff = np.diff(ref.astype(np.float64))
    # Only count steps where ref actually moves (skip zero-diff = ambiguous)
    nonzero = np.abs(ref_diff) > 1e-9
    if nonzero.sum() == 0:
        return float("nan")
    same_sign = (np.sign(pred_diff[nonzero]) == np.sign(ref_diff[nonzero])).mean()
    return float(same_sign)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to <checkpoint-parent>/eval/")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match TrainConfig.seed to reproduce val split")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-mc", type=int, default=30)
    parser.add_argument("--sqi-threshold", type=float, default=0.3)
    parser.add_argument("--min-windows", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.checkpoint.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = out_dir / "per_subject_timeseries"
    ts_dir.mkdir(exist_ok=True)

    # ── Reproduce the val split ──
    npz_paths = sorted(args.data_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files in {args.data_dir}")
    train_paths, val_paths = _subject_split(npz_paths, args.val_fraction, args.seed)
    print(f"Loaded {len(npz_paths)} day-files; reproduced val split with seed={args.seed}: "
          f"{len(train_paths)} train / {len(val_paths)} val")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Load model + preprocessing stats ──
    stats = load_model_and_preprocessing(args.checkpoint)
    print(f"  best_epoch={stats['epoch']}, val_loss={stats['val_loss']:.2f}, "
          f"n_features={stats['feat_mean'].shape[0]}")

    # ── Use the dataset class to do SQI filtering + relativization, but bypass
    #    the training-time z-score (which is already in the dataset's items).
    #    We need raw features then apply our load_model_and_preprocessing chain.
    # The cleanest approach: load each val .npz directly, apply preprocessing.
    all_pred_sbp, all_pred_dbp, all_pred_map = [], [], []
    all_ref_sbp,  all_ref_dbp,  all_ref_map  = [], [], []
    all_pred_sbp_std, all_pred_dbp_std       = [], []
    all_pid: list[str] = []
    per_subject: dict[str, dict] = {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    print()
    print(f"Evaluating {len(val_paths)} val records with {args.n_mc} MC samples each...")
    for path in val_paths:
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype(np.float32)  # (T, 176)
        targets  = data["targets"].astype(np.float32)   # (T, 3) RELATIVE
        sqi      = data["sqi"].astype(np.float32) if "sqi" in data else np.ones(len(features))
        pid      = str(data["participant_id"]) if "participant_id" in data else path.stem.rsplit("_", 1)[0]

        # Same QC as dataset
        mask = (sqi >= args.sqi_threshold) & np.all(np.isfinite(targets), axis=1)
        features = features[mask]
        targets  = targets[mask]
        if len(features) < args.min_windows:
            continue

        # Relativize targets per record (matches train.py)
        targets = targets - targets.mean(axis=0, keepdims=True)

        # Preprocess + inference
        feats_normed = preprocess_features(features, stats)
        pred_mean, pred_std = predict_with_mc_dropout(stats["model"], feats_normed, n_mc=args.n_mc)

        # Per-subject metrics
        try:
            pr_sbp, _ = pearsonr(pred_mean[:, 0], targets[:, 0])
            sr_sbp, _ = spearmanr(pred_mean[:, 0], targets[:, 0])
            pr_dbp, _ = pearsonr(pred_mean[:, 1], targets[:, 1])
            sr_dbp, _ = spearmanr(pred_mean[:, 1], targets[:, 1])
        except Exception:
            pr_sbp = sr_sbp = pr_dbp = sr_dbp = float("nan")
        rmse_sbp = float(np.sqrt(np.mean((pred_mean[:, 0] - targets[:, 0]) ** 2)))
        rmse_dbp = float(np.sqrt(np.mean((pred_mean[:, 1] - targets[:, 1]) ** 2)))
        trend_sbp = trend_direction_agreement(pred_mean[:, 0], targets[:, 0])
        trend_dbp = trend_direction_agreement(pred_mean[:, 1], targets[:, 1])

        per_subject[pid] = {
            "n_windows":          int(len(features)),
            "pearson_sbp":        float(pr_sbp),
            "spearman_sbp":       float(sr_sbp),
            "rmse_sbp":           rmse_sbp,
            "trend_agreement_sbp": trend_sbp,
            "pearson_dbp":        float(pr_dbp),
            "spearman_dbp":       float(sr_dbp),
            "rmse_dbp":           rmse_dbp,
            "trend_agreement_dbp": trend_dbp,
            "mean_pred_uncertainty_sbp": float(pred_std[:, 0].mean()),
            "mean_pred_uncertainty_dbp": float(pred_std[:, 1].mean()),
        }

        # Accumulate pooled
        all_pred_sbp.extend(pred_mean[:, 0]); all_pred_dbp.extend(pred_mean[:, 1]); all_pred_map.extend(pred_mean[:, 2])
        all_ref_sbp.extend(targets[:, 0]);    all_ref_dbp.extend(targets[:, 1]);    all_ref_map.extend(targets[:, 2])
        all_pred_sbp_std.extend(pred_std[:, 0]); all_pred_dbp_std.extend(pred_std[:, 1])
        all_pid.extend([pid] * len(features))

        # Time-series plot for this subject
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        t_min = np.arange(len(features)) * 5  # 5-min step
        ax[0].plot(t_min, targets[:, 0], label="actual rel SBP", color="C0", lw=1.5)
        ax[0].plot(t_min, pred_mean[:, 0], label="predicted rel SBP", color="C3", lw=1.2)
        ax[0].fill_between(t_min, pred_mean[:, 0] - pred_std[:, 0], pred_mean[:, 0] + pred_std[:, 0],
                           color="C3", alpha=0.2, label="MC dropout ±1σ")
        ax[0].set_ylabel("rel SBP (mmHg)"); ax[0].legend(loc="upper right", fontsize=8); ax[0].grid(alpha=0.3)
        ax[0].set_title(f"{pid}  •  Pearson r SBP={pr_sbp:.3f}  •  trend-agree SBP={trend_sbp:.2f}")
        ax[1].plot(t_min, targets[:, 1], label="actual rel DBP", color="C0", lw=1.5)
        ax[1].plot(t_min, pred_mean[:, 1], label="predicted rel DBP", color="C3", lw=1.2)
        ax[1].fill_between(t_min, pred_mean[:, 1] - pred_std[:, 1], pred_mean[:, 1] + pred_std[:, 1],
                           color="C3", alpha=0.2)
        ax[1].set_xlabel("minutes from record start"); ax[1].set_ylabel("rel DBP (mmHg)")
        ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=0.3)
        ax[1].set_title(f"Pearson r DBP={pr_dbp:.3f}  •  trend-agree DBP={trend_dbp:.2f}")
        fig.tight_layout()
        fig.savefig(ts_dir / f"{pid}.png", dpi=110)
        plt.close(fig)

    # ── Pooled summary ──
    pred_sbp = np.array(all_pred_sbp); ref_sbp = np.array(all_ref_sbp)
    pred_dbp = np.array(all_pred_dbp); ref_dbp = np.array(all_ref_dbp)
    pids_arr = np.array(all_pid)

    pearson_sbp, spearman_sbp = within_person_correlations(pred_sbp, ref_sbp, pids_arr)
    pearson_dbp, spearman_dbp = within_person_correlations(pred_dbp, ref_dbp, pids_arr)
    err_sbp = pred_sbp - ref_sbp
    err_dbp = pred_dbp - ref_dbp
    ba_sbp = bland_altman(pred_sbp, ref_sbp)
    ba_dbp = bland_altman(pred_dbp, ref_dbp)

    summary = {
        "n_val_records":   len(per_subject),
        "n_val_windows":   int(len(pred_sbp)),
        "checkpoint":      str(args.checkpoint),
        "best_epoch":      stats["epoch"],
        "val_loss":        float(stats["val_loss"]),
        # Per-subject (primary)
        "pearson_sbp_median": float(np.median(pearson_sbp)) if len(pearson_sbp) else float("nan"),
        "pearson_sbp_q25":    float(np.percentile(pearson_sbp, 25)) if len(pearson_sbp) else float("nan"),
        "pearson_sbp_q75":    float(np.percentile(pearson_sbp, 75)) if len(pearson_sbp) else float("nan"),
        "spearman_sbp_median": float(np.median(spearman_sbp)) if len(spearman_sbp) else float("nan"),
        "pearson_dbp_median": float(np.median(pearson_dbp)) if len(pearson_dbp) else float("nan"),
        "pearson_dbp_q25":    float(np.percentile(pearson_dbp, 25)) if len(pearson_dbp) else float("nan"),
        "pearson_dbp_q75":    float(np.percentile(pearson_dbp, 75)) if len(pearson_dbp) else float("nan"),
        "spearman_dbp_median": float(np.median(spearman_dbp)) if len(spearman_dbp) else float("nan"),
        # Trend-direction agreement
        "trend_agree_sbp_median": float(np.median([m["trend_agreement_sbp"] for m in per_subject.values()
                                                   if np.isfinite(m["trend_agreement_sbp"])])),
        "trend_agree_dbp_median": float(np.median([m["trend_agreement_dbp"] for m in per_subject.values()
                                                   if np.isfinite(m["trend_agreement_dbp"])])),
        # Pooled
        "sbp_mae":       float(np.mean(np.abs(err_sbp))),
        "sbp_rmse":      float(np.sqrt(np.mean(err_sbp ** 2))),
        "sbp_me":        float(np.mean(err_sbp)),
        "sbp_ba_mean":   ba_sbp[0], "sbp_ba_loa_upper": ba_sbp[1], "sbp_ba_loa_lower": ba_sbp[2],
        "sbp_bhs":       bhs_grade(err_sbp),
        "dbp_mae":       float(np.mean(np.abs(err_dbp))),
        "dbp_rmse":      float(np.sqrt(np.mean(err_dbp ** 2))),
        "dbp_me":        float(np.mean(err_dbp)),
        "dbp_ba_mean":   ba_dbp[0], "dbp_ba_loa_upper": ba_dbp[1], "dbp_ba_loa_lower": ba_dbp[2],
        "dbp_bhs":       bhs_grade(err_dbp),
        # MC dropout uncertainty
        "mean_uncertainty_sbp": float(np.mean(all_pred_sbp_std)),
        "mean_uncertainty_dbp": float(np.mean(all_pred_dbp_std)),
    }

    with open(out_dir / "per_subject_metrics_v2.json", "w") as fh:
        json.dump(per_subject, fh, indent=2)
    with open(out_dir / "pooled_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── Aggregate plots ──
    # Scatter pred vs actual
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(ref_sbp, pred_sbp, s=8, alpha=0.4)
    lim = [min(ref_sbp.min(), pred_sbp.min()), max(ref_sbp.max(), pred_sbp.max())]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("actual rel SBP (mmHg)"); ax[0].set_ylabel("predicted rel SBP (mmHg)")
    ax[0].set_title(f"SBP: pooled (n={len(pred_sbp)})\nPearson r median = {summary['pearson_sbp_median']:.3f}")
    ax[0].grid(alpha=0.3)
    ax[1].scatter(ref_dbp, pred_dbp, s=8, alpha=0.4, color="C2")
    lim = [min(ref_dbp.min(), pred_dbp.min()), max(ref_dbp.max(), pred_dbp.max())]
    ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("actual rel DBP (mmHg)"); ax[1].set_ylabel("predicted rel DBP (mmHg)")
    ax[1].set_title(f"DBP: pooled\nPearson r median = {summary['pearson_dbp_median']:.3f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "scatter_pred_vs_actual.png", dpi=120); plt.close(fig)

    # Bland-Altman
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    mean_sbp = (pred_sbp + ref_sbp) / 2
    diff_sbp = pred_sbp - ref_sbp
    ax[0].scatter(mean_sbp, diff_sbp, s=8, alpha=0.4)
    ax[0].axhline(ba_sbp[0], color="C3", lw=1.5, label=f"mean diff {ba_sbp[0]:+.1f}")
    ax[0].axhline(ba_sbp[1], color="C3", lw=1, ls="--", label=f"+1.96 SD {ba_sbp[1]:+.1f}")
    ax[0].axhline(ba_sbp[2], color="C3", lw=1, ls="--", label=f"-1.96 SD {ba_sbp[2]:+.1f}")
    ax[0].set_xlabel("(pred + ref) / 2 (mmHg)"); ax[0].set_ylabel("pred - ref (mmHg)")
    ax[0].set_title(f"Bland-Altman SBP — BHS Grade {summary['sbp_bhs']}")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    mean_dbp = (pred_dbp + ref_dbp) / 2
    diff_dbp = pred_dbp - ref_dbp
    ax[1].scatter(mean_dbp, diff_dbp, s=8, alpha=0.4, color="C2")
    ax[1].axhline(ba_dbp[0], color="C3", lw=1.5, label=f"mean diff {ba_dbp[0]:+.1f}")
    ax[1].axhline(ba_dbp[1], color="C3", lw=1, ls="--", label=f"+1.96 SD {ba_dbp[1]:+.1f}")
    ax[1].axhline(ba_dbp[2], color="C3", lw=1, ls="--", label=f"-1.96 SD {ba_dbp[2]:+.1f}")
    ax[1].set_xlabel("(pred + ref) / 2 (mmHg)"); ax[1].set_ylabel("pred - ref (mmHg)")
    ax[1].set_title(f"Bland-Altman DBP — BHS Grade {summary['dbp_bhs']}")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "bland_altman.png", dpi=120); plt.close(fig)

    # Per-subject Pearson r histograms
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(pearson_sbp, bins=15, edgecolor="black")
    ax[0].axvline(np.median(pearson_sbp), color="C3", lw=2, label=f"median={np.median(pearson_sbp):.3f}")
    ax[0].set_xlabel("per-subject Pearson r (SBP)"); ax[0].set_ylabel("# subjects")
    ax[0].set_title(f"Within-person trend correlation — SBP (n={len(pearson_sbp)})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].hist(pearson_dbp, bins=15, edgecolor="black", color="C2")
    ax[1].axvline(np.median(pearson_dbp), color="C3", lw=2, label=f"median={np.median(pearson_dbp):.3f}")
    ax[1].set_xlabel("per-subject Pearson r (DBP)"); ax[1].set_ylabel("# subjects")
    ax[1].set_title(f"Within-person trend correlation — DBP")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "per_subject_pearson_hist.png", dpi=120); plt.close(fig)

    print()
    print("=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Val records (subjects): {summary['n_val_records']}")
    print(f"  Val windows:            {summary['n_val_windows']}")
    print(f"  Checkpoint epoch:       {summary['best_epoch']}")
    print(f"  Best val_loss:          {summary['val_loss']:.2f}")
    print()
    print("PRIMARY METRIC — Within-person Pearson r (median [Q25, Q75]):")
    print(f"  SBP: {summary['pearson_sbp_median']:+.3f} "
          f"[{summary['pearson_sbp_q25']:+.3f}, {summary['pearson_sbp_q75']:+.3f}]")
    print(f"  DBP: {summary['pearson_dbp_median']:+.3f} "
          f"[{summary['pearson_dbp_q25']:+.3f}, {summary['pearson_dbp_q75']:+.3f}]")
    print()
    print("Trend-direction agreement (median across subjects):")
    print(f"  SBP: {summary['trend_agree_sbp_median']:.3f}  (0.5 = random)")
    print(f"  DBP: {summary['trend_agree_dbp_median']:.3f}")
    print()
    print("Pooled population metrics:")
    print(f"  SBP MAE = {summary['sbp_mae']:.2f}  RMSE = {summary['sbp_rmse']:.2f}  ME = {summary['sbp_me']:+.2f}  BHS = {summary['sbp_bhs']}")
    print(f"  DBP MAE = {summary['dbp_mae']:.2f}  RMSE = {summary['dbp_rmse']:.2f}  ME = {summary['dbp_me']:+.2f}  BHS = {summary['dbp_bhs']}")
    print()
    print(f"Bland-Altman SBP: mean diff = {summary['sbp_ba_mean']:+.2f}, "
          f"LoA = [{summary['sbp_ba_loa_lower']:+.2f}, {summary['sbp_ba_loa_upper']:+.2f}]")
    print(f"Bland-Altman DBP: mean diff = {summary['dbp_ba_mean']:+.2f}, "
          f"LoA = [{summary['dbp_ba_loa_lower']:+.2f}, {summary['dbp_ba_loa_upper']:+.2f}]")
    print()
    print(f"MC dropout uncertainty (mean ±σ): SBP={summary['mean_uncertainty_sbp']:.2f}, "
          f"DBP={summary['mean_uncertainty_dbp']:.2f}")
    print()
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
