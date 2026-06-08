"""Per-participant fine-tuning of the Phase 5 BP-trend model.

Freezes the LSTM + feature extraction; fits a tiny ridge-regression head
from the (n_windows, 64) bidirectional-LSTM output to the participant's
own (n_paired,) relative cuff SBP and DBP.  Uses leave-one-out
cross-validation to (a) select the ridge λ and (b) produce honest
out-of-sample Pearson r — the only fair way to evaluate a model trained
on n=12 samples.

Pipeline:

  PPG window  ──[frozen Phase 5 LSTM + dense head]──>  64-d embedding
                                                         │
                                       (12 cuff-matched embeddings)
                                                         │
                              ridge(α, scaled X), leave-one-out CV
                                                         │
                                       new linear head per participant
                                                         │
                          applied to all 995 embeddings  →  finetuned trace

The "fine-tuning" is intentionally minimal: just the output projection.
With only n=12 paired samples we cannot responsibly tune the LSTM itself
(it has ~10K params).  The 64-d LSTM output already encodes most of the
PPG-to-BP mapping; we're only learning *the participant-specific linear
combination* of those features.

Outputs (under ``--out-dir``):

  - finetune_metrics.json     vanilla vs LOO-finetuned r/p/CI per channel
  - finetune_comparison.md    side-by-side markdown report
  - finetuned_predictions.csv 995-window trace with the new linear head
  - finetune_diagnostics.png  scatter of LOO predictions vs cuff

Usage:

  python -m path_a_radha.finetune_participant \\
      --ppg-dir data/ppg-green-100-hz \\
      --checkpoint benchmark/results/path_a_phase5/best_model.pt \\
      --cuff benchmark/results/centrepoint_SUBJ01/cuff_readings.csv \\
      --participant-id SUBJ01 \\
      --out-dir benchmark/results/centrepoint_SUBJ01_finetune_phase5
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.io import read_cuff_bp
from common.io.centrepoint import load_centrepoint_recording
from path_a_radha.calibrate_from_predictions import (
    BOOTSTRAP_CI_PCT,
    BOOTSTRAP_RESAMPLES,
    MIN_N_FOR_RELIABLE,
    P_VALUE_THRESHOLD,
    RELIABLE_R_THRESHOLD,
    WINDOW_SECONDS,
    _bootstrap_r_ci,
    classify_reliability,
)
from path_a_radha.estimator import RadhaEstimator
from path_a_radha.run_leap_inference import find_continuous_bursts, slice_windows

logger = logging.getLogger(__name__)

RIDGE_LAMBDAS: np.ndarray = np.logspace(-3, 4, 36)


# ───────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ───────────────────────────────────────────────────────────────────────────

def extract_phase5_embeddings(
    ppg_files: list[Path],
    checkpoint: Path,
    participant_id: str,
    tz_offset: float,
    ppg_fs: int,
) -> tuple[np.ndarray, list[pd.Timestamp], np.ndarray, RadhaEstimator]:
    """Run frozen Phase 5 on each day's PPG; hook the LSTM to grab the
    (T, 64) bidirectional output that feeds the final linear layer.

    Returns:
        embeddings : (n_windows, 64) float32
        starts     : list of pd.Timestamp (window start UTC), len n_windows
        predictions: (n_windows, 3) float32 — original Phase 5 rel SBP/DBP/MAP
        estimator  : the RadhaEstimator (kept around for caller convenience)
    """
    # n_mc=0 → deterministic single forward pass per window (no MC dropout).
    estimator = RadhaEstimator(
        checkpoint_path=checkpoint,
        calibration_sbp_mean=0.0,
        calibration_dbp_mean=0.0,
        device="cpu",
        n_mc_samples=0,
    )

    captured: list[np.ndarray] = []

    def _lstm_hook(_module: torch.nn.Module, _inp: tuple, output: tuple) -> None:
        # output = (lstm_out, (h_n, c_n)); lstm_out is (batch, seq, 64)
        lstm_out = output[0].detach().cpu().numpy()
        # Drop batch dim (always 1 in inference)
        captured.append(lstm_out[0])

    handle = estimator.model.lstm.register_forward_hook(_lstm_hook)

    all_embeddings: list[np.ndarray] = []
    all_starts:     list[pd.Timestamp] = []
    all_predictions: list[np.ndarray] = []

    try:
        for ppg_file in ppg_files:
            rec = load_centrepoint_recording(
                ppg_paths=[ppg_file],
                participant_id=participant_id,
                tz_offset_hours=tz_offset,
                ppg_fs=ppg_fs,
            )
            estimator.cfg.fs = rec.fs
            bursts = find_continuous_bursts(rec.timestamps_utc, rec.fs)
            try:
                ppg_wins, accel_wins, starts = slice_windows(
                    rec.ppg_green, rec.accel, rec.timestamps_utc, fs=rec.fs,
                )
            except ValueError as exc:
                logger.warning("Skipping %s: %s", ppg_file.name, exc)
                continue

            window_starts_min = np.array(
                [(s - starts[0]).total_seconds() / 60.0 for s in starts],
                dtype=np.float32,
            )

            captured.clear()
            result = estimator.predict(
                ppg=ppg_wins, accel=accel_wins, fs=rec.fs,
                window_starts_min=window_starts_min,
            )
            # n_mc=0 should fire the hook exactly once per day
            if len(captured) != 1:
                # If MC was somehow on, average the captures
                day_emb = np.mean(np.stack(captured, axis=0), axis=0)
            else:
                day_emb = captured[0]
            assert day_emb.shape[0] == len(starts), (
                f"hook output windows {day_emb.shape[0]} != starts {len(starts)}"
            )

            logger.info(
                "  %s: %d windows, LSTM emb (%d, %d), SQI median %.2f",
                ppg_file.name, len(starts), *day_emb.shape, float(np.median(result["confidence"])),
            )
            all_embeddings.append(day_emb)
            all_starts.extend(starts)
            all_predictions.append(result["relative"])
    finally:
        handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    predictions = np.concatenate(all_predictions, axis=0).astype(np.float32)
    logger.info("Total: %d windows, embedding shape %s", len(all_starts), embeddings.shape)
    return embeddings, all_starts, predictions, estimator


# ───────────────────────────────────────────────────────────────────────────
# Cuff ↔ window matching
# ───────────────────────────────────────────────────────────────────────────

def match_cuff_to_windows(
    starts: list[pd.Timestamp],
    cuff_path: Path,
    participant_id: str,
    max_lag_seconds: float = 900.0,
) -> tuple[list[dict], np.ndarray]:
    """Match each cuff reading to its nearest PPG window centre.

    Returns:
        pairs : list of dicts (cuff timestamp, SBP, DBP, lag_s, matched bool, window_idx)
        idx_matched : np.ndarray[int] of length n_matched — indices into starts/embeddings
    """
    cuff = read_cuff_bp(cuff_path, participant_id=participant_id)
    if not cuff:
        raise SystemExit(f"No cuff readings for {participant_id!r} in {cuff_path}")

    win_starts_s = np.array([s.value / 1e9 for s in starts], dtype=np.float64)
    win_centres_s = win_starts_s + WINDOW_SECONDS / 2.0

    pairs: list[dict] = []
    idx_matched_l: list[int] = []
    for c in sorted(cuff, key=lambda x: x.timestamp_utc):
        c_s = pd.Timestamp(c.timestamp_utc).value / 1e9
        d = np.abs(win_centres_s - c_s)
        j = int(np.argmin(d))
        matched = bool(d[j] <= max_lag_seconds)
        pairs.append({
            "cuff_timestamp_utc": pd.Timestamp(c.timestamp_utc).isoformat(),
            "cuff_sbp_mmhg": float(c.sbp_mmhg),
            "cuff_dbp_mmhg": float(c.dbp_mmhg),
            "matched": matched,
            "lag_seconds": float(win_centres_s[j] - c_s) if matched else None,
            "window_idx": j if matched else None,
        })
        if matched:
            idx_matched_l.append(j)

    logger.info(
        "Cuff readings: %d, matched within ±%.0fs: %d",
        len(pairs), max_lag_seconds, len(idx_matched_l),
    )
    return pairs, np.array(idx_matched_l, dtype=int)


# ───────────────────────────────────────────────────────────────────────────
# Ridge LOO core
# ───────────────────────────────────────────────────────────────────────────

def fit_ridge_loo(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray = RIDGE_LAMBDAS,
) -> dict:
    """Ridge regression with leave-one-out λ selection.

    Returns dict with:
        best_lambda : float
        loo_preds   : (n,) honest out-of-sample predictions
        in_preds    : (n,) in-sample (full-fit) predictions — for diagnosing overfit
        full_pipe   : sklearn Pipeline fit on all data at best λ
        lambda_grid : (n_lambda,)
        loo_mse_grid: (n_lambda,)
    """
    n = len(y)
    loo = LeaveOneOut()
    loo_mse_grid = np.zeros(len(lambdas))
    loo_preds_grid = np.zeros((len(lambdas), n))

    for k, lam in enumerate(lambdas):
        preds = np.zeros(n)
        for train_idx, test_idx in loo.split(X):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=lam, fit_intercept=True)),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            preds[test_idx] = pipe.predict(X[test_idx])[0]
        loo_preds_grid[k] = preds
        loo_mse_grid[k] = float(np.mean((preds - y) ** 2))

    best_k = int(np.argmin(loo_mse_grid))
    best_lambda = float(lambdas[best_k])
    loo_preds = loo_preds_grid[best_k]

    full_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_lambda, fit_intercept=True)),
    ]).fit(X, y)
    in_preds = full_pipe.predict(X)

    return {
        "best_lambda":   best_lambda,
        "loo_preds":     loo_preds,
        "in_preds":      in_preds,
        "full_pipe":     full_pipe,
        "lambda_grid":   lambdas,
        "loo_mse_grid":  loo_mse_grid,
    }


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Pearson r + p, Spearman, bootstrap CI, gate classification."""
    if len(y_true) < 3:
        return {"label": label, "n": len(y_true), "r_pearson": float("nan")}
    r, p = pearsonr(y_pred, y_true)
    rs, _ = spearmanr(y_pred, y_true)
    ci_lo, ci_hi = _bootstrap_r_ci(y_pred, y_true)
    cls = classify_reliability(float(r), p=float(p), n=len(y_true), ci_lower=float(ci_lo))
    return {
        "label":      label,
        "n":          int(len(y_true)),
        "r_pearson":  float(r),
        "r_spearman": float(rs),
        "p_value":    float(p),
        "ci_lower":   float(ci_lo),
        "ci_upper":   float(ci_hi),
        "rmse":       float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "classification": cls,
    }


# ───────────────────────────────────────────────────────────────────────────
# Report writer
# ───────────────────────────────────────────────────────────────────────────

def write_markdown_report(
    out_path: Path,
    participant_id: str,
    n_paired: int,
    sbp_vanilla: dict,
    sbp_in_sample: dict,
    sbp_loo: dict,
    dbp_vanilla: dict,
    dbp_in_sample: dict,
    dbp_loo: dict,
    best_lambda_sbp: float,
    best_lambda_dbp: float,
) -> None:
    lines: list[str] = []
    lines.append(f"# Per-participant fine-tune — {participant_id}\n")
    lines.append(f"**n paired (cuff, PPG) samples:** {n_paired}\n")
    lines.append("Fine-tuning approach: freeze Phase 5; fit a ridge-regression linear head ")
    lines.append("from the (2 * lstm_cells)-d bidirectional-LSTM output to relative cuff BP, with ")
    lines.append("leave-one-out CV selecting alpha.  **LOO is the only honest column** — ")
    lines.append("in-sample shows what overfitting looks like, vanilla is the un-tuned baseline.\n")

    lines.append(f"\nRidge λ chosen by LOO: SBP = {best_lambda_sbp:.4g}, DBP = {best_lambda_dbp:.4g}\n")

    def _row(d: dict) -> str:
        return (
            f"| {d['label']} | {d['n']} | "
            f"{d.get('r_pearson', float('nan')):+.3f} | "
            f"{d.get('p_value', float('nan')):.3f} | "
            f"[{d.get('ci_lower', float('nan')):+.3f}, {d.get('ci_upper', float('nan')):+.3f}] | "
            f"{d.get('rmse', float('nan')):.2f} | "
            f"**{d.get('classification', '?')}** |"
        )

    lines.append("\n## SBP — within-person Pearson r against cuff\n")
    lines.append("| Configuration | n | r | p | 95% bootstrap CI | RMSE | Classification |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(_row(sbp_vanilla))
    lines.append(_row(sbp_in_sample))
    lines.append(_row(sbp_loo))

    lines.append("\n## DBP — within-person Pearson r against cuff\n")
    lines.append("| Configuration | n | r | p | 95% bootstrap CI | RMSE | Classification |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(_row(dbp_vanilla))
    lines.append(_row(dbp_in_sample))
    lines.append(_row(dbp_loo))

    lines.append("\n## Interpretation\n")
    delta_sbp = sbp_loo["r_pearson"] - sbp_vanilla["r_pearson"]
    delta_dbp = dbp_loo["r_pearson"] - dbp_vanilla["r_pearson"]
    lines.append(f"- **Δ SBP r (LOO − vanilla)** = {delta_sbp:+.3f}")
    lines.append(f"- **Δ DBP r (LOO − vanilla)** = {delta_dbp:+.3f}")
    in_minus_loo_sbp = sbp_in_sample["r_pearson"] - sbp_loo["r_pearson"]
    lines.append(
        f"- Overfit gap SBP (in-sample − LOO) = {in_minus_loo_sbp:+.3f}  "
        f"(large positive ⇒ fine-tune is memorising; near 0 ⇒ generalisable)"
    )
    if sbp_loo["classification"] == "reliable":
        lines.append("- **Verdict:** fine-tuning lifts SBP into the reliable gate band.")
    elif delta_sbp > 0.0:
        lines.append("- **Verdict:** fine-tuning improves SBP r but not enough to clear the gate.")
    else:
        lines.append("- **Verdict:** fine-tuning does not improve SBP r — model output ordering is the bottleneck.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ppg-dir", type=Path,
                    help="Dir of daily CentrePoint PPG CSVs")
    src.add_argument("--ppg", type=Path, nargs="+", help="Explicit list of PPG files")
    ap.add_argument("--checkpoint",   required=True, type=Path)
    ap.add_argument("--cuff",         required=True, type=Path)
    ap.add_argument("--participant-id", required=True)
    ap.add_argument("--out-dir",      required=True, type=Path)
    ap.add_argument("--ppg-fs",       type=int, default=100)
    ap.add_argument("--tz-offset",    type=float, default=-7.0)
    ap.add_argument("--max-lag-seconds", type=float, default=900.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ppg_dir is not None:
        ppg_files = sorted(args.ppg_dir.glob("*.csv"))
    else:
        ppg_files = sorted(args.ppg)
    if not ppg_files:
        raise SystemExit("No PPG files found.")
    logger.info("Processing %d daily file(s) from %s",
                len(ppg_files), args.ppg_dir or "list")

    # 1. Extract LSTM embeddings + vanilla Phase 5 predictions across all days
    embeddings, starts, predictions, _est = extract_phase5_embeddings(
        ppg_files=ppg_files,
        checkpoint=args.checkpoint,
        participant_id=args.participant_id,
        tz_offset=args.tz_offset,
        ppg_fs=args.ppg_fs,
    )

    # 2. Match cuff readings to windows
    pairs, idx_matched = match_cuff_to_windows(
        starts=starts,
        cuff_path=args.cuff,
        participant_id=args.participant_id,
        max_lag_seconds=args.max_lag_seconds,
    )
    n_paired = len(idx_matched)
    if n_paired < 4:
        raise SystemExit(f"Only {n_paired} matched pairs — need ≥4 for LOO CV.")

    # 3. Build supervised data: X = LSTM embedding at matched window, y = relative cuff BP
    X = embeddings[idx_matched]                     # (n, 64)
    cuff_sbp = np.array([
        p["cuff_sbp_mmhg"] for p in pairs if p["matched"]
    ], dtype=np.float64)
    cuff_dbp = np.array([
        p["cuff_dbp_mmhg"] for p in pairs if p["matched"]
    ], dtype=np.float64)
    rel_cuff_sbp = cuff_sbp - cuff_sbp.mean()
    rel_cuff_dbp = cuff_dbp - cuff_dbp.mean()

    # 4. Vanilla Phase 5 predictions at the matched windows (baseline)
    vanilla_pred_sbp = predictions[idx_matched, 0]
    vanilla_pred_dbp = predictions[idx_matched, 1]
    sbp_vanilla = metrics_from_predictions(
        rel_cuff_sbp, vanilla_pred_sbp, "Vanilla Phase 5",
    )
    dbp_vanilla = metrics_from_predictions(
        rel_cuff_dbp, vanilla_pred_dbp, "Vanilla Phase 5",
    )

    # 5. Ridge LOO for SBP and DBP
    logger.info("Fitting LOO ridge for SBP (n=%d, d=%d)...", *X.shape)
    sbp_fit = fit_ridge_loo(X, rel_cuff_sbp)
    logger.info("Fitting LOO ridge for DBP (n=%d, d=%d)...", *X.shape)
    dbp_fit = fit_ridge_loo(X, rel_cuff_dbp)

    sbp_in_sample = metrics_from_predictions(
        rel_cuff_sbp, sbp_fit["in_preds"], "Fine-tuned (in-sample)",
    )
    sbp_loo = metrics_from_predictions(
        rel_cuff_sbp, sbp_fit["loo_preds"], "Fine-tuned (LOO honest)",
    )
    dbp_in_sample = metrics_from_predictions(
        rel_cuff_dbp, dbp_fit["in_preds"], "Fine-tuned (in-sample)",
    )
    dbp_loo = metrics_from_predictions(
        rel_cuff_dbp, dbp_fit["loo_preds"], "Fine-tuned (LOO honest)",
    )

    # 6. Apply learned head to ALL embeddings → new prediction trace
    finetuned_sbp_all = sbp_fit["full_pipe"].predict(embeddings).astype(np.float32)
    finetuned_dbp_all = dbp_fit["full_pipe"].predict(embeddings).astype(np.float32)

    # Persist a finetuned predictions.csv aligned with the original schema
    ft_df = pd.DataFrame({
        "window_start_utc": [t.isoformat() for t in starts],
        "rel_sbp_phase5":   predictions[:, 0],
        "rel_dbp_phase5":   predictions[:, 1],
        "rel_sbp_finetuned": finetuned_sbp_all,
        "rel_dbp_finetuned": finetuned_dbp_all,
        # Absolute predictions: add the participant cuff mean as anchor
        "sbp_finetuned_mmhg": finetuned_sbp_all + cuff_sbp.mean(),
        "dbp_finetuned_mmhg": finetuned_dbp_all + cuff_dbp.mean(),
    })
    ft_csv = args.out_dir / "finetuned_predictions.csv"
    ft_df.to_csv(ft_csv, index=False)
    logger.info("Wrote %s", ft_csv)

    # 7. Diagnostic scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for axi, channel, y_true, vanilla, loo, fit, ttl in [
        (ax[0], "SBP", rel_cuff_sbp, vanilla_pred_sbp, sbp_fit["loo_preds"], sbp_loo, "SBP"),
        (ax[1], "DBP", rel_cuff_dbp, vanilla_pred_dbp, dbp_fit["loo_preds"], dbp_loo, "DBP"),
    ]:
        axi.scatter(y_true, vanilla, s=60, label=f"vanilla Phase 5 r={sbp_vanilla['r_pearson']:+.2f}"
                    if channel == "SBP" else f"vanilla Phase 5 r={dbp_vanilla['r_pearson']:+.2f}",
                    color="C0", alpha=0.6)
        axi.scatter(y_true, loo,     s=60, label=f"LOO fine-tune r={fit['r_pearson']:+.2f}",
                    color="C3", alpha=0.6, marker="^")
        lim = [min(y_true.min(), vanilla.min(), loo.min()) - 2,
               max(y_true.max(), vanilla.max(), loo.max()) + 2]
        axi.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        axi.set_xlabel(f"actual rel {channel} (mmHg)")
        axi.set_ylabel(f"predicted rel {channel} (mmHg)")
        axi.set_title(f"{ttl} — n={n_paired}")
        axi.legend(loc="lower right", fontsize=9)
        axi.grid(alpha=0.3)
    fig.suptitle(f"{args.participant_id} — vanilla vs LOO fine-tuned (n={n_paired})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out_dir / "finetune_diagnostics.png", dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", args.out_dir / "finetune_diagnostics.png")

    # 8. JSON + markdown
    summary = {
        "participant_id":      args.participant_id,
        "checkpoint":          str(args.checkpoint),
        "n_paired":            n_paired,
        "n_total_windows":     int(len(starts)),
        "ridge_lambdas":       RIDGE_LAMBDAS.tolist(),
        "best_lambda_sbp":     sbp_fit["best_lambda"],
        "best_lambda_dbp":     dbp_fit["best_lambda"],
        "loo_mse_grid_sbp":    sbp_fit["loo_mse_grid"].tolist(),
        "loo_mse_grid_dbp":    dbp_fit["loo_mse_grid"].tolist(),
        "sbp": {
            "vanilla":   sbp_vanilla,
            "in_sample": sbp_in_sample,
            "loo":       sbp_loo,
        },
        "dbp": {
            "vanilla":   dbp_vanilla,
            "in_sample": dbp_in_sample,
            "loo":       dbp_loo,
        },
        "gate_thresholds": {
            "r_min":      RELIABLE_R_THRESHOLD,
            "p_max":      P_VALUE_THRESHOLD,
            "n_min":      MIN_N_FOR_RELIABLE,
            "ci_lower_min": 0.0,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_ci_pct":    BOOTSTRAP_CI_PCT,
        },
        "pairs": pairs,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "finetune_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    write_markdown_report(
        out_path=args.out_dir / "finetune_comparison.md",
        participant_id=args.participant_id,
        n_paired=n_paired,
        sbp_vanilla=sbp_vanilla,
        sbp_in_sample=sbp_in_sample,
        sbp_loo=sbp_loo,
        dbp_vanilla=dbp_vanilla,
        dbp_in_sample=dbp_in_sample,
        dbp_loo=dbp_loo,
        best_lambda_sbp=sbp_fit["best_lambda"],
        best_lambda_dbp=dbp_fit["best_lambda"],
    )

    # 9. Console summary
    print()
    print("=" * 72)
    print(f"  FINE-TUNE — {args.participant_id}  (n={n_paired})")
    print("=" * 72)
    print(f"  SBP vanilla    : r={sbp_vanilla['r_pearson']:+.3f} p={sbp_vanilla['p_value']:.3f}  -> {sbp_vanilla['classification']}")
    print(f"  SBP in-sample  : r={sbp_in_sample['r_pearson']:+.3f} p={sbp_in_sample['p_value']:.3f}  -> {sbp_in_sample['classification']}")
    print(f"  SBP LOO honest : r={sbp_loo['r_pearson']:+.3f} p={sbp_loo['p_value']:.3f}  CI=[{sbp_loo['ci_lower']:+.3f}, {sbp_loo['ci_upper']:+.3f}]  -> {sbp_loo['classification']}")
    print(f"  Best lambda SBP     : {sbp_fit['best_lambda']:.4g}")
    print()
    print(f"  DBP vanilla    : r={dbp_vanilla['r_pearson']:+.3f} p={dbp_vanilla['p_value']:.3f}  -> {dbp_vanilla['classification']}")
    print(f"  DBP in-sample  : r={dbp_in_sample['r_pearson']:+.3f} p={dbp_in_sample['p_value']:.3f}  -> {dbp_in_sample['classification']}")
    print(f"  DBP LOO honest : r={dbp_loo['r_pearson']:+.3f} p={dbp_loo['p_value']:.3f}  CI=[{dbp_loo['ci_lower']:+.3f}, {dbp_loo['ci_upper']:+.3f}]  -> {dbp_loo['classification']}")
    print(f"  Best lambda DBP     : {dbp_fit['best_lambda']:.4g}")
    print()
    print(f"  Artifacts: {args.out_dir}")


if __name__ == "__main__":
    main()
