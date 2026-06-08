"""Per-participant calibration of the BP-trend model.

For each cuff reading from the baseline visit, extract the simultaneous
5-minute PPG window, run the Phase 3 LSTM, and compare predicted relative BP
to (cuff_value - participant_mean_cuff). Output a calibration.json with the
participant-level reliability classification used by biometric_report.py to
decide whether to show or hide the BP panels.

Reliability classes (based on Phase 3 eval finding — three failure modes
clustering around Pearson r):
  - r >= +0.40       → "reliable" (slow-trend tracker mode)        — show BP
  - 0.00 < r < 0.40  → "weak"     (predicts-near-mean mode)        — show w/ caveat
  - r <= 0.00        → "unreliable"(inverted or no signal)         — hide BP

Notes
-----
- Requires at least 3 paired cuff+PPG samples for Pearson r to be meaningful.
- Uses MC dropout = 0 for the calibration pass (deterministic point estimates)
  because we want a stable per-window prediction to compute correlation.

Usage:
  python -m path_a_radha.calibrate_participant \\
      --ppg "<participant-ppg.csv>" --accel "<participant-raw.csv>" \\
      --cuff "<cuff_readings.csv>" --participant-id MK01 \\
      --checkpoint benchmark/results/path_a_first_run/best_model.pt \\
      --out benchmark/results/calibration_MK01
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.io import (
    align_cuff_to_windows,
    load_leap_recording,
    read_cuff_bp,
)
from path_a_radha.estimator import RadhaEstimator


RELIABLE_R_THRESHOLD: float = 0.40
WEAK_R_THRESHOLD: float = 0.0


def classify_reliability(r: float) -> str:
    if not np.isfinite(r):
        return "unknown"
    if r >= RELIABLE_R_THRESHOLD:
        return "reliable"
    if r > WEAK_R_THRESHOLD:
        return "weak"
    return "unreliable"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppg",         required=True, type=Path)
    p.add_argument("--accel",       required=True, type=Path)
    p.add_argument("--cuff",        required=True, type=Path)
    p.add_argument("--checkpoint",  required=True, type=Path)
    p.add_argument("--participant-id", required=True)
    p.add_argument("--ppg-fs",      type=int, default=25)
    p.add_argument("--tz-offset",   type=float, default=-5.0)
    p.add_argument("--window-seconds",   type=float, default=300.0)
    p.add_argument("--max-lag-seconds",  type=float, default=900.0)
    p.add_argument("--out",         type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out if args.out is not None else (
        Path("benchmark/results") / f"calibration_{args.participant_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Loading cuff readings: {args.cuff}")
    cuff_readings = read_cuff_bp(args.cuff, participant_id=args.participant_id)
    if not cuff_readings:
        raise ValueError(
            f"No cuff readings found for participant {args.participant_id!r} in {args.cuff}"
        )
    print(f"  Loaded {len(cuff_readings)} cuff reading(s)")

    print(f"Loading LEAP recording")
    rec = load_leap_recording(
        ppg_path=args.ppg, accel_path=args.accel,
        participant_id=args.participant_id,
        tz_offset_hours=args.tz_offset, ppg_fs=args.ppg_fs,
    )
    print(f"  Duration: {len(rec.ppg_green) / rec.fs / 3600:.2f} hr")

    # ── Pair cuff readings with PPG windows ──
    aligned = align_cuff_to_windows(
        rec, cuff_readings,
        window_seconds=args.window_seconds,
        max_lag_seconds=args.max_lag_seconds,
    )
    print(f"  Pairs after alignment: {len(aligned)} (window {args.window_seconds:.0f}s, "
          f"max lag {args.max_lag_seconds:.0f}s)")
    if len(aligned) < 3:
        print(f"  WARNING: only {len(aligned)} paired samples — Pearson r will be unreliable. "
              "Recommend taking 3+ cuff readings during baseline visit.")

    # ── Run model on each paired window ──
    win_samp = int(args.window_seconds * rec.fs)
    n = len(aligned)
    # Crop or zero-pad each window to exactly win_samp (alignment can have ±1 sample)
    ppg_wins   = np.zeros((n, win_samp), dtype=np.float64)
    accel_wins = np.zeros((n, win_samp, 3), dtype=np.float64)
    for i, aw in enumerate(aligned):
        L = min(len(aw.ppg), win_samp)
        ppg_wins[i,   :L]    = aw.ppg[:L]
        accel_wins[i, :L, :] = aw.accel[:L, :]

    print(f"Loading model: {args.checkpoint}")
    est = RadhaEstimator(
        checkpoint_path=args.checkpoint,
        calibration_sbp_mean=0.0,  # we want relative output to compare with relative cuff
        calibration_dbp_mean=0.0,
        device="cpu",
        n_mc_samples=0,            # deterministic point estimates for correlation
    )
    est.cfg.fs = rec.fs
    # Position features: relative time from the first cuff reading
    base_time = aligned[0].cuff.timestamp_utc
    window_starts_min = np.array(
        [(aw.cuff.timestamp_utc - base_time).total_seconds() / 60.0 for aw in aligned],
        dtype=np.float32,
    )
    result = est.predict(
        ppg=ppg_wins, accel=accel_wins, fs=rec.fs,
        window_starts_min=window_starts_min,
    )
    pred_sbp = result["relative"][:, 0]
    pred_dbp = result["relative"][:, 1]

    # ── Cuff side: subtract per-participant mean ──
    cuff_sbp = np.array([aw.cuff.sbp_mmhg for aw in aligned], dtype=np.float64)
    cuff_dbp = np.array([aw.cuff.dbp_mmhg for aw in aligned], dtype=np.float64)
    rel_cuff_sbp = cuff_sbp - cuff_sbp.mean()
    rel_cuff_dbp = cuff_dbp - cuff_dbp.mean()

    # ── Pearson r ──
    if len(aligned) >= 3:
        r_sbp_p, _ = pearsonr(pred_sbp, rel_cuff_sbp)
        r_dbp_p, _ = pearsonr(pred_dbp, rel_cuff_dbp)
        r_sbp_s, _ = spearmanr(pred_sbp, rel_cuff_sbp)
        r_dbp_s, _ = spearmanr(pred_dbp, rel_cuff_dbp)
    else:
        r_sbp_p = r_dbp_p = r_sbp_s = r_dbp_s = float("nan")

    classification_sbp = classify_reliability(r_sbp_p)
    classification_dbp = classify_reliability(r_dbp_p)

    cal = {
        "participant_id":          args.participant_id,
        "n_paired_readings":       len(aligned),
        "checkpoint":              str(args.checkpoint),
        "cuff_baseline_sbp_mmhg":  float(cuff_sbp.mean()),
        "cuff_baseline_dbp_mmhg":  float(cuff_dbp.mean()),
        "calib_r_sbp_pearson":     float(r_sbp_p),
        "calib_r_sbp_spearman":    float(r_sbp_s),
        "calib_r_dbp_pearson":     float(r_dbp_p),
        "calib_r_dbp_spearman":    float(r_dbp_s),
        "classification_sbp":      classification_sbp,
        "classification_dbp":      classification_dbp,
        "recommendation_sbp":      {
            "reliable":   "Show BP trend in the biometric report.",
            "weak":       "Show BP trend with the caveat that signal is noisy.",
            "unreliable": "Hide BP trend; replace with 'BP trend not available for this participant'.",
            "unknown":    "Not enough paired cuff+PPG samples (need ≥3).",
        }[classification_sbp],
        "pairs": [
            {
                "cuff_timestamp_utc": aw.cuff.timestamp_utc.isoformat(),
                "cuff_sbp_mmhg":   float(aw.cuff.sbp_mmhg),
                "cuff_dbp_mmhg":   float(aw.cuff.dbp_mmhg),
                "lag_seconds":     float(aw.lag_seconds),
                "predicted_rel_sbp": float(pred_sbp[i]),
                "predicted_rel_dbp": float(pred_dbp[i]),
            }
            for i, aw in enumerate(aligned)
        ],
        "generated_utc":           datetime.now(timezone.utc).isoformat(),
    }
    out_path = out_dir / "calibration.json"
    with open(out_path, "w") as fh:
        json.dump(cal, fh, indent=2)

    # ── Console summary ──
    print()
    print("=" * 60)
    print(f"  PER-PARTICIPANT CALIBRATION — {args.participant_id}")
    print("=" * 60)
    print(f"  N paired readings:  {len(aligned)}")
    print(f"  Cuff baseline:      SBP {cuff_sbp.mean():.1f}  /  DBP {cuff_dbp.mean():.1f} mmHg")
    print(f"  SBP Pearson r:      {r_sbp_p:+.3f}  ({classification_sbp})")
    print(f"  DBP Pearson r:      {r_dbp_p:+.3f}  ({classify_reliability(r_dbp_p)})")
    print(f"  Recommendation:     {cal['recommendation_sbp']}")
    print()
    print(f"  Pass to biometric_report.py: --calib-r-sbp {r_sbp_p:.3f}")
    print(f"                                --calib-sbp {cuff_sbp.mean():.1f}")
    print(f"                                --calib-dbp {cuff_dbp.mean():.1f}")
    print()
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
