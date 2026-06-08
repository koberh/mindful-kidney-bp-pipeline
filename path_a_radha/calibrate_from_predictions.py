"""Per-participant calibration from an existing predictions.csv.

Unlike ``calibrate_participant.py`` (which re-runs the model on cuff-centred
windows of a single LEAP recording), this matches cuff readings against the
**already-computed** per-window predictions from a full inference run. That
guarantees the calibration uses the exact relative predictions shown in the
report, and it works across multiple UTC-split daily files (a PDT evening
reading lands in the next UTC day's predictions, matched purely by timestamp).

For each cuff reading we find the nearest prediction window within
``--max-lag-seconds`` and pair its relative SBP/DBP with the cuff value.
Then:
  * Pearson/Spearman r between predicted-relative and (cuff - mean cuff)
    → reliability tier (reliable / weak / unreliable), same thresholds as
      calibrate_participant.py.
  * Absolute-BP anchor offset = mean(cuff) - mean(predicted relative at the
    matched windows). Passing this as --calib-sbp/--calib-dbp to the report
    makes absolute predictions average to the measured cuff mean (the model's
    relative output is NOT zero-centred, so we cannot just add the cuff mean).

Usage:
  python -m path_a_radha.calibrate_from_predictions \\
      --predictions benchmark/results/centrepoint_SUBJ01/predictions.csv \\
      --cuff        benchmark/results/centrepoint_SUBJ01/cuff_readings.csv \\
      --participant-id SUBJ01 \\
      --out benchmark/results/centrepoint_SUBJ01/calibration.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.io import read_cuff_bp

RELIABLE_R_THRESHOLD: float = 0.40
WEAK_R_THRESHOLD: float = 0.0
WINDOW_SECONDS: float = 300.0

# Reliability-gate hardening: the original gate (r >= 0.40 only) produced a
# false "reliable" verdict on r=0.40, p=0.32, n=8 — the r swung +0.26 to +0.69
# under leave-one-out, indicating the result was not trustworthy. The new gate
# adds n / p / bootstrap-CI checks so a borderline r on too few readings cannot
# clear the bar.
MIN_N_FOR_RELIABLE: int = 5
P_VALUE_THRESHOLD: float = 0.05
BOOTSTRAP_RESAMPLES: int = 1000
BOOTSTRAP_CI_PCT: float = 95.0


def _bootstrap_r_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    ci_pct: float = BOOTSTRAP_CI_PCT,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI on Pearson r over paired (x, y) samples.

    Resamples (x_i, y_i) pairs with replacement, computes Pearson r each time,
    returns the (lower, upper) percentiles. Captures leave-one-out style
    instability that a single p-value misses (e.g., a borderline r=0.40 / p=0.32
    case where one outlier swings r between +0.26 and +0.69).
    """
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        # Degenerate resamples (all-same x or y) have undefined r; skip them.
        if np.std(xi) < 1e-12 or np.std(yi) < 1e-12:
            continue
        r = float(pearsonr(xi, yi)[0])
        if np.isfinite(r):
            rs.append(r)
    if len(rs) < n_boot // 5:
        # > 80% of bootstraps degenerate → not enough variation to trust the CI
        return float("nan"), float("nan")
    lo = float(np.percentile(rs, (100.0 - ci_pct) / 2.0))
    hi = float(np.percentile(rs, 100.0 - (100.0 - ci_pct) / 2.0))
    return lo, hi


def classify_reliability(
    r: float,
    p: float | None = None,
    n: int | None = None,
    ci_lower: float | None = None,
) -> str:
    """Tier a calibration as reliable / weak / unreliable / unknown.

    The "reliable" tier requires ALL of:
      - r ≥ 0.40 (clinically meaningful tracking)
      - n ≥ MIN_N_FOR_RELIABLE
      - p < 0.05 (parametric two-sided significance under H0: r=0)
      - bootstrap lower 95% CI > 0 (positivity stable under resampling)

    Any positive r that doesn't clear all four bars falls back to "weak".  A
    non-positive r is "unreliable" regardless of n.

    All extra-argument checks are gracefully skipped when the inputs are None
    or non-finite, so the function is still callable with just `r` for legacy
    use (e.g., a unit test that only wants to check the r ≥ 0.40 boundary).
    """
    if not np.isfinite(r):
        return "unknown"
    if r <= WEAK_R_THRESHOLD:
        return "unreliable"
    # r is positive; decide between "reliable" and "weak"
    if r < RELIABLE_R_THRESHOLD:
        return "weak"
    if n is not None and n < MIN_N_FOR_RELIABLE:
        return "weak"
    if p is not None and np.isfinite(p) and p >= P_VALUE_THRESHOLD:
        return "weak"
    if ci_lower is not None and np.isfinite(ci_lower) and ci_lower <= 0.0:
        return "weak"
    return "reliable"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--cuff", required=True, type=Path)
    p.add_argument("--participant-id", required=True)
    p.add_argument("--max-lag-seconds", type=float, default=900.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out_path = args.out or (args.predictions.parent / "calibration.json")

    # ── Load predictions; window CENTRE = start + half-window ──
    pred = pd.read_csv(args.predictions)
    win_start = pd.to_datetime(pred["window_start_utc"]).dt.tz_localize(None)
    win_center = win_start + pd.Timedelta(seconds=WINDOW_SECONDS / 2)
    win_center_s = win_center.astype("int64").to_numpy() / 1e9  # epoch seconds
    rel_sbp_all = pred["rel_sbp"].to_numpy(dtype=np.float64)
    rel_dbp_all = pred["rel_dbp"].to_numpy(dtype=np.float64)

    # ── Load cuff readings ──
    cuff = read_cuff_bp(args.cuff, participant_id=args.participant_id)
    if not cuff:
        raise SystemExit(f"No cuff readings for {args.participant_id!r} in {args.cuff}")

    pairs = []
    for c in sorted(cuff, key=lambda x: x.timestamp_utc):
        c_s = pd.Timestamp(c.timestamp_utc).value / 1e9
        d = np.abs(win_center_s - c_s)
        j = int(np.argmin(d))
        lag = float(win_center_s[j] - c_s)
        matched = bool(d[j] <= args.max_lag_seconds)
        pairs.append({
            "cuff_timestamp_utc": pd.Timestamp(c.timestamp_utc).isoformat(),
            "cuff_sbp_mmhg": float(c.sbp_mmhg),
            "cuff_dbp_mmhg": float(c.dbp_mmhg),
            "notes": c.notes,
            "matched": matched,
            "lag_seconds": lag if matched else None,
            "nearest_window_utc": pd.Timestamp(win_start.iloc[j]).isoformat(),
            "nearest_gap_seconds": float(d[j]),
            "predicted_rel_sbp": float(rel_sbp_all[j]) if matched else None,
            "predicted_rel_dbp": float(rel_dbp_all[j]) if matched else None,
        })

    used = [pr for pr in pairs if pr["matched"]]
    print(f"  Cuff readings: {len(pairs)}  *  matched to a PPG window "
          f"(+/-{args.max_lag_seconds:.0f}s): {len(used)}")
    for pr in pairs:
        tag = (f"lag {pr['lag_seconds']:+.0f}s" if pr["matched"]
               else f"NO MATCH (nearest {pr['nearest_gap_seconds']:.0f}s away)")
        print(f"    {pr['cuff_timestamp_utc']}  {pr['cuff_sbp_mmhg']:.0f}/"
              f"{pr['cuff_dbp_mmhg']:.0f}  -> {tag}")

    if len(used) < 3:
        print(f"  WARNING: only {len(used)} matched pairs — Pearson r unreliable (need >=3).")

    cuff_sbp = np.array([pr["cuff_sbp_mmhg"] for pr in used])
    cuff_dbp = np.array([pr["cuff_dbp_mmhg"] for pr in used])
    pred_sbp = np.array([pr["predicted_rel_sbp"] for pr in used])
    pred_dbp = np.array([pr["predicted_rel_dbp"] for pr in used])

    if len(used) >= 3:
        r_sbp_p, p_sbp = pearsonr(pred_sbp, cuff_sbp - cuff_sbp.mean())
        r_dbp_p, p_dbp = pearsonr(pred_dbp, cuff_dbp - cuff_dbp.mean())
        r_sbp_p, p_sbp = float(r_sbp_p), float(p_sbp)
        r_dbp_p, p_dbp = float(r_dbp_p), float(p_dbp)
        r_sbp_s = float(spearmanr(pred_sbp, cuff_sbp)[0])
        r_dbp_s = float(spearmanr(pred_dbp, cuff_dbp)[0])
        # Bootstrap 95% CI — captures leave-one-out instability that p-value misses
        ci_sbp_lo, ci_sbp_hi = _bootstrap_r_ci(pred_sbp, cuff_sbp - cuff_sbp.mean())
        ci_dbp_lo, ci_dbp_hi = _bootstrap_r_ci(pred_dbp, cuff_dbp - cuff_dbp.mean())
    else:
        r_sbp_p = r_dbp_p = r_sbp_s = r_dbp_s = float("nan")
        p_sbp = p_dbp = float("nan")
        ci_sbp_lo = ci_sbp_hi = ci_dbp_lo = ci_dbp_hi = float("nan")

    # Anchor offset so absolute = rel + offset averages to the cuff mean.
    anchor_sbp = float(cuff_sbp.mean() - pred_sbp.mean()) if len(used) else float("nan")
    anchor_dbp = float(cuff_dbp.mean() - pred_dbp.mean()) if len(used) else float("nan")

    cls_sbp = classify_reliability(r_sbp_p, p=p_sbp, n=len(used), ci_lower=ci_sbp_lo)
    cls_dbp = classify_reliability(r_dbp_p, p=p_dbp, n=len(used), ci_lower=ci_dbp_lo)
    cal = {
        "participant_id":        args.participant_id,
        "n_cuff_readings":       len(pairs),
        "n_paired_readings":     len(used),
        "predictions_file":      str(args.predictions),
        "cuff_baseline_sbp_mmhg": float(cuff_sbp.mean()) if len(used) else float("nan"),
        "cuff_baseline_dbp_mmhg": float(cuff_dbp.mean()) if len(used) else float("nan"),
        "calib_anchor_sbp_mmhg": anchor_sbp,
        "calib_anchor_dbp_mmhg": anchor_dbp,
        "calib_r_sbp_pearson":   r_sbp_p,
        "calib_r_sbp_spearman":  r_sbp_s,
        "calib_p_sbp":           p_sbp,
        "calib_ci_sbp_lower":    ci_sbp_lo,
        "calib_ci_sbp_upper":    ci_sbp_hi,
        "calib_r_dbp_pearson":   r_dbp_p,
        "calib_r_dbp_spearman":  r_dbp_s,
        "calib_p_dbp":           p_dbp,
        "calib_ci_dbp_lower":    ci_dbp_lo,
        "calib_ci_dbp_upper":    ci_dbp_hi,
        "gate_thresholds": {
            "r_min":      RELIABLE_R_THRESHOLD,
            "p_max":      P_VALUE_THRESHOLD,
            "n_min":      MIN_N_FOR_RELIABLE,
            "ci_lower_min": 0.0,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_ci_pct":    BOOTSTRAP_CI_PCT,
        },
        "classification_sbp":    cls_sbp,
        "classification_dbp":    cls_dbp,
        "recommendation_sbp": {
            "reliable":   ("Show BP trend in the biometric report. "
                           "All four checks passed (r ≥ 0.40, p < 0.05, "
                           "bootstrap CI > 0, n ≥ {n_min})."),
            "weak":       ("Show BP trend with caveats — positive r but at least "
                           "one stability check failed (insufficient n, p ≥ 0.05, "
                           "or bootstrap CI crosses 0)."),
            "unreliable": "Hide BP trend; replace with 'BP trend not available'.",
            "unknown":    "Not enough matched cuff+PPG samples (need ≥3).",
        }[cls_sbp].format(n_min=MIN_N_FOR_RELIABLE),
        "pairs": pairs,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w") as fh:
        json.dump(cal, fh, indent=2)

    print()
    print("=" * 60)
    print(f"  CALIBRATION — {args.participant_id}")
    print("=" * 60)
    print(f"  Matched pairs:    {len(used)}")
    if len(used):
        print(f"  Cuff baseline:    SBP {cuff_sbp.mean():.1f} / DBP {cuff_dbp.mean():.1f} mmHg")
        print(f"  SBP: r={r_sbp_p:+.3f}  p={p_sbp:.3f}  "
              f"95% CI=[{ci_sbp_lo:+.3f}, {ci_sbp_hi:+.3f}]  ->  {cls_sbp}")
        print(f"  DBP: r={r_dbp_p:+.3f}  p={p_dbp:.3f}  "
              f"95% CI=[{ci_dbp_lo:+.3f}, {ci_dbp_hi:+.3f}]  ->  {cls_dbp}")
        print(f"  Absolute anchor:  --calib-sbp {anchor_sbp:.1f}  --calib-dbp {anchor_dbp:.1f}")
        print(f"  Report gate:      --calib-r-sbp {r_sbp_p:.3f}")
        # Diagnostic call-out for the specific failure mode this gate was built for
        if (np.isfinite(r_sbp_p) and r_sbp_p >= RELIABLE_R_THRESHOLD
                and cls_sbp != "reliable"):
            failed = []
            if len(used) < MIN_N_FOR_RELIABLE:
                failed.append(f"n={len(used)} < {MIN_N_FOR_RELIABLE}")
            if np.isfinite(p_sbp) and p_sbp >= P_VALUE_THRESHOLD:
                failed.append(f"p={p_sbp:.3f} >= {P_VALUE_THRESHOLD}")
            if np.isfinite(ci_sbp_lo) and ci_sbp_lo <= 0.0:
                failed.append(f"bootstrap CI lower {ci_sbp_lo:+.3f} <= 0")
            print(f"  [!] r passed 0.40 but downgraded to 'weak': {', '.join(failed)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
