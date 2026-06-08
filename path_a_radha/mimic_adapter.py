"""MIMIC-III → Path A 174-feature adapter.

Reads raw MIMIC .npz records (pleth + abp @ 125 Hz) and produces the
per-record .npz format that path_a_radha/train.py consumes:

    features  : (T, 176) float32   — 174 PPG + 2 position features per 5-min window
    targets   : (T, 3)   float32   — [rel_SBP, rel_DBP, rel_MAP] mmHg
    sqi       : (T,)     float32   — skewness-based SQI per window
    window_start_min : (T,) float32 — minutes since record start (for diagnostics
                                       and to support sub-sequence chunking
                                       downstream if desired)
    participant_id: str            — subject_id, e.g. "p002636"
    record    : str                — full record ID, e.g. "p002636-2109-11-02-17-57"
    source_path: str

Pipeline (per record):
  1. Linearly interpolate NaN gaps in PLETH and ABP (record original NaN
     mask for the per-window QC gate; sosfiltfilt propagates NaN otherwise)
  2. Bandpass filter PLETH 0.5–4 Hz (Yao 2022 §C.1)
  3. Detect ABP systolic peaks with foot-based DBP (see detect_abp_beats);
     filter beats to physiological SBP/DBP/PP ranges
  4. Compute record-median SBP/DBP/MAP for relative targets
  5. Slice signal into 5-min windows, 5-min step (non-overlapping)
  6. Per window:
       - Reject if >5% of pre-interpolation raw samples were NaN (PLETH or ABP)
       - Reject if >5% of ABP samples are at clip bounds (transducer artifact)
       - Reject if fewer than MIN_ABP_BEATS_PER_WINDOW good ABP beats
       - Per-beat outlier rejection: drop beats whose SBP differs from
         window-median SBP by >20 mmHg (catches ectopic / artifact transients)
       - Compute 174 PPG features (HRV + Elgendi + Gaussian + Monte-Moreno);
         SQI from PLETH skewness; relative SBP/DBP/MAP from beat medians
       - Reject if all features NaN, or if max |feature| > 1e4
  7. Output features with NaN preserved — cohort-level imputation happens
     in train.py (so the same medians are used across all records)

Activity features (21) are NOT computed — MIMIC has no accelerometer.
See OPEN_QUESTIONS.md Q9 for the rationale (LSTM input dimension is 174).

Sample rate: MIMIC native 125 Hz (no resampling); see OPEN_QUESTIONS.md Q4.

Usage:
    python -m path_a_radha.mimic_adapter \\
        --allowlist  ckd_allowlist.csv \\
        --output-dir data/processed/path_a

    # Or single-record smoke test:
    python -m path_a_radha.mimic_adapter \\
        --single-record p002636-2109-11-02-17-57 \\
        --allowlist  ... --output-dir ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from common.preprocessing import (
    SQIMethod,
    bandpass_filter,
    compute_sqi,
)
from path_a_radha.features import (
    extract_elgendi_features,
    extract_gaussian_features,
    extract_hrv_features,
    extract_monte_moreno_features,
)


# ── Config ────────────────────────────────────────────────────────────────────

FS_MIMIC: int = 125             # MIMIC-III waveform sample rate
WINDOW_SECONDS: float = 300.0   # 5-min windows (Radha §3.2.5)
WINDOW_STEP_SECONDS: float = 300.0  # 5-min step (non-overlapping)
BP_LOW_HZ: float = 0.5          # Yao 2022 §C.1
BP_HIGH_HZ: float = 4.0
BP_FILTER_ORDER: int = 4

# ABP beat detection
ABP_PEAK_DIST_S: float = 0.3    # min 0.3 s between systolic peaks (~200 bpm max)
ABP_PEAK_PROM_MMHG: float = 8.0 # mmHg prominence
ABP_CLIP_LOW: float = 0.0       # ABP < 0 = transducer artifact; clip before beat detection
ABP_CLIP_HIGH: float = 250.0    # ABP > 250 = saturation; clip before beat detection

# Physiological beat-level ranges
SBP_RANGE: tuple[float, float] = (60.0, 260.0)
DBP_RANGE: tuple[float, float] = (20.0, 150.0)
PP_RANGE: tuple[float, float] = (20.0, 150.0)   # pulse pressure (SBP-DBP); ICU literature standard

# Window-level ABP quality gates
SATURATION_FRAC_MAX: float = 0.05   # reject window if >5% of ABP samples are at clip bounds
MIN_ABP_BEATS_PER_WINDOW: int = 10  # need >=10 good beats per 5-min window
# Window-level NaN-coverage gate (some MIMIC records have NaN gaps in PLETH/ABP)
WINDOW_NAN_FRAC_MAX: float = 0.05   # reject window if >5% of either raw input was NaN

N_PPG_FEATURES: int = 174    # 5 HRV + 40 Elgendi + 60 Gaussian + 69 Monte-Moreno
N_POSITION_FEATURES: int = 2 # time_since_start_min + gap_to_prev_valid_min
N_FEATURES_LSTM: int = N_PPG_FEATURES + N_POSITION_FEATURES  # 176 total


# ── ABP beat extraction ──────────────────────────────────────────────────────

def detect_abp_beats(abp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (peak_idx, sbp_mmhg, dbp_mmhg) from a continuous ABP waveform.

    SBP = ABP value at each systolic peak (skipping the first peak, since
    we cannot identify its preceding diastolic foot without a prior peak).
    DBP = "foot" detection — the minimum ABP in the late portion of the
    interval immediately before the systolic peak, restricted to:
        max(prev_peak + 0.5*RR, peak - 0.4 s) ... peak

    Restricting to the late half rejects the dicrotic notch (which occurs
    early after the systolic peak — in the first half of the cardiac cycle).
    Restricting to within 0.4 s before the peak rejects artifact spikes
    in long RR intervals (e.g. arrhythmic pauses).

    Returns empty arrays if fewer than 2 peaks are found.
    """
    clean = abp.copy()
    med = float(np.nanmedian(clean))
    clean[np.isnan(clean)] = med if np.isfinite(med) else 80.0
    clean = np.clip(clean, 0.0, 300.0)

    peaks, _ = find_peaks(
        clean,
        distance=int(ABP_PEAK_DIST_S * FS_MIMIC),
        prominence=ABP_PEAK_PROM_MMHG,
    )
    if len(peaks) < 2:
        return np.array([], dtype=np.intp), np.array([]), np.array([])

    # For each peak after the first, find the foot (DBP) just before it.
    max_lookback_samples = int(0.4 * FS_MIMIC)  # 0.4 s window before peak
    out_peaks: list[int] = []
    out_sbp: list[float] = []
    out_dbp: list[float] = []
    for i in range(1, len(peaks)):
        prev_pk = int(peaks[i - 1])
        pk = int(peaks[i])
        rr = pk - prev_pk
        if rr <= 2:
            continue  # too tight to find a foot
        # Late half of the RR interval, capped at 0.4 s before the peak
        late_half_start = prev_pk + rr // 2
        bounded_start = max(late_half_start, pk - max_lookback_samples)
        start = min(bounded_start, pk - 2)
        end = pk
        if start >= end:
            continue
        seg = abp[start:end]
        if not np.any(np.isfinite(seg)):
            continue
        dbp_val = float(np.nanmin(seg))
        sbp_val = float(abp[pk])
        out_peaks.append(pk)
        out_sbp.append(sbp_val)
        out_dbp.append(dbp_val)

    if not out_peaks:
        return np.array([], dtype=np.intp), np.array([]), np.array([])
    return (
        np.array(out_peaks, dtype=np.intp),
        np.array(out_sbp, dtype=np.float64),
        np.array(out_dbp, dtype=np.float64),
    )


def filter_beats_physiological(
    peaks: np.ndarray,
    sbp: np.ndarray,
    dbp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only beats with SBP/DBP in physiological range, SBP > DBP, and
    pulse pressure in [20, 120] mmHg (rejects transducer-zero artifact beats
    that produce abnormally large pulse pressures).

    Returns (peaks, sbp, dbp, map_bp).
    """
    pp = sbp - dbp
    valid = (
        (sbp >= SBP_RANGE[0]) & (sbp <= SBP_RANGE[1])
        & (dbp >= DBP_RANGE[0]) & (dbp <= DBP_RANGE[1])
        & (pp >= PP_RANGE[0]) & (pp <= PP_RANGE[1])
    )
    peaks = peaks[valid]
    sbp = sbp[valid]
    dbp = dbp[valid]
    map_bp = dbp + (sbp - dbp) / 3.0
    return peaks, sbp, dbp, map_bp


def window_abp_saturation_frac(abp_window: np.ndarray) -> float:
    """Fraction of samples that are at or near the clip bounds (saturation flag)."""
    if len(abp_window) == 0:
        return 1.0
    finite = abp_window[np.isfinite(abp_window)]
    if len(finite) == 0:
        return 1.0
    # Flag samples within 0.5 mmHg of either clip bound
    near_low = float(np.mean(finite <= ABP_CLIP_LOW + 0.5))
    near_high = float(np.mean(finite >= ABP_CLIP_HIGH - 0.5))
    return near_low + near_high


# ── Per-record adapter ───────────────────────────────────────────────────────

@dataclass
class AdapterStats:
    record: str
    subject_id: str
    n_windows_total: int = 0
    n_windows_kept: int = 0
    skip_reasons: dict[str, int] | None = None
    record_sbp_median: float = float("nan")
    record_dbp_median: float = float("nan")
    elapsed_s: float = 0.0


def process_record(
    raw_npz_path: Path,
    out_npz_path: Path,
    verbose: bool = False,
) -> AdapterStats:
    """Process one raw MIMIC .npz → one Path A .npz.

    Returns AdapterStats. Writes the .npz to out_npz_path even if zero windows
    are kept (caller checks n_windows_kept before considering it usable).
    """
    t0 = time.time()
    data = np.load(raw_npz_path, allow_pickle=True)
    pleth = data["pleth"].astype(np.float32)
    abp = data["abp"].astype(np.float32)
    fs = int(data["fs"])
    record = str(data["record"])
    subject_id = record.split("-")[0]  # e.g. "p002636"

    if fs != FS_MIMIC:
        raise ValueError(f"Expected fs={FS_MIMIC}, got {fs} in {raw_npz_path}")

    n_samples = min(len(pleth), len(abp))
    pleth = pleth[:n_samples]
    abp = abp[:n_samples]

    # ── Interpolate NaN gaps before filtering ────────────────────────────────
    # MIMIC records often have short NaN runs (sensor dropouts, telemetry gaps).
    # sosfiltfilt propagates NaN through the entire output, so we linearly
    # interpolate through gaps here and use the original NaN mask later to
    # reject windows that had too much missing data.
    pleth_nan_mask = np.isnan(pleth)
    abp_nan_mask = np.isnan(abp)
    if pleth_nan_mask.any():
        valid = ~pleth_nan_mask
        if valid.sum() < 2:
            pleth_filled = np.zeros_like(pleth, dtype=np.float32)
        else:
            valid_idx = np.where(valid)[0]
            pleth_filled = np.interp(
                np.arange(len(pleth)), valid_idx, pleth[valid_idx]
            ).astype(np.float32)
    else:
        pleth_filled = pleth
    if abp_nan_mask.any():
        valid = ~abp_nan_mask
        if valid.sum() < 2:
            abp_filled = np.full_like(abp, 80.0, dtype=np.float32)
        else:
            valid_idx = np.where(valid)[0]
            abp_filled = np.interp(
                np.arange(len(abp)), valid_idx, abp[valid_idx]
            ).astype(np.float32)
    else:
        abp_filled = abp
    pleth = pleth_filled
    abp = abp_filled

    # ── Bandpass PLETH ───────────────────────────────────────────────────────
    try:
        pleth_filt = bandpass_filter(
            pleth, fs=fs, low=BP_LOW_HZ, high=BP_HIGH_HZ, order=BP_FILTER_ORDER,
        ).astype(np.float32)
    except ValueError:
        # Signal too short — abort with zero windows
        np.savez(
            out_npz_path,
            features=np.empty((0, N_FEATURES_LSTM), dtype=np.float32),
            targets=np.empty((0, 3), dtype=np.float32),
            sqi=np.empty((0,), dtype=np.float32),
            participant_id=subject_id,
            record=record,
            source_path=str(raw_npz_path),
        )
        return AdapterStats(record=record, subject_id=subject_id,
                            skip_reasons={"signal_too_short_for_bandpass": 1},
                            elapsed_s=time.time() - t0)

    # ── ABP beats + per-record baseline ──────────────────────────────────────
    peak_idx, sbp_all, dbp_all = detect_abp_beats(abp)
    peak_idx, sbp_all, dbp_all, map_all = filter_beats_physiological(
        peak_idx, sbp_all, dbp_all
    )
    if len(peak_idx) < 20:
        np.savez(
            out_npz_path,
            features=np.empty((0, N_FEATURES_LSTM), dtype=np.float32),
            targets=np.empty((0, 3), dtype=np.float32),
            sqi=np.empty((0,), dtype=np.float32),
            participant_id=subject_id,
            record=record,
            source_path=str(raw_npz_path),
        )
        return AdapterStats(record=record, subject_id=subject_id,
                            skip_reasons={"too_few_valid_abp_beats": 1},
                            elapsed_s=time.time() - t0)

    sbp_base = float(np.median(sbp_all))
    dbp_base = float(np.median(dbp_all))
    map_base = float(np.median(map_all))

    # ── Windowing ────────────────────────────────────────────────────────────
    win_samp = int(WINDOW_SECONDS * fs)        # 37 500 at 125 Hz
    step_samp = int(WINDOW_STEP_SECONDS * fs)  # 37 500 (non-overlapping)
    n_windows = max(0, (n_samples - win_samp) // step_samp + 1)

    feat_list: list[np.ndarray] = []
    targ_list: list[np.ndarray] = []
    sqi_list: list[float] = []
    window_start_min_list: list[float] = []   # for position features
    last_kept_window_start_min: float | None = None
    skip = {
        "too_few_beats": 0, "abp_saturation": 0,
        "all_nan_features": 0, "extreme_feature_value": 0,
        "window_too_much_raw_nan": 0,
    }
    nan_imputed_per_window: list[int] = []  # count of NaN features imputed per kept window

    for w in range(n_windows):
        s = w * step_samp
        e = s + win_samp
        ppg_win = pleth_filt[s:e]
        abp_win = abp[s:e]

        # Reject windows where too much of the raw input was NaN before
        # interpolation (interpolated values are guesses; trust gates here)
        pleth_nan_frac = float(pleth_nan_mask[s:e].mean())
        abp_nan_frac = float(abp_nan_mask[s:e].mean())
        if pleth_nan_frac > WINDOW_NAN_FRAC_MAX or abp_nan_frac > WINDOW_NAN_FRAC_MAX:
            skip["window_too_much_raw_nan"] += 1
            continue

        # Reject windows where the raw ABP is saturated/clipped above threshold
        sat = window_abp_saturation_frac(abp_win)
        if sat > SATURATION_FRAC_MAX:
            skip["abp_saturation"] += 1
            continue

        # ABP beats (post physiological filter) falling in this window
        mask_abp = (peak_idx >= s) & (peak_idx < e)
        if int(mask_abp.sum()) < MIN_ABP_BEATS_PER_WINDOW:
            skip["too_few_beats"] += 1
            continue

        # Tier 1: per-beat outlier rejection inside the window. Reject beats
        # whose SBP differs from the window-median SBP by >20 mmHg (catches
        # ectopic beats / artifact transients that the per-beat physiological
        # filter let through). Re-check the beat count after outlier rejection.
        win_sbp_beats = sbp_all[mask_abp]
        win_dbp_beats = dbp_all[mask_abp]
        win_map_beats = map_all[mask_abp]
        median_sbp = float(np.median(win_sbp_beats))
        outlier_mask = np.abs(win_sbp_beats - median_sbp) <= 20.0
        if int(outlier_mask.sum()) < MIN_ABP_BEATS_PER_WINDOW:
            skip["too_few_beats"] += 1
            continue
        win_sbp_beats = win_sbp_beats[outlier_mask]
        win_dbp_beats = win_dbp_beats[outlier_mask]
        win_map_beats = win_map_beats[outlier_mask]

        win_sbp = float(np.median(win_sbp_beats))
        win_dbp = float(np.median(win_dbp_beats))
        win_map = float(np.median(win_map_beats))

        # 174 PPG features
        hrv = extract_hrv_features(ppg_win.astype(np.float64), fs)
        elg = extract_elgendi_features(ppg_win.astype(np.float64), fs)
        gau = extract_gaussian_features(ppg_win.astype(np.float64), fs)
        mm = extract_monte_moreno_features(ppg_win.astype(np.float64), fs)
        ppg_feats = np.concatenate([hrv, elg, gau, mm]).astype(np.float32)
        assert ppg_feats.shape[0] == N_PPG_FEATURES, (
            f"PPG feature dim mismatch: got {ppg_feats.shape[0]} expected {N_PPG_FEATURES}"
        )
        # 2 window-position features (Tier 1.5):
        #   time_since_record_start_min: monotonic across the record
        #   gap_to_prev_valid_window_min: 0.0 for first kept window; otherwise the
        #     elapsed minutes since the previous KEPT window's start. Tells the
        #     LSTM whether adjacent samples in the sequence are time-adjacent or
        #     separated by rejected windows.
        window_start_min = float(s) / float(fs) / 60.0
        if last_kept_window_start_min is None:
            gap_to_prev_min = 0.0
        else:
            gap_to_prev_min = window_start_min - last_kept_window_start_min
        position_feats = np.array(
            [window_start_min, gap_to_prev_min], dtype=np.float32
        )
        feats = np.concatenate([ppg_feats, position_feats]).astype(np.float32)
        assert feats.shape[0] == N_FEATURES_LSTM, (
            f"Total feature dim mismatch: got {feats.shape[0]} expected {N_FEATURES_LSTM}"
        )

        # Reject windows where ALL features are NaN (catastrophic extraction failure)
        if not np.any(np.isfinite(feats)):
            skip["all_nan_features"] += 1
            continue
        # Reject windows with pathologically large feature values (numerical
        # blow-up; train-time z-score step would still normalize but the
        # gradient instability during early epochs hurts convergence)
        finite_max = float(np.nanmax(np.abs(feats)))
        if finite_max > 1e4:
            skip["extreme_feature_value"] += 1
            continue

        # SQI on filtered PPG window (skewness method, fast)
        sqi_val = float(compute_sqi(ppg_win.astype(np.float64), SQIMethod.SKEWNESS, fs=fs))

        # Relative targets
        targ = np.array(
            [win_sbp - sbp_base, win_dbp - dbp_base, win_map - map_base],
            dtype=np.float32,
        )

        nan_imputed_per_window.append(int(np.isnan(feats).sum()))
        feat_list.append(feats)
        targ_list.append(targ)
        sqi_list.append(sqi_val)
        window_start_min_list.append(window_start_min)
        last_kept_window_start_min = window_start_min

    if not feat_list:
        np.savez(
            out_npz_path,
            features=np.empty((0, N_FEATURES_LSTM), dtype=np.float32),
            targets=np.empty((0, 3), dtype=np.float32),
            sqi=np.empty((0,), dtype=np.float32),
            participant_id=subject_id,
            record=record,
            source_path=str(raw_npz_path),
        )
        return AdapterStats(
            record=record, subject_id=subject_id,
            n_windows_total=n_windows,
            n_windows_kept=0, skip_reasons=skip,
            record_sbp_median=sbp_base, record_dbp_median=dbp_base,
            elapsed_s=time.time() - t0,
        )

    features = np.stack(feat_list).astype(np.float32)      # (T, 174)
    targets = np.stack(targ_list).astype(np.float32)        # (T, 3)
    sqi = np.array(sqi_list, dtype=np.float32)              # (T,)

    # Tier 1: preserve NaN in adapter output — cohort-level imputation
    # happens in train.py (so the same median is used across all records).
    # NaN counts are still logged for diagnostics.
    total_nan_in_output = int(np.isnan(features).sum())
    nan_frac_per_feature = np.isnan(features).mean(axis=0)
    n_always_nan_this_record = int((nan_frac_per_feature == 1.0).sum())

    window_start_min_arr = np.array(window_start_min_list, dtype=np.float32)

    np.savez(
        out_npz_path,
        features=features,
        targets=targets,
        sqi=sqi,
        window_start_min=window_start_min_arr,
        participant_id=subject_id,
        record=record,
        source_path=str(raw_npz_path),
    )

    if verbose:
        print(
            f"  {record}: {len(feat_list)}/{n_windows} windows kept "
            f"(SBP_med={sbp_base:.1f}, DBP_med={dbp_base:.1f} mmHg, "
            f"NaN cells in output: {total_nan_in_output}, "
            f"always-NaN features for this record: {n_always_nan_this_record})"
        )

    stats = AdapterStats(
        record=record, subject_id=subject_id,
        n_windows_total=n_windows,
        n_windows_kept=len(feat_list),
        skip_reasons=skip,
        record_sbp_median=sbp_base, record_dbp_median=dbp_base,
        elapsed_s=time.time() - t0,
    )
    stats.skip_reasons = {
        **(stats.skip_reasons or {}),
        "_n_always_nan_features_this_record": n_always_nan_this_record,
        "_n_nan_cells_in_output": total_nan_in_output,
    }
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_allowlist(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    p = argparse.ArgumentParser(description="MIMIC-III → Path A 174-feature adapter")
    p.add_argument("--allowlist", required=True, type=Path,
                   help="CKD allowlist CSV with columns subject_id, record_id, source_path")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write per-record .npz files")
    p.add_argument("--single-record", default=None,
                   help="Run on just one record_id (smoke test)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_allowlist(args.allowlist)

    if args.single_record:
        rows = [r for r in rows if r["record_id"] == args.single_record]
        if not rows:
            print(f"No matching record_id={args.single_record!r} in allowlist.")
            sys.exit(1)

    print(f"Processing {len(rows)} record(s) → {args.output_dir}")
    all_stats: list[AdapterStats] = []
    for i, row in enumerate(rows, 1):
        rec = row["record_id"]
        src = Path(row["source_path"])
        sid = row["subject_id"]
        out = args.output_dir / f"p{int(sid):06d}_{rec}.npz"

        print(f"[{i}/{len(rows)}] {src.name} → {out.name}", flush=True)
        try:
            stats = process_record(src, out, verbose=args.verbose)
            all_stats.append(stats)
            print(
                f"  kept {stats.n_windows_kept}/{stats.n_windows_total} windows "
                f"(SBP_med={stats.record_sbp_median:.1f}, "
                f"elapsed {stats.elapsed_s:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001 — log and continue across records
            print(f"  ERROR: {exc}")

    # Summary
    total_kept = sum(s.n_windows_kept for s in all_stats)
    n_ok = sum(1 for s in all_stats if s.n_windows_kept > 0)
    print(f"\nDone. {n_ok}/{len(rows)} records produced ≥1 window; total windows: {total_kept}")

    log_path = args.output_dir / "_adapter_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "record": s.record, "subject_id": s.subject_id,
                    "n_windows_total": s.n_windows_total,
                    "n_windows_kept": s.n_windows_kept,
                    "skip_reasons": s.skip_reasons,
                    "record_sbp_median": s.record_sbp_median,
                    "record_dbp_median": s.record_dbp_median,
                    "elapsed_s": s.elapsed_s,
                }
                for s in all_stats
            ],
            fh, indent=2, default=str,
        )
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
