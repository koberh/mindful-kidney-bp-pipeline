"""Aurora-BP oscillometric arm → Path A 176-feature adapter.

Reads the Aurora-BP measurements_oscillometric.tsv + measurements_oscillometric.zip
and produces per-participant-day .npz files for Phase 5 training.

Output .npz schema (matches the MIMIC adapter / build_day_npz convention):
    features         : (T, 176) float32  — 174 PPG + 2 position features
    targets          : (T, 3)   float32  — absolute [SBP, DBP, MAP] mmHg
                                           (relativized inside ParticipantDayDataset)
    sqi              : (T,)     float32  — optical_quality from the measurements TSV
    window_start_min : (T,)     float32  — minutes since first measurement of day
    participant_id   : str
    date             : str  (YYYY-MM-DD)

Waveform format inside the zip:
    TSV files with columns: t, ekg, optical, pressure, accel_x, accel_y, accel_z
    Sampled at 500 Hz (resampled to common timebase by Aurora-BP pipeline).
    We downsample optical 500 → 100 Hz to match the LEAP deployment rate.

Short-window note:
    Ambulatory snippets are ~15 s (375 samples at 25 Hz, ~18 pulses at 72 bpm).
    Activity features [0:21] are all-NaN (no wrist accel from aurora cuff arm —
    only tonometry accel is present, not wrist accel). HRV features [21:26]
    will mostly be NaN (need 25+ peaks). Both groups are handled gracefully:
    activity features are sliced off before the LSTM; NaN features are imputed
    at training time via cohort-level medians in train.py's Tier 1 pipeline.
    All 169 morphology features (Elgendi / Gaussian / Monte-Moreno) work fine.

Usage:
    python -m path_a_radha.aurora_bp_adapter \\
        --tsv        data/aurora_bp/measurements_oscillometric.tsv \\
        --zip        data/aurora_bp/measurements_oscillometric.zip \\
        --output-dir data/processed/path_a_aurora \\
        [--min-quality 0.65] \\
        [--min-readings-per-day 3] \\
        [--single-subject o000]
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

from common.preprocessing import (
    NormMethod,
    SQIMethod,
    bandpass_filter,
    compute_sqi,
    normalize_window,
)
from path_a_radha.features import (
    extract_elgendi_features,
    extract_gaussian_features,
    extract_hrv_features,
    extract_monte_moreno_features,
)

logger = logging.getLogger(__name__)

FS_AURORA: int = 500        # native sample rate inside the zip waveform files
FS_OUT: int = 100           # downsample to match LEAP 100 Hz deployment rate
DECIMATE_FACTOR: int = FS_AURORA // FS_OUT   # 5
MIN_SAMPLES: int = 200      # skip snippets shorter than this after downsampling (~2s at 100 Hz)


def _downsample(sig: np.ndarray, factor: int = DECIMATE_FACTOR) -> np.ndarray:
    """Downsample by integer factor using scipy.signal.decimate (anti-alias FIR)."""
    if factor == 1:
        return sig.astype(np.float64)
    return decimate(sig.astype(np.float64), factor, ftype="fir", zero_phase=True)


def _extract_ppg_features(ppg_500hz: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downsample, filter, and extract 174 PPG features from a 500 Hz snippet.

    Returns:
        feats : (174,) float64 — NaN where extraction failed
        sqi   : float — template-based SQI of the filtered signal
    """
    ppg_100 = _downsample(ppg_500hz)
    if len(ppg_100) < MIN_SAMPLES:
        return np.full(174, np.nan, dtype=np.float64), 0.0

    try:
        ppg_filt = bandpass_filter(ppg_100, FS_OUT, 0.5, 4.0)
    except ValueError:
        ppg_filt = ppg_100.copy()

    sqi = float(compute_sqi(ppg_filt, SQIMethod.TEMPLATE, fs=FS_OUT))
    ppg_norm = normalize_window(ppg_filt, NormMethod.ZSCORE)

    # Dummy accelerometer: constant 1 g on +Z (activity features → NaN / 0, sliced off
    # before LSTM — same pattern as centrepoint.py with no wrist accel export).
    dummy_accel = np.zeros((len(ppg_norm), 3), dtype=np.float64)
    dummy_accel[:, 2] = 1.0

    # Extract only the 174 PPG sub-families (skip activity [0:21], extract 21:195).
    hrv = extract_hrv_features(ppg_norm, FS_OUT)            # (5,)
    elg = extract_elgendi_features(ppg_norm, FS_OUT)        # (40,)
    gau = extract_gaussian_features(ppg_norm, FS_OUT)       # (60,)
    mm  = extract_monte_moreno_features(ppg_norm, FS_OUT)   # (69,)
    feats = np.concatenate([hrv, elg, gau, mm]).astype(np.float64)   # (174,)
    return feats, sqi


def process_subject_day(
    day_rows: pd.DataFrame,
    zip_handle: zipfile.ZipFile,
    min_quality: float,
) -> dict | None:
    """
    Build one .npz payload for a single (pid, date) group.

    Returns None if fewer than min_readings_per_day valid windows remain.
    """
    feature_list: list[np.ndarray] = []
    target_list:  list[list[float]] = []
    sqi_list:     list[float] = []
    t_start_list: list[float] = []

    # Sort chronologically so position features make sense
    day_rows = day_rows.sort_values("date_time").reset_index(drop=True)
    first_dt = pd.to_datetime(day_rows["date_time"].iloc[0])

    for _, row in day_rows.iterrows():
        # Skip low-quality or missing waveforms
        oq = float(row["optical_quality"])
        if oq < min_quality:
            continue
        wf_path = str(row["waveform_file_path"])
        if wf_path not in zip_handle.namelist():
            logger.debug("Waveform not found in zip: %s", wf_path)
            continue

        # Load waveform TSV from zip
        try:
            raw_bytes = zip_handle.read(wf_path)
            wf_df = pd.read_csv(StringIO(raw_bytes.decode("utf-8")), sep="\t")
        except Exception as exc:
            logger.debug("Failed to read %s: %s", wf_path, exc)
            continue
        if "optical" not in wf_df.columns:
            continue

        optical = wf_df["optical"].to_numpy(dtype=np.float64)
        if len(optical) < DECIMATE_FACTOR * MIN_SAMPLES:
            continue

        feats, sqi = _extract_ppg_features(optical)
        if not np.any(np.isfinite(feats)):
            continue  # all-NaN extraction — skip

        sbp  = float(row["sbp"])
        dbp  = float(row["dbp"])
        map_ = (sbp + 2.0 * dbp) / 3.0

        dt = pd.to_datetime(row["date_time"])
        t_start_min = float((dt - first_dt).total_seconds() / 60.0)

        feature_list.append(feats)
        target_list.append([sbp, dbp, map_])
        sqi_list.append(sqi)
        t_start_list.append(t_start_min)

    if not feature_list:
        return None

    features_arr = np.array(feature_list, dtype=np.float32)    # (T, 174)
    t_arr = np.array(t_start_list, dtype=np.float32)           # (T,)

    # Position features: window_start_min and gap_to_prev_min
    gap_arr = np.empty(len(t_arr), dtype=np.float32)
    gap_arr[0] = 0.0
    if len(t_arr) > 1:
        gap_arr[1:] = np.diff(t_arr)
    pos_feats = np.stack([t_arr, gap_arr], axis=1)             # (T, 2)

    features_176 = np.concatenate(
        [features_arr, pos_feats], axis=1
    ).astype(np.float32)                                        # (T, 176)

    return {
        "features":         features_176,
        "targets":          np.array(target_list, dtype=np.float32),
        "sqi":              np.array(sqi_list, dtype=np.float32),
        "window_start_min": t_arr,
    }


def run(
    tsv_path: Path,
    zip_path: Path,
    output_dir: Path,
    min_quality: float = 0.65,
    min_readings_per_day: int = 3,
    single_subject: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tsv_path, sep="\t")
    # Ambulatory phase only
    df = df[df["phase"] == "ambulatory"].copy()
    df["date"] = pd.to_datetime(df["date_time"]).dt.date.astype(str)

    if single_subject:
        df = df[df["pid"] == single_subject]
        if df.empty:
            raise ValueError(f"Subject {single_subject!r} not found in TSV")

    logger.info(
        "Loaded %d ambulatory measurements across %d subjects",
        len(df), df["pid"].nunique(),
    )

    n_written = 0
    n_skipped = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = set(zf.namelist())
        logger.info("Zip contains %d files", len(zip_names))

        groups = df.groupby(["pid", "date"])
        for (pid, date), day_rows in groups:
            payload = process_subject_day(day_rows, zf, min_quality)
            if payload is None or len(payload["features"]) < min_readings_per_day:
                n_skipped += 1
                continue

            out_path = output_dir / f"{pid}_{date}.npz"
            np.savez(
                out_path,
                features=payload["features"],
                targets=payload["targets"],
                sqi=payload["sqi"],
                window_start_min=payload["window_start_min"],
                participant_id=str(pid),
                date=str(date),
            )
            n_written += 1
            if n_written % 50 == 0:
                logger.info("Written %d / %d participant-days", n_written, len(groups))

    logger.info(
        "Done. Wrote %d participant-day .npz files, skipped %d (too few valid readings).",
        n_written, n_skipped,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Aurora-BP → Path A .npz adapter")
    ap.add_argument("--tsv",        required=True, help="measurements_oscillometric.tsv")
    ap.add_argument("--zip",        required=True, help="measurements_oscillometric.zip")
    ap.add_argument("--output-dir", required=True, help="directory for output .npz files")
    ap.add_argument("--min-quality",          type=float, default=0.65,
                    help="minimum optical_quality to include (default 0.65)")
    ap.add_argument("--min-readings-per-day", type=int,   default=3,
                    help="skip days with fewer valid readings (default 3)")
    ap.add_argument("--single-subject", default=None,
                    help="process only this pid (smoke-test mode)")
    args = ap.parse_args()

    run(
        tsv_path=Path(args.tsv),
        zip_path=Path(args.zip),
        output_dir=Path(args.output_dir),
        min_quality=args.min_quality,
        min_readings_per_day=args.min_readings_per_day,
        single_subject=args.single_subject,
    )


if __name__ == "__main__":
    main()
