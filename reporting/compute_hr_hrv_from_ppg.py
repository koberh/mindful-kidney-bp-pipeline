#!/usr/bin/env python3
"""
Compute HeartRate.csv and HeartRateVar.csv from raw 100 Hz green PPG day-files.

Input:  data/<subject-id>/ppg-green-100-hz/*.csv
        (gzip-compressed, two interleaved channels, Unix-ms timestamps)
Output: data\\OneDrive_actigraph\\HeartRate.csv
        data\\OneDrive_actigraph\\HeartRateVar.csv
        (format matches what generate_full_report.py's load_hr_hrv_weekly expects)

Strategy:
  - Load each day, use Channel 0 (negated so systolic peaks point up)
  - Bandpass 0.5–4 Hz across the full day, then detect peaks
  - Aggregate IBIs into 1-min HR windows and 5-min RMSSD windows
  - Reliability score: fraction of IBIs in 400–2000 ms range × 100
"""

import glob
import io
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# ── Paths ─────────────────────────────────────────────────────────────────────

PPG_DIR  = Path(r"D:\Mindful Kidney\CentrePoint Data Pull\data\<subject-id>\ppg-green-100-hz")
OUT_DIR  = Path(r"data/hr_hrv_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS             = 100          # Hz
SAT_THRESHOLD  = 2_000_000    # |green| >= this → saturated, drop
IBI_MIN_MS     = 400          # 150 bpm upper bound
IBI_MAX_MS     = 2000         # 30 bpm lower bound
HR_WINDOW_S    = 60           # seconds per HR epoch
HRV_WINDOW_S   = 300          # 5-minute HRV window
HRV_STEP_S     = 30           # 30-second hop
MIN_VALID_FRAC = 0.5          # drop windows where >50% of samples are saturated

# ── Signal processing ─────────────────────────────────────────────────────────

def make_bandpass():
    return signal.butter(4, [0.5, 4.0], btype="band", fs=FS, output="sos")


def detect_peaks_in_day(green_neg: np.ndarray) -> np.ndarray:
    """Bandpass filter and detect systolic peaks. Returns sample indices."""
    sos = make_bandpass()
    filtered = signal.sosfiltfilt(sos, green_neg)

    # Adaptive prominence: 40th-percentile of absolute values of the filtered signal
    prom = np.percentile(np.abs(filtered), 40)
    peaks, _ = signal.find_peaks(filtered, distance=40, prominence=prom)
    return peaks


def compute_ibis(peak_indices: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """Return inter-beat intervals in ms from peak sample indices."""
    if len(peak_indices) < 2:
        return np.array([], dtype=float)
    t_peaks = timestamps_ms[peak_indices]
    return np.diff(t_peaks).astype(float)


# ── Per-window aggregation ────────────────────────────────────────────────────

def ibis_in_window(peak_ms: np.ndarray, win_start_ms: float, win_end_ms: float):
    """IBIs whose *start* peak falls within [win_start_ms, win_end_ms)."""
    # Each IBI i spans peak_ms[i] → peak_ms[i+1]; attribute to start peak
    if len(peak_ms) < 2:
        return np.array([], dtype=float)
    starts = peak_ms[:-1]
    ibis   = np.diff(peak_ms).astype(float)
    mask   = (starts >= win_start_ms) & (starts < win_end_ms)
    return ibis[mask]


def hr_from_ibis(ibis: np.ndarray) -> tuple[float | None, float]:
    """Return (hr_bpm, reliability_pct) from raw IBIs."""
    if len(ibis) == 0:
        return None, 0.0
    valid = ibis[(ibis >= IBI_MIN_MS) & (ibis <= IBI_MAX_MS)]
    reliability = 100.0 * len(valid) / len(ibis) if len(ibis) > 0 else 0.0
    if len(valid) == 0:
        return None, reliability
    return 60_000.0 / valid.mean(), reliability


def rmssd_from_ibis(ibis: np.ndarray) -> tuple[float | None, float]:
    """Return (rmssd_ms, reliability_pct) from raw IBIs."""
    if len(ibis) == 0:
        return None, 0.0
    valid = ibis[(ibis >= IBI_MIN_MS) & (ibis <= IBI_MAX_MS)]
    reliability = 100.0 * len(valid) / len(ibis) if len(ibis) > 0 else 0.0
    if len(valid) < 4:
        return None, reliability
    diffs = np.diff(valid)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    return rmssd, reliability


# ── Day processing ────────────────────────────────────────────────────────────

def process_day(fpath: Path, hr_rows: list, hrv_rows: list) -> None:
    print(f"  Processing {fpath.name} ...", end=" ", flush=True)

    with gzip.open(fpath, "rb") as gz:
        df = pd.read_csv(
            io.BytesIO(gz.read()),
            usecols=["Green", "Channel", "Timestamp"],
            dtype={"Green": "int64", "Channel": "int8", "Timestamp": "int64"},
        )

    ch0 = df[df["Channel"] == 0].copy().reset_index(drop=True)
    del df

    # Drop saturated samples and negate (so systolic peaks point up)
    ch0 = ch0[ch0["Green"].abs() < SAT_THRESHOLD].reset_index(drop=True)
    if len(ch0) < FS * 60:
        print("too little data, skipping")
        return

    timestamps_ms = ch0["Timestamp"].to_numpy(dtype=np.int64)
    green_neg     = (-ch0["Green"]).to_numpy(dtype=np.float64)

    # Detect peaks across the full day's (filtered) signal
    peaks = detect_peaks_in_day(green_neg)
    if len(peaks) < 10:
        print("no peaks detected, skipping")
        return

    peak_ms = timestamps_ms[peaks]

    day_start_ms = int(timestamps_ms[0])
    day_end_ms   = int(timestamps_ms[-1])

    # ── 1-minute HR windows ──────────────────────────────────────────────────
    t = day_start_ms
    while t + HR_WINDOW_S * 1000 <= day_end_ms:
        t_end = t + HR_WINDOW_S * 1000
        ibis  = ibis_in_window(peak_ms, t, t_end)
        hr, _ = hr_from_ibis(ibis)
        if hr is not None:
            ts_utc = pd.Timestamp(t, unit="ms", tz="UTC")
            hr_rows.append({
                "timestamp":    ts_utc.isoformat(),
                "HeartRate":    round(hr, 2),
                "timestamputc": ts_utc.isoformat(),
            })
        t = t_end

    # ── 5-minute HRV windows (30-second hop) ────────────────────────────────
    t = day_start_ms
    while t + HRV_WINDOW_S * 1000 <= day_end_ms:
        t_end  = t + HRV_WINDOW_S * 1000
        ibis   = ibis_in_window(peak_ms, t, t_end)
        rmssd, reliability = rmssd_from_ibis(ibis)
        if rmssd is not None:
            ts_start = pd.Timestamp(t,     unit="ms", tz="UTC")
            ts_end   = pd.Timestamp(t_end, unit="ms", tz="UTC")
            hr, _    = hr_from_ibis(ibis)
            mean_nn  = (60_000.0 / hr) if hr else np.nan
            hrv_rows.append({
                "start":                       ts_start.isoformat(),
                "end":                         ts_end.isoformat(),
                "hrv_epoch_start":             ts_start.isoformat(),
                "hrv_epoch_end":               ts_end.isoformat(),
                "window_size_seconds":         HRV_WINDOW_S,
                "hrv_reliability":             round(reliability, 1),
                "rmssd":                       round(rmssd, 3),
                "mean_nn":                     round(mean_nn, 2) if not np.isnan(mean_nn) else np.nan,
            })
        t += HRV_STEP_S * 1000

    print(f"done ({len(peaks)} peaks)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = sorted(PPG_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No PPG files found in {PPG_DIR}")

    print(f"Found {len(files)} day-file(s) in {PPG_DIR}\n")

    hr_rows:  list[dict] = []
    hrv_rows: list[dict] = []

    for fpath in files:
        process_day(fpath, hr_rows, hrv_rows)

    hr_df  = pd.DataFrame(hr_rows)
    hrv_df = pd.DataFrame(hrv_rows)

    hr_out  = OUT_DIR / "HeartRate.csv"
    hrv_out = OUT_DIR / "HeartRateVar.csv"

    hr_df.to_csv(hr_out,  index=False)
    hrv_df.to_csv(hrv_out, index=False)

    print(f"\nWrote {len(hr_df)} HR rows  -> {hr_out}")
    print(f"Wrote {len(hrv_df)} HRV rows -> {hrv_out}")

    # Quick sanity check
    if not hr_df.empty:
        hrs = pd.to_numeric(hr_df["HeartRate"], errors="coerce").dropna()
        print(f"\nHR stats:  mean={hrs.mean():.1f} bpm, "
              f"min={hrs.min():.1f}, max={hrs.max():.1f}")
    if not hrv_df.empty:
        rmssd = pd.to_numeric(hrv_df["rmssd"], errors="coerce").dropna()
        rel   = pd.to_numeric(hrv_df["hrv_reliability"], errors="coerce").dropna()
        print(f"HRV stats: mean RMSSD={rmssd.mean():.1f} ms, "
              f"mean reliability={rel.mean():.1f}%")
        print(f"Windows with reliability ≥ 70: {(rel >= 70).sum()} / {len(rel)}")


if __name__ == "__main__":
    main()
