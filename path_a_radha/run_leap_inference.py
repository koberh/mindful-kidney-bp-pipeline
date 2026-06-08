"""End-to-end LEAP inference — wrist PPG → relative BP trend.

Loads a LEAP recording (PPG + accel from CSV exports), windows it into
5-min segments, runs the Phase 3 model via RadhaEstimator (with MC dropout
for uncertainty), and emits BP trend predictions + a time-series plot.

Validates end-to-end on real wrist-worn PPG:
  - LEAP I/O reads correctly (CLAUDE.md gate 2)
  - Preprocessing handles 25 Hz wrist PPG (vs the 125 Hz MIMIC training rate)
  - Estimator produces sensible BP predictions
  - MC dropout uncertainty bands are reasonable

Output to benchmark/results/leap_inference_test/:
  - predictions.csv          per-window timestamp + SBP/DBP/MAP mean/std + SQI
  - timeseries.png           full-recording BP trend with MC dropout bands
  - summary.json             metadata, window counts, run stats

Usage:
  python -m path_a_radha.run_leap_inference \\
      --ppg     data/leap/SUBJ01_ppg25Hz.csv \\
      --accel   data/leap/SUBJ01_RAW.csv \\
      --checkpoint benchmark/results/path_a_first_run/best_model.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import load_leap_recording
from path_a_radha.estimator import RadhaEstimator


# ── Config ───────────────────────────────────────────────────────────────────

WINDOW_SECONDS: float = 300.0    # 5-min windows to match Radha §3.2.5
WINDOW_STEP_SECONDS: float = 300.0   # non-overlapping

# Sleep/wake detection — van Hees et al. 2013 (J Appl Physiol 115:1220) +
# time-of-day prior. ENMO threshold alone flags ALL sedentary activity as sleep,
# which is wrong for someone at a desk or watching TV. We require BOTH:
#   (1) ENMO below threshold (true stillness; tighter than van Hees default)
#   (2) Local clock-time in the typical nighttime window
ENMO_SLEEP_THRESHOLD_G: float = 0.015  # stricter than van Hees 0.04 (we add a TOD prior)
SLEEP_TIME_START_HOUR: int = 22        # 22:00 local
SLEEP_TIME_END_HOUR: int = 8           # 08:00 local
SLEEP_SMOOTHING_WINDOWS: int = 5       # rolling-window majority
SLEEP_MIN_BOUT_WINDOWS: int = 12       # min 60 min contiguous to count as sleep


def detect_sleep_windows(
    accel_windows: np.ndarray,
    window_local_hours: np.ndarray,
    enmo_threshold_g: float = ENMO_SLEEP_THRESHOLD_G,
    sleep_start_hour: int = SLEEP_TIME_START_HOUR,
    sleep_end_hour: int = SLEEP_TIME_END_HOUR,
    smoothing_n: int = SLEEP_SMOOTHING_WINDOWS,
    min_bout_n: int = SLEEP_MIN_BOUT_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sleep/wake detection from per-window accel + time-of-day prior.

    A window is classified as sleep iff ALL three:
      (1) Mean ENMO below threshold (van Hees 2013, threshold tightened from
          0.04 to 0.015 g because we additionally require condition 2)
      (2) Local clock-time within the typical sleep window
          (default 22:00 to 08:00 local)
      (3) Belongs to a contiguous sleep bout of at least min_bout_n windows
          (default 60 min) after rolling-majority smoothing

    accel_windows:      (T, W, 3) — tri-axial accel per window
    window_local_hours: (T,) — local-clock hour (0-23.99) at window start

    Returns:
      enmo_per_window: (T,) mean ENMO per window (g)
      sleep_flag:      (T,) bool — True if window is classified as sleep
    """
    # Per-window mean ENMO (van Hees 2013)
    accel_mag = np.linalg.norm(accel_windows, axis=2)
    enmo = np.maximum(0.0, accel_mag - 1.0)
    enmo_per_window = enmo.mean(axis=1).astype(np.float32)

    # Condition (1): low motion
    low_motion = enmo_per_window < enmo_threshold_g
    # Condition (2): in typical sleep clock hours
    if sleep_start_hour < sleep_end_hour:
        in_night = (window_local_hours >= sleep_start_hour) & (window_local_hours < sleep_end_hour)
    else:
        # Wrap around midnight (e.g. 22 -> 08)
        in_night = (window_local_hours >= sleep_start_hour) | (window_local_hours < sleep_end_hour)
    candidate = low_motion & in_night

    # Rolling-majority smoothing
    T = len(candidate)
    half = smoothing_n // 2
    smoothed = np.zeros(T, dtype=bool)
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        smoothed[i] = candidate[lo:hi].mean() >= 0.5

    # Condition (3): minimum bout length
    sleep_flag = smoothed.copy()
    i = 0
    while i < T:
        if not sleep_flag[i]:
            i += 1
            continue
        j = i
        while j < T and sleep_flag[j]:
            j += 1
        if (j - i) < min_bout_n:
            sleep_flag[i:j] = False
        i = j
    return enmo_per_window, sleep_flag


def compute_nocturnal_dip(
    sbp_mmhg: np.ndarray,
    dbp_mmhg: np.ndarray,
    sleep_flag: np.ndarray,
) -> dict:
    """Nocturnal dip per Radha §3.5: mean_awake - mean_sleep.

    Also reports dip as percentage of mean_awake (clinical convention:
    >10% normal "dippers", <10% "non-dippers"; Pickering 1990).

    Returns dict with sleep/wake means + dip values + dipper status.
    Returns dict with NaNs if either sleep or wake group is empty.
    """
    n_sleep = int(sleep_flag.sum())
    n_wake = int((~sleep_flag).sum())
    if n_sleep == 0 or n_wake == 0:
        return {
            "n_wake_windows":  n_wake,
            "n_sleep_windows": n_sleep,
            "mean_wake_sbp_mmhg":  float("nan"),
            "mean_sleep_sbp_mmhg": float("nan"),
            "mean_wake_dbp_mmhg":  float("nan"),
            "mean_sleep_dbp_mmhg": float("nan"),
            "sbp_dip_mmhg":  float("nan"),
            "dbp_dip_mmhg":  float("nan"),
            "sbp_dip_pct":   float("nan"),
            "dbp_dip_pct":   float("nan"),
            "dipper_status": "insufficient_data",
        }
    wake_sbp  = float(sbp_mmhg[~sleep_flag].mean())
    sleep_sbp = float(sbp_mmhg[sleep_flag].mean())
    wake_dbp  = float(dbp_mmhg[~sleep_flag].mean())
    sleep_dbp = float(dbp_mmhg[sleep_flag].mean())
    sbp_dip = wake_sbp - sleep_sbp
    dbp_dip = wake_dbp - sleep_dbp
    sbp_dip_pct = (sbp_dip / wake_sbp) * 100.0 if wake_sbp > 0 else float("nan")
    dbp_dip_pct = (dbp_dip / wake_dbp) * 100.0 if wake_dbp > 0 else float("nan")
    # Pickering 1990: SBP dip >=10% = "dipper" (normal); <10% = "non-dipper"
    if not np.isfinite(sbp_dip_pct):
        status = "unknown"
    elif sbp_dip_pct >= 10.0:
        status = "dipper"
    elif sbp_dip_pct >= 0.0:
        status = "non-dipper"
    else:
        status = "reverse-dipper"
    return {
        "n_wake_windows":  n_wake,
        "n_sleep_windows": n_sleep,
        "mean_wake_sbp_mmhg":  wake_sbp,
        "mean_sleep_sbp_mmhg": sleep_sbp,
        "mean_wake_dbp_mmhg":  wake_dbp,
        "mean_sleep_dbp_mmhg": sleep_dbp,
        "sbp_dip_mmhg":  sbp_dip,
        "dbp_dip_mmhg":  dbp_dip,
        "sbp_dip_pct":   sbp_dip_pct,
        "dbp_dip_pct":   dbp_dip_pct,
        "dipper_status": status,
    }


def find_continuous_bursts(
    timestamps: pd.DatetimeIndex,
    fs: int,
    max_gap_samples: int = 5,
) -> list[tuple[int, int]]:
    """Find contiguous-sample bursts in a duty-cycled recording.

    LEAP production records 15-min on / 15-min off (CentrePoint config), so
    `timestamps` has long gaps between bursts. A "continuous burst" is any
    stretch of consecutive samples whose inter-sample gap doesn't exceed
    max_gap_samples × (1/fs) seconds.

    Returns:
      list of (start_idx, end_idx) — half-open ranges into the sample array.
      A continuous recording returns [(0, n)].
    """
    n = len(timestamps)
    if n == 0:
        return []
    dt_expected = pd.Timedelta(seconds=1.0 / fs)
    dt_threshold = dt_expected * max_gap_samples
    diffs = timestamps[1:] - timestamps[:-1]
    gap_idx = np.where(diffs > dt_threshold)[0]  # indices i where ts[i+1] is too far from ts[i]
    if len(gap_idx) == 0:
        return [(0, n)]
    bursts: list[tuple[int, int]] = []
    prev_end = 0
    for gi in gap_idx:
        bursts.append((prev_end, int(gi) + 1))  # close after gi
        prev_end = int(gi) + 1
    bursts.append((prev_end, n))
    return bursts


def slice_windows(
    ppg: np.ndarray,
    accel: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fs: int,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = WINDOW_STEP_SECONDS,
    gap_aware: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Slice into fixed-length windows, respecting duty-cycle gaps.

    When `gap_aware=True` (default), the recording is first partitioned into
    continuous-sample bursts (find_continuous_bursts) and windows are emitted
    only within each burst. Bursts shorter than one window are skipped.

    Returns:
      ppg_windows:    (T, window_samples)
      accel_windows:  (T, window_samples, 3)
      window_starts:  list of pd.Timestamp (one per window)
    """
    n_samples = len(ppg)
    win_samp = int(window_seconds * fs)
    step_samp = int(step_seconds * fs)
    if n_samples < win_samp:
        raise ValueError(
            f"Recording too short: {n_samples / fs:.1f} s, need >= {window_seconds} s"
        )

    if gap_aware:
        bursts = find_continuous_bursts(timestamps, fs)
    else:
        bursts = [(0, n_samples)]

    ppg_list: list[np.ndarray] = []
    accel_list: list[np.ndarray] = []
    starts: list[pd.Timestamp] = []
    burst_diagnostics: list[dict] = []
    for b_start, b_end in bursts:
        b_len = b_end - b_start
        n_burst_windows = max(0, (b_len - win_samp) // step_samp + 1)
        burst_diagnostics.append({
            "start_idx": b_start, "end_idx": b_end,
            "duration_min": (b_len / fs) / 60.0,
            "n_windows": n_burst_windows,
        })
        for w in range(n_burst_windows):
            s = b_start + w * step_samp
            e = s + win_samp
            ppg_list.append(ppg[s:e])
            accel_list.append(accel[s:e])
            starts.append(timestamps[s])
    if not ppg_list:
        burst_summary = ", ".join(f"{b['duration_min']:.1f}m" for b in burst_diagnostics)
        raise ValueError(
            f"No bursts long enough for a {window_seconds}s window "
            f"(saw {len(bursts)} bursts of durations [{burst_summary}])"
        )
    return np.stack(ppg_list), np.stack(accel_list), starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppg", required=True, type=Path,
                        help="Path to LEAP ppg CSV (25 Hz or 100 Hz)")
    parser.add_argument("--accel", required=True, type=Path,
                        help="Path to LEAP RAW.csv (32 Hz accelerometer)")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to trained best_model.pt")
    parser.add_argument("--participant-id", default="TEST_LEAP",
                        help="Label for this recording")
    parser.add_argument("--ppg-fs", type=int, default=25,
                        help="LEAP PPG sample rate (25 or 100)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmark/results/leap_inference_test"))
    parser.add_argument("--n-mc", type=int, default=30,
                        help="Monte Carlo dropout samples per window")
    parser.add_argument("--tz-offset", type=float, default=-5.0,
                        help="Hours from UTC (test recording is EST = -5.0)")
    parser.add_argument("--calib-sbp", type=float, default=120.0,
                        help="Per-participant baseline SBP added to relative predictions (mmHg)")
    parser.add_argument("--calib-dbp", type=float, default=75.0,
                        help="Per-participant baseline DBP (mmHg)")
    parser.add_argument("--simulate-duty-cycle", action="store_true",
                        help="Inject 15-min on / 15-min off gaps into the recording "
                             "before windowing (synthetic test of production handling).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LEAP recording")
    print(f"  PPG:   {args.ppg}")
    print(f"  Accel: {args.accel}")
    rec = load_leap_recording(
        ppg_path=args.ppg,
        accel_path=args.accel,
        participant_id=args.participant_id,
        tz_offset_hours=args.tz_offset,
        ppg_fs=args.ppg_fs,
    )
    duration_hr = len(rec.ppg_green) / rec.fs / 3600.0
    print(f"  Loaded: {len(rec.ppg_green):,} samples @ {rec.fs} Hz = {duration_hr:.2f} hr")
    print(f"  Start UTC: {rec.metadata.start_utc}")
    print(f"  PPG range: [{rec.ppg_green.min():.3g}, {rec.ppg_green.max():.3g}]")
    print(f"  Accel mag range: [{np.linalg.norm(rec.accel, axis=1).min():.3f}, "
          f"{np.linalg.norm(rec.accel, axis=1).max():.3f}] g")

    # Optional: inject synthetic duty-cycle gaps for testing production handling.
    # Keep every 15-min "on" segment, drop every 15-min "off" segment.
    if args.simulate_duty_cycle:
        on_samples = 15 * 60 * rec.fs
        cycle_samples = 30 * 60 * rec.fs  # 15 on + 15 off
        n = len(rec.ppg_green)
        keep_mask = (np.arange(n) % cycle_samples) < on_samples
        rec.ppg_ch0[~keep_mask] = np.nan  # this won't actually drop, just mark
        # For real "duty-cycle" behavior we need to DROP samples, not NaN them.
        # Build a new ppg / accel / timestamps array with only the kept samples.
        kept_idx = np.where(keep_mask)[0]
        rec.ppg_ch0 = rec.ppg_ch0[kept_idx]
        rec.ppg_ch1 = rec.ppg_ch1[kept_idx]
        rec.accel = rec.accel[kept_idx]
        rec.timestamps_utc = rec.timestamps_utc[kept_idx]
        print(f"  [SIMULATED DUTY CYCLE] kept {len(kept_idx):,}/{n:,} samples "
              f"({len(kept_idx)/n*100:.1f}%) in 15-min on/off pattern")

    print()
    print(f"Slicing into 5-min windows (step={WINDOW_STEP_SECONDS:.0f}s, gap-aware)")
    # Detect continuous bursts first for diagnostic reporting
    bursts = find_continuous_bursts(rec.timestamps_utc, rec.fs)
    burst_durations_min = [
        ((rec.timestamps_utc[e - 1] - rec.timestamps_utc[s]).total_seconds() / 60.0)
        for s, e in bursts
    ]
    print(f"  Detected {len(bursts)} continuous burst(s); "
          f"durations: {[f'{d:.1f}' for d in burst_durations_min[:10]]}"
          f"{' ...' if len(bursts) > 10 else ''} min")
    ppg_wins, accel_wins, starts = slice_windows(
        rec.ppg_green, rec.accel, rec.timestamps_utc, fs=rec.fs,
    )
    print(f"  Windows: {len(ppg_wins)}")

    # Loading model + running inference
    print()
    print(f"Loading model from {args.checkpoint}")
    estimator = RadhaEstimator(
        checkpoint_path=args.checkpoint,
        calibration_sbp_mean=args.calib_sbp,
        calibration_dbp_mean=args.calib_dbp,
        device="cpu",
        n_mc_samples=args.n_mc,
    )
    # Override the estimator's preprocess config fs so bandpass uses LEAP fs
    estimator.cfg.fs = rec.fs
    print(f"  Calibration baseline: SBP={args.calib_sbp:.1f}, DBP={args.calib_dbp:.1f} mmHg")
    print(f"  MC dropout samples: {args.n_mc}")

    print()
    print(f"Running inference on {len(ppg_wins)} windows ...")
    # Position features (matches mimic_adapter convention):
    # time_since_start_min for each window's start.
    window_starts_min = np.array(
        [(s - starts[0]).total_seconds() / 60.0 for s in starts],
        dtype=np.float32,
    )
    result = estimator.predict(
        ppg=ppg_wins, accel=accel_wins, fs=rec.fs,
        window_starts_min=window_starts_min,
    )

    # Sleep / wake detection
    print()
    print("Detecting sleep/wake windows via ENMO threshold (van Hees 2013) + time-of-day prior ...")
    # Local clock hour per window (using tz_offset)
    window_local_hours = np.array(
        [((s + pd.Timedelta(hours=args.tz_offset)).hour
          + (s + pd.Timedelta(hours=args.tz_offset)).minute / 60.0)
         for s in starts],
        dtype=np.float32,
    )
    enmo_per_win, sleep_flag = detect_sleep_windows(
        accel_wins, window_local_hours=window_local_hours,
    )
    n_sleep = int(sleep_flag.sum())
    print(f"  Sleep windows:  {n_sleep:>4} ({n_sleep * 5:>4} min)")
    print(f"  Awake windows:  {int((~sleep_flag).sum()):>4} "
          f"({int((~sleep_flag).sum()) * 5:>4} min)")
    dip_metrics = compute_nocturnal_dip(
        sbp_mmhg=result["sbp"].astype(np.float64),
        dbp_mmhg=result["dbp"].astype(np.float64),
        sleep_flag=sleep_flag,
    )
    if dip_metrics["dipper_status"] != "insufficient_data":
        print(f"  Mean awake SBP:  {dip_metrics['mean_wake_sbp_mmhg']:5.1f} mmHg  •  "
              f"sleep: {dip_metrics['mean_sleep_sbp_mmhg']:5.1f}  •  "
              f"dip: {dip_metrics['sbp_dip_mmhg']:+.1f} mmHg "
              f"({dip_metrics['sbp_dip_pct']:+.1f}%) — {dip_metrics['dipper_status']}")
        print(f"  Mean awake DBP:  {dip_metrics['mean_wake_dbp_mmhg']:5.1f} mmHg  •  "
              f"sleep: {dip_metrics['mean_sleep_dbp_mmhg']:5.1f}  •  "
              f"dip: {dip_metrics['dbp_dip_mmhg']:+.1f} mmHg "
              f"({dip_metrics['dbp_dip_pct']:+.1f}%)")

    # Persist
    df = pd.DataFrame({
        "window_start_utc":   [t.isoformat() for t in starts],
        "minutes_since_start": [(t - starts[0]).total_seconds() / 60.0 for t in starts],
        "sbp_mmhg":           result["sbp"].astype(np.float64),
        "sbp_std_mmhg":       result["sbp_std"].astype(np.float64),
        "dbp_mmhg":           result["dbp"].astype(np.float64),
        "dbp_std_mmhg":       result["dbp_std"].astype(np.float64),
        "rel_sbp":            result["relative"][:, 0].astype(np.float64),
        "rel_dbp":            result["relative"][:, 1].astype(np.float64),
        "rel_map":            result["relative"][:, 2].astype(np.float64),
        "sqi":                result["confidence"].astype(np.float64),
        "enmo_g":             enmo_per_win.astype(np.float64),
        "sleep_flag":         sleep_flag.astype(bool),
    })
    csv_path = args.out_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Predictions: {csv_path}")

    summary = {
        "participant_id":      args.participant_id,
        "recording_start_utc": rec.metadata.start_utc.isoformat(),
        "duration_hours":      duration_hr,
        "n_windows":           int(len(ppg_wins)),
        "ppg_fs":              rec.fs,
        "checkpoint":          str(args.checkpoint),
        "n_mc_samples":        args.n_mc,
        "calibration_sbp":     args.calib_sbp,
        "calibration_dbp":     args.calib_dbp,
        "sbp_mmhg":            {"mean": float(df["sbp_mmhg"].mean()),
                                "median": float(df["sbp_mmhg"].median()),
                                "min": float(df["sbp_mmhg"].min()),
                                "max": float(df["sbp_mmhg"].max())},
        "dbp_mmhg":            {"mean": float(df["dbp_mmhg"].mean()),
                                "median": float(df["dbp_mmhg"].median()),
                                "min": float(df["dbp_mmhg"].min()),
                                "max": float(df["dbp_mmhg"].max())},
        "sqi":                 {"mean": float(df["sqi"].mean()),
                                "median": float(df["sqi"].median()),
                                "min": float(df["sqi"].min()),
                                "max": float(df["sqi"].max()),
                                "frac_above_0.5": float((df["sqi"] >= 0.5).mean())},
        "uncertainty_sbp_mean": float(df["sbp_std_mmhg"].mean()),
        "uncertainty_dbp_mean": float(df["dbp_std_mmhg"].mean()),
        "dip_metrics":         dip_metrics,
        "generated_utc":       datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary:     {args.out_dir / 'summary.json'}")

    # Time-series plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    t = df["minutes_since_start"].to_numpy()

    # Helper: shade sleep regions on all subplots
    def shade_sleep(axis):
        in_bout = False
        bout_start = 0
        for i, s in enumerate(sleep_flag):
            if s and not in_bout:
                in_bout = True
                bout_start = i
            elif not s and in_bout:
                axis.axvspan(t[bout_start], t[i], color="C0", alpha=0.10, lw=0)
                in_bout = False
        if in_bout:
            axis.axvspan(t[bout_start], t[-1], color="C0", alpha=0.10, lw=0)

    # SBP with MC band
    ax[0].plot(t, df["sbp_mmhg"], color="C3", lw=1.2, label="predicted SBP")
    ax[0].fill_between(t, df["sbp_mmhg"] - df["sbp_std_mmhg"], df["sbp_mmhg"] + df["sbp_std_mmhg"],
                       color="C3", alpha=0.2, label="MC dropout ±1σ")
    ax[0].axhline(args.calib_sbp, color="C7", ls="--", lw=0.8, label=f"baseline {args.calib_sbp:.0f}")
    shade_sleep(ax[0])
    ax[0].set_ylabel("SBP (mmHg)"); ax[0].legend(loc="upper right", fontsize=8); ax[0].grid(alpha=0.3)
    title = (f"{args.participant_id} — BP trend "
             f"(calib SBP={args.calib_sbp:.0f}/DBP={args.calib_dbp:.0f}; "
             f"shaded = sleep)")
    if dip_metrics["dipper_status"] != "insufficient_data":
        title += (f" — SBP dip {dip_metrics['sbp_dip_mmhg']:+.1f} mmHg "
                  f"({dip_metrics['sbp_dip_pct']:+.1f}%, {dip_metrics['dipper_status']})")
    ax[0].set_title(title)

    # DBP with MC band
    ax[1].plot(t, df["dbp_mmhg"], color="C0", lw=1.2, label="predicted DBP")
    ax[1].fill_between(t, df["dbp_mmhg"] - df["dbp_std_mmhg"], df["dbp_mmhg"] + df["dbp_std_mmhg"],
                       color="C0", alpha=0.2, label="MC dropout ±1σ")
    ax[1].axhline(args.calib_dbp, color="C7", ls="--", lw=0.8, label=f"baseline {args.calib_dbp:.0f}")
    shade_sleep(ax[1])
    ax[1].set_ylabel("DBP (mmHg)"); ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=0.3)

    # ENMO + sleep threshold
    ax[2].plot(t, df["enmo_g"], color="C4", lw=1.0)
    ax[2].axhline(ENMO_SLEEP_THRESHOLD_G, color="C7", ls="--", lw=0.8,
                  label=f"sleep threshold {ENMO_SLEEP_THRESHOLD_G} g (van Hees 2013)")
    shade_sleep(ax[2])
    ax[2].set_ylabel("ENMO (g)"); ax[2].set_ylim(0, max(0.3, df["enmo_g"].max() * 1.05))
    ax[2].legend(loc="upper right", fontsize=8); ax[2].grid(alpha=0.3)

    # SQI
    ax[3].plot(t, df["sqi"], color="C2", lw=1.2)
    ax[3].axhline(0.5, color="C7", ls="--", lw=0.8, label="SQI=0.5 (low-quality threshold)")
    shade_sleep(ax[3])
    ax[3].set_xlabel("Minutes from start of recording")
    ax[3].set_ylabel("Signal Quality Index"); ax[3].set_ylim(0, 1)
    ax[3].legend(loc="upper right", fontsize=8); ax[3].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = args.out_dir / "timeseries.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot:        {plot_path}")

    print()
    print("=" * 60)
    print(f"  Recording duration: {duration_hr:.2f} hr ({len(ppg_wins)} 5-min windows)")
    print(f"  SBP: median={summary['sbp_mmhg']['median']:.1f} mmHg, "
          f"range=[{summary['sbp_mmhg']['min']:.1f}, {summary['sbp_mmhg']['max']:.1f}]")
    print(f"  DBP: median={summary['dbp_mmhg']['median']:.1f} mmHg, "
          f"range=[{summary['dbp_mmhg']['min']:.1f}, {summary['dbp_mmhg']['max']:.1f}]")
    print(f"  SQI: median={summary['sqi']['median']:.3f}, "
          f"{summary['sqi']['frac_above_0.5']*100:.1f}% of windows have SQI >= 0.5")
    print(f"  MC dropout σ: SBP={summary['uncertainty_sbp_mean']:.2f}, "
          f"DBP={summary['uncertainty_dbp_mean']:.2f} mmHg")
    print("=" * 60)


if __name__ == "__main__":
    main()
