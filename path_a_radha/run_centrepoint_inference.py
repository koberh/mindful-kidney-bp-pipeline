"""End-to-end BP-trend inference on CentrePoint V3 API PPG exports.

Sibling of ``run_leap_inference.py`` for data pulled from the CentrePoint API
(gzipped, epoch-ms timestamps, no accelerometer) rather than LEAP ActiLife
CSV exports. See ``common/io/centrepoint.py`` for the format details.

Processes one or more daily PPG files. Each day is loaded, sliced into
gap-aware 5-min windows (the recording is duty-cycled 15-min on/off), run
through the Phase 3 RadhaLSTM, and the per-day predictions are concatenated
into a single recording-level predictions.csv + multi-day plot.

The model outputs RELATIVE BP (deviation from a baseline). With no cuff
calibration the absolute mmHg are anchored to --calib-sbp/--calib-dbp
(default 120/75) and only the TRENDS are meaningful.

Usage:
  python -m path_a_radha.run_centrepoint_inference \\
      --ppg-dir data/ppg-green-100-hz \\
      --checkpoint benchmark/results/path_a_phase5/best_model.pt \\
      --ppg-fs 100 --tz-offset -7.0 \\
      --participant-id SUBJ01 \\
      --out-dir benchmark/results/centrepoint_SUBJ01
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.io.centrepoint import load_centrepoint_recording
from path_a_radha.estimator import RadhaEstimator
from path_a_radha.run_leap_inference import (
    WINDOW_STEP_SECONDS,
    compute_nocturnal_dip,
    detect_sleep_windows,
    find_continuous_bursts,
    slice_windows,
)


def _process_day(
    ppg_file: Path,
    estimator: RadhaEstimator,
    participant_id: str,
    tz_offset: float,
    ppg_fs: int,
) -> pd.DataFrame | None:
    """Load one daily CentrePoint file, window it, run inference.

    Returns a per-window DataFrame (absolute UTC timestamps) or None if the
    day had no burst long enough for a 5-min window.
    """
    rec = load_centrepoint_recording(
        ppg_paths=[ppg_file],
        participant_id=participant_id,
        tz_offset_hours=tz_offset,
        ppg_fs=ppg_fs,
    )
    dur_hr = len(rec.ppg_green) / rec.fs / 3600.0
    bursts = find_continuous_bursts(rec.timestamps_utc, rec.fs)
    print(f"  {ppg_file.name}: {len(rec.ppg_green):,} samp @ {rec.fs}Hz "
          f"({dur_hr:.1f}h), {len(bursts)} burst(s), "
          f"PPG range [{rec.ppg_green.min():.3g}, {rec.ppg_green.max():.3g}]")

    try:
        ppg_wins, accel_wins, starts = slice_windows(
            rec.ppg_green, rec.accel, rec.timestamps_utc, fs=rec.fs,
        )
    except ValueError as e:
        print(f"    SKIP: {e}")
        return None

    # Position feature: minutes since this day's first window (per-day relative,
    # matching the single-stay scale the model was trained on).
    window_starts_min = np.array(
        [(s - starts[0]).total_seconds() / 60.0 for s in starts],
        dtype=np.float32,
    )
    result = estimator.predict(
        ppg=ppg_wins, accel=accel_wins, fs=rec.fs,
        window_starts_min=window_starts_min,
    )

    # Sleep/wake from time-of-day prior (dummy accel → ENMO ~0 everywhere)
    window_local_hours = np.array(
        [((s + pd.Timedelta(hours=tz_offset)).hour
          + (s + pd.Timedelta(hours=tz_offset)).minute / 60.0) for s in starts],
        dtype=np.float32,
    )
    enmo_per_win, sleep_flag = detect_sleep_windows(
        accel_wins, window_local_hours=window_local_hours,
    )

    print(f"    windows: {len(ppg_wins):3d}  "
          f"SBP rel median {np.median(result['relative'][:, 0]):+.1f}  "
          f"SQI median {np.median(result['confidence']):.2f}")

    return pd.DataFrame({
        "window_start_utc":   [t.isoformat() for t in starts],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ppg-dir", type=Path,
                     help="Directory of daily CentrePoint ppg CSVs (YYYY-MM-DD.csv)")
    src.add_argument("--ppg", type=Path, nargs="+",
                     help="Explicit daily ppg CSV file(s)")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--participant-id", default="CP_SUBJECT")
    parser.add_argument("--ppg-fs", type=int, default=100)
    parser.add_argument("--tz-offset", type=float, default=-7.0,
                        help="Local-to-UTC hours for sleep/wake (San Diego May = -7.0 PDT)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmark/results/centrepoint_inference"))
    parser.add_argument("--n-mc", type=int, default=30)
    parser.add_argument("--calib-sbp", type=float, default=120.0)
    parser.add_argument("--calib-dbp", type=float, default=75.0)
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Optional subset of YYYY-MM-DD dates to process")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Process at most this many days (for quick tests)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the list of daily files
    if args.ppg_dir is not None:
        files = sorted(args.ppg_dir.glob("*.csv"))
    else:
        files = sorted(args.ppg)
    if args.dates:
        wanted = set(args.dates)
        files = [f for f in files if f.stem in wanted]
    if args.max_days:
        files = files[: args.max_days]
    if not files:
        raise SystemExit("No matching daily PPG files found.")

    print(f"Loading model from {args.checkpoint}")
    estimator = RadhaEstimator(
        checkpoint_path=args.checkpoint,
        calibration_sbp_mean=args.calib_sbp,
        calibration_dbp_mean=args.calib_dbp,
        device="cpu",
        n_mc_samples=args.n_mc,
    )
    estimator.cfg.fs = args.ppg_fs
    print(f"  Calibration baseline: SBP={args.calib_sbp:.1f}, DBP={args.calib_dbp:.1f} mmHg")
    print(f"  Processing {len(files)} day(s) at {args.ppg_fs} Hz (native), "
          f"tz_offset={args.tz_offset}\n")

    day_frames: list[pd.DataFrame] = []
    for f in files:
        df_day = _process_day(f, estimator, args.participant_id,
                              args.tz_offset, args.ppg_fs)
        if df_day is not None and len(df_day):
            day_frames.append(df_day)

    if not day_frames:
        raise SystemExit("No windows produced across any day.")

    df = pd.concat(day_frames, axis=0).reset_index(drop=True)
    df["window_start_utc_dt"] = pd.to_datetime(df["window_start_utc"])
    df = df.sort_values("window_start_utc_dt").reset_index(drop=True)
    t0 = df["window_start_utc_dt"].iloc[0]
    df["hours_since_start"] = (df["window_start_utc_dt"] - t0).dt.total_seconds() / 3600.0

    csv_path = args.out_dir / "predictions.csv"
    df.drop(columns=["window_start_utc_dt"]).to_csv(csv_path, index=False)
    print(f"\n  Predictions: {csv_path}  ({len(df)} windows)")

    sleep_flag = df["sleep_flag"].to_numpy(dtype=bool)
    dip_metrics = compute_nocturnal_dip(
        sbp_mmhg=df["sbp_mmhg"].to_numpy(),
        dbp_mmhg=df["dbp_mmhg"].to_numpy(),
        sleep_flag=sleep_flag,
    )

    summary = {
        "participant_id":      args.participant_id,
        "n_days":              len(files),
        "n_windows":           int(len(df)),
        "ppg_fs":              args.ppg_fs,
        "tz_offset_hours":     args.tz_offset,
        "checkpoint":          str(args.checkpoint),
        "calibration_sbp":     args.calib_sbp,
        "calibration_dbp":     args.calib_dbp,
        "recording_start_utc": t0.isoformat(),
        "recording_end_utc":   df["window_start_utc_dt"].iloc[-1].isoformat(),
        "rel_sbp":             {"mean": float(df["rel_sbp"].mean()),
                                "min": float(df["rel_sbp"].min()),
                                "max": float(df["rel_sbp"].max())},
        "sbp_mmhg":            {"mean": float(df["sbp_mmhg"].mean()),
                                "median": float(df["sbp_mmhg"].median()),
                                "min": float(df["sbp_mmhg"].min()),
                                "max": float(df["sbp_mmhg"].max())},
        "dbp_mmhg":            {"mean": float(df["dbp_mmhg"].mean()),
                                "median": float(df["dbp_mmhg"].median()),
                                "min": float(df["dbp_mmhg"].min()),
                                "max": float(df["dbp_mmhg"].max())},
        "sqi":                 {"median": float(df["sqi"].median()),
                                "frac_above_0.5": float((df["sqi"] >= 0.5).mean())},
        "uncertainty_sbp_mean": float(df["sbp_std_mmhg"].mean()),
        "uncertainty_dbp_mean": float(df["dbp_std_mmhg"].mean()),
        "dip_metrics":         dip_metrics,
        "generated_utc":       datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary:     {args.out_dir / 'summary.json'}")

    # ── Plot (x-axis = hours since recording start) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df["hours_since_start"].to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    def shade_sleep(axis):
        in_bout = False
        start = 0
        for i, s in enumerate(sleep_flag):
            if s and not in_bout:
                in_bout, start = True, i
            elif not s and in_bout:
                axis.axvspan(t[start], t[i], color="C0", alpha=0.10, lw=0)
                in_bout = False
        if in_bout:
            axis.axvspan(t[start], t[-1], color="C0", alpha=0.10, lw=0)

    ax[0].plot(t, df["sbp_mmhg"], color="C3", lw=1.0, label="predicted SBP")
    ax[0].fill_between(t, df["sbp_mmhg"] - df["sbp_std_mmhg"],
                       df["sbp_mmhg"] + df["sbp_std_mmhg"], color="C3", alpha=0.2)
    ax[0].axhline(args.calib_sbp, color="C7", ls="--", lw=0.8,
                  label=f"baseline {args.calib_sbp:.0f}")
    shade_sleep(ax[0])
    ax[0].set_ylabel("SBP (mmHg)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)
    title = (f"{args.participant_id} — CentrePoint {args.ppg_fs}Hz PPG, "
             f"{len(files)} day(s), UNCALIBRATED (baseline {args.calib_sbp:.0f}/"
             f"{args.calib_dbp:.0f}); shaded = sleep (time-of-day)")
    if dip_metrics["dipper_status"] != "insufficient_data":
        title += (f"\nSBP dip {dip_metrics['sbp_dip_mmhg']:+.1f} mmHg "
                  f"({dip_metrics['sbp_dip_pct']:+.1f}%, {dip_metrics['dipper_status']})")
    ax[0].set_title(title)

    ax[1].plot(t, df["dbp_mmhg"], color="C0", lw=1.0, label="predicted DBP")
    ax[1].fill_between(t, df["dbp_mmhg"] - df["dbp_std_mmhg"],
                       df["dbp_mmhg"] + df["dbp_std_mmhg"], color="C0", alpha=0.2)
    ax[1].axhline(args.calib_dbp, color="C7", ls="--", lw=0.8,
                  label=f"baseline {args.calib_dbp:.0f}")
    shade_sleep(ax[1])
    ax[1].set_ylabel("DBP (mmHg)"); ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, df["sqi"], color="C2", lw=1.0)
    ax[2].axhline(0.5, color="C7", ls="--", lw=0.8, label="SQI=0.5")
    shade_sleep(ax[2])
    ax[2].set_ylabel("SQI"); ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("Hours since recording start")
    ax[2].legend(loc="upper right", fontsize=8); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = args.out_dir / "timeseries.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot:        {plot_path}")

    print("\n" + "=" * 60)
    print(f"  {len(df)} windows over {len(files)} day(s)")
    print(f"  SBP: median={summary['sbp_mmhg']['median']:.1f} mmHg, "
          f"range=[{summary['sbp_mmhg']['min']:.1f}, {summary['sbp_mmhg']['max']:.1f}]")
    print(f"  DBP: median={summary['dbp_mmhg']['median']:.1f} mmHg, "
          f"range=[{summary['dbp_mmhg']['min']:.1f}, {summary['dbp_mmhg']['max']:.1f}]")
    print(f"  SQI: median={summary['sqi']['median']:.3f}, "
          f"{summary['sqi']['frac_above_0.5']*100:.1f}% windows SQI>=0.5")
    print("=" * 60)


if __name__ == "__main__":
    main()
