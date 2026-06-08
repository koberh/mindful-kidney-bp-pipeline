#!/usr/bin/env python3
"""
Detect sleep onset and wake time from ActiGraph LEAP wrist skin temperature.

Wrist skin temperature rises at sleep onset (peripheral vasodilation) and
drops at wake.  The script smooths the signal, applies a per-night threshold,
finds the longest sustained sleep bout, and falls back to a fixed clock window
when no qualifying bout is found.

Output: {out_dir}/{subject_id}/sleep_windows.json

Usage:
  python detect_sleep.py \
      --temp-csv   "SUBJ01_Temperature.csv" \
      --subject-id SUBJ01 \
      --out-dir    inference_results/ \
      --timezone   US/Pacific

Requirements: numpy pandas scipy (already in bp_env — no new deps)
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SMOOTHING_SAMPLES = 150   # 10 min at 4-s sample rate
MIN_SLEEP_SAMPLES = 450   # 30 min at 4-s sample rate


def parse_args():
    p = argparse.ArgumentParser(description='Detect sleep from LEAP wrist temperature')
    p.add_argument('--temp-csv', required=True,
                   help='Path to LEAP temperature CSV')
    p.add_argument('--subject-id', required=True,
                   help='Subject identifier (used for output subfolder)')
    p.add_argument('--out-dir',
                   default=r'inference_results',
                   help='Output base directory (subject subfolder created automatically)')
    p.add_argument('--timezone', default='UTC',
                   help='Local timezone string (e.g. US/Pacific). Default: UTC.')
    p.add_argument('--fallback-sleep-start', type=int, default=22,
                   help='Hour (0-23) for fallback sleep onset when detection fails. Default: 22.')
    p.add_argument('--fallback-sleep-end', type=int, default=6,
                   help='Hour (0-23) for fallback wake time when detection fails. Default: 6.')
    return p.parse_args()


def load_temp_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    header_row = next(i for i, l in enumerate(lines) if l.strip().startswith('Timestamp'))
    df = pd.read_csv(csv_path, skiprows=header_row)
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['Timestamp', 'Temperature']).sort_values('Timestamp').reset_index(drop=True)
    return df


def find_sleep_bouts(labels, min_samples):
    """Return list of (start_idx, end_idx_exclusive, duration) for runs >= min_samples."""
    bouts = []
    n = len(labels)
    in_bout = False
    start_i = 0
    for i in range(n):
        if labels[i] and not in_bout:
            in_bout = True
            start_i = i
        elif not labels[i] and in_bout:
            in_bout = False
            dur = i - start_i
            if dur >= min_samples:
                bouts.append((start_i, i, dur))
    if in_bout:
        dur = n - start_i
        if dur >= min_samples:
            bouts.append((start_i, n, dur))
    return bouts


def ts_to_utc_ms(ts):
    """pd.Timestamp (tz-aware UTC) → integer UTC milliseconds."""
    return int(ts.value // 1_000_000)


def main():
    args = parse_args()
    csv_path = Path(args.temp_csv)
    if not csv_path.exists():
        print(f'ERROR: {csv_path} not found.')
        sys.exit(1)

    out_dir = Path(args.out_dir) / args.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Subject:  {args.subject_id}')
    print(f'Timezone: {args.timezone}')

    df = load_temp_csv(str(csv_path))
    print(f'Loaded {len(df):,} temperature samples '
          f'({df["Timestamp"].iloc[0]} to {df["Timestamp"].iloc[-1]} UTC)')

    # Timestamps in LEAP files are UTC — localize then convert to local
    df['ts_utc'] = df['Timestamp'].dt.tz_localize('UTC')
    try:
        df['ts_local'] = df['ts_utc'].dt.tz_convert(args.timezone)
    except Exception as e:
        print(f'  WARNING: timezone "{args.timezone}" not recognised ({e}). Using UTC.')
        df['ts_local'] = df['ts_utc']

    # 10-minute rolling median smoothing
    df['smoothed'] = df['Temperature'].rolling(
        window=SMOOTHING_SAMPLES, center=True, min_periods=50
    ).median()

    local_dates = sorted(df['ts_local'].dt.date.unique())
    print(f'Calendar dates (local): {[str(d) for d in local_dates]}')
    print()

    nights = []
    fallback_count = 0

    for d in local_dates:
        next_d = d + timedelta(days=1)

        # Plausible sleep window 19:00 → next-day 11:00 (spans midnight)
        try:
            win_start = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                     hour=19, tz=args.timezone)
            win_end   = pd.Timestamp(year=next_d.year, month=next_d.month, day=next_d.day,
                                     hour=11, tz=args.timezone)
        except Exception:
            continue

        mask     = (df['ts_local'] >= win_start) & (df['ts_local'] < win_end)
        night_df = df[mask].copy().reset_index(drop=True)

        if len(night_df) < MIN_SLEEP_SAMPLES:
            continue   # not enough data to analyse this night

        smoothed = night_df['smoothed'].values
        valid    = ~np.isnan(smoothed)

        if valid.sum() < MIN_SLEEP_SAMPLES:
            continue

        q25 = np.percentile(smoothed[valid], 25)
        q75 = np.percentile(smoothed[valid], 75)
        iqr = q75 - q25
        threshold = q25 + 0.3 * iqr

        sleep_labels = smoothed > threshold
        sleep_labels[~valid] = False

        bouts = find_sleep_bouts(sleep_labels, MIN_SLEEP_SAMPLES)

        if bouts:
            start_i, end_i, _ = max(bouts, key=lambda x: x[2])

            onset_ts = night_df['ts_utc'].iloc[start_i]
            wake_ts  = night_df['ts_utc'].iloc[end_i - 1]

            start_ms = ts_to_utc_ms(onset_ts)
            end_ms   = ts_to_utc_ms(wake_ts)
            dur_h    = (end_ms - start_ms) / 3_600_000

            onset_local = onset_ts.tz_convert(args.timezone)
            wake_local  = wake_ts.tz_convert(args.timezone)

            print(f'  Night {d}:  onset {onset_local.isoformat()}')
            print(f'             wake  {wake_local.isoformat()}')
            print(f'             duration {dur_h:.1f} h  [temperature]')

            nights.append({
                'date':               str(d),
                'sleep_start_utc_ms': start_ms,
                'sleep_end_utc_ms':   end_ms,
                'sleep_start_local':  onset_local.isoformat(),
                'sleep_end_local':    wake_local.isoformat(),
                'duration_hours':     round(dur_h, 2),
                'source':             'temperature',
            })
        else:
            # Fall back to fixed clock window for this night
            try:
                fb_start = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                        hour=args.fallback_sleep_start, tz=args.timezone)
                fb_end   = pd.Timestamp(year=next_d.year, month=next_d.month, day=next_d.day,
                                        hour=args.fallback_sleep_end, tz=args.timezone)
            except Exception:
                continue

            start_ms = ts_to_utc_ms(fb_start)
            end_ms   = ts_to_utc_ms(fb_end)
            dur_h    = (end_ms - start_ms) / 3_600_000

            print(f'  Night {d}:  no qualifying bout — fallback clock '
                  f'{args.fallback_sleep_start:02d}:00–{args.fallback_sleep_end:02d}:00  [fallback]')
            fallback_count += 1

            nights.append({
                'date':               str(d),
                'sleep_start_utc_ms': start_ms,
                'sleep_end_utc_ms':   end_ms,
                'sleep_start_local':  fb_start.isoformat(),
                'sleep_end_local':    fb_end.isoformat(),
                'duration_hours':     round(dur_h, 2),
                'source':             'fallback_clock',
            })

    n_temp = len(nights) - fallback_count
    print()
    print(f'Nights processed: {len(nights)}  '
          f'(temperature: {n_temp},  fallback clock: {fallback_count})')

    out = {
        'subject_id':     args.subject_id,
        'timezone':       args.timezone,
        'generated_from': 'temperature',
        'nights':         nights,
    }
    out_path = out_dir / 'sleep_windows.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {out_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
