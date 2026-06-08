#!/usr/bin/env python3
"""
Run LSTM inference on preprocessed LEAP participant features.

Inputs:
  --features-npz  output of preprocess_leap.py ({subject_id}.npz)
  --model-dir     directory with best_model.keras (from train_lstm.py)

Outputs: out_dir/{subject_id}/
  bp_trend.csv          per-window BP predictions with timestamps
  nocturnal_dip.json    per-night nocturnal dip analysis
  summary.json          overall summary statistics

IMPORTANT — output units:
  All BP values are RELATIVE mmHg (within-person deviations from personal
  baseline). Absolute calibration requires Model 2 (ridge regression on
  home cuff readings from Qualtrics weekly survey).
  The nocturnal dip is a difference (wake - sleep), so the baseline offset
  cancels: dip values are interpretable even without Model 2 calibration.

Nocturnal dip definition:
  Sleep window : sleep_start (default 22:00) to sleep_end (default 06:00)
  Wake window  : sleep_end to sleep_start on the same calendar day
  Dip (mmHg)   : mean_wake_SBP - mean_sleep_SBP
  Dip %        : dip / |mean_wake_SBP| * 100  (when mean_wake_SBP != 0)
  Dipper       : dip_pct >= 10 %  (standard clinical threshold)

Usage:
  python run_inference.py \\
      --features-npz participant_features/MK01.npz \\
      --model-dir    lstm_model/

  # With temperature-based sleep windows from detect_sleep.py:
  python run_inference.py \\
      --features-npz participant_features/MK01.npz \\
      --model-dir    lstm_model/ \\
      --timezone     US/Pacific \\
      --sleep-windows-json inference_results/MK01/sleep_windows.json

  # Custom sleep hours or timezone:
  python run_inference.py \\
      --features-npz participant_features/MK01.npz \\
      --model-dir    lstm_model/ \\
      --timezone     US/Pacific \\
      --sleep-start  23 \\
      --sleep-end    7

Requirements: pip install tensorflow numpy scipy pandas
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# ── Constants (must match train_lstm.py) ──────────────────────────────────────
SEQ_LEN    = 20    # windows per LSTM sequence
SEQ_STEP   = 1     # stride for inference (maximum coverage)
N_FEATURES = 13
N_OUTPUTS  = 3     # SBP, DBP, MAP

DIPPER_THRESHOLD_PCT = 10.0   # standard clinical threshold

# ── Minimal stub for loading model with custom loss ───────────────────────────
class WeightedMSE(tf.keras.losses.Loss):
    def __init__(self, train_mean_sbp=0.0, **kwargs):
        super().__init__(**kwargs)
        self.train_mean_sbp = float(train_mean_sbp)

    def call(self, y_true, y_pred):
        sbp_true = tf.cast(y_true[..., 0], tf.float32)
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.cast(y_pred, tf.float32)
        weight   = tf.abs(sbp_true - self.train_mean_sbp) + 1.0
        mse      = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return tf.reduce_mean(mse * weight)

    def get_config(self):
        cfg = super().get_config()
        cfg['train_mean_sbp'] = self.train_mean_sbp
        return cfg


def parse_args():
    p = argparse.ArgumentParser(description='LSTM BP inference on LEAP data')
    p.add_argument('--features-npz', required=True,
                   help='Path to subject .npz from preprocess_leap.py')
    p.add_argument('--model-dir',
                   default=r'lstm_model',
                   help='Directory containing best_model.keras')
    p.add_argument('--out-dir',
                   default=r'inference_results',
                   help='Output directory (subject subfolder created automatically)')
    p.add_argument('--seq-len', type=int, default=SEQ_LEN,
                   help='LSTM sequence length (must match training)')
    p.add_argument('--batch-size', type=int, default=512,
                   help='Batch size for model.predict (larger = faster)')
    p.add_argument('--timezone', default='UTC',
                   help='Timezone string for nocturnal dip grouping '
                        '(e.g. US/Eastern, US/Pacific). Default: UTC.')
    p.add_argument('--sleep-start', type=int, default=22,
                   help='Hour (0-23) when sleep window begins. Default: 22')
    p.add_argument('--sleep-end', type=int, default=6,
                   help='Hour (0-23) when sleep window ends. Default: 6')
    p.add_argument('--sleep-windows-json', default=None,
                   help='Path to sleep_windows.json from detect_sleep.py. '
                        'When provided, uses per-night detected windows instead of '
                        '--sleep-start/--sleep-end. The fixed-clock args remain as '
                        'fallback within detect_sleep.py.')
    return p.parse_args()


# ── Sequence construction ─────────────────────────────────────────────────────

def build_sequences(features, seq_len):
    """
    Build sliding-window sequences from feature matrix.
    features : (W, 13)
    Returns  : X (N, seq_len, 13), seq_end_idx (N,) — last window index per seq
    """
    W = len(features)
    n_seqs = W - seq_len + 1
    if n_seqs <= 0:
        return None, None
    X = np.lib.stride_tricks.as_strided(
        features,
        shape=(n_seqs, seq_len, N_FEATURES),
        strides=(features.strides[0], features.strides[0], features.strides[1]),
    ).copy().astype(np.float32)
    seq_end_idx = np.arange(seq_len - 1, W, dtype=np.int64)[:n_seqs]
    return X, seq_end_idx


# ── Nocturnal dip ─────────────────────────────────────────────────────────────

def _to_local(ts_ms_arr, tz_str):
    """Convert Unix ms array to pandas DatetimIndex in local timezone."""
    utc_idx = pd.to_datetime(ts_ms_arr, unit='ms', utc=True)
    try:
        return utc_idx.tz_convert(tz_str)
    except Exception as e:
        print(f"  WARNING: timezone '{tz_str}' not recognised ({e}). Using UTC.")
        return utc_idx


def is_sleep(local_dt_series, sleep_start_hr, sleep_end_hr):
    """
    Return boolean Series: True if the datetime is within the sleep window.
    Handles wrap-around (e.g. 22:00-06:00 spans midnight).
    """
    h = local_dt_series.dt.hour
    if sleep_start_hr > sleep_end_hr:   # wraps midnight
        return (h >= sleep_start_hr) | (h < sleep_end_hr)
    else:
        return (h >= sleep_start_hr) & (h < sleep_end_hr)


def _is_sleep_from_windows(ts_ms_arr, nights):
    """
    Return boolean array: True if timestamp falls inside any per-night window.
    nights: list of dicts with sleep_start_utc_ms / sleep_end_utc_ms.
    """
    result = np.zeros(len(ts_ms_arr), dtype=bool)
    for night in nights:
        result |= (ts_ms_arr >= night['sleep_start_utc_ms']) & \
                  (ts_ms_arr <= night['sleep_end_utc_ms'])
    return result


def compute_nocturnal_dip(df, sleep_start_hr, sleep_end_hr, tz_str, sleep_windows=None):
    """
    df must have columns: timestamp_utc_ms, datetime_local (DatetimeTZDtype), sbp_rel.
    sleep_windows: list of night dicts from sleep_windows.json (optional).
      When provided, uses UTC-ms window boundaries and per-night date labels
      instead of the fixed hour-based logic.
    Returns (nights list, summary dict).
    """
    df = df.copy()

    if sleep_windows is not None:
        ts_ms        = df['timestamp_utc_ms'].values.astype(np.float64)
        is_sleep_arr = np.zeros(len(ts_ms), dtype=bool)
        night_dates  = np.full(len(ts_ms), None, dtype=object)
        centers      = []
        date_strs    = []

        for night in sleep_windows:
            start_ms = night['sleep_start_utc_ms']
            end_ms   = night['sleep_end_utc_ms']
            mask     = (ts_ms >= start_ms) & (ts_ms <= end_ms)
            is_sleep_arr[mask] = True
            night_dates[mask]  = night['date']
            centers.append((start_ms + end_ms) / 2.0)
            date_strs.append(night['date'])

        # Assign wake rows to their nearest night (by window centre)
        wake_mask = ~is_sleep_arr
        if wake_mask.any() and centers:
            centers_arr = np.array(centers)
            wake_idxs   = np.where(wake_mask)[0]
            diffs        = np.abs(ts_ms[wake_idxs, np.newaxis] - centers_arr[np.newaxis, :])
            nearest      = np.argmin(diffs, axis=1)
            night_dates[wake_idxs] = np.array(date_strs)[nearest]

        df['_sleep']      = is_sleep_arr
        df['_night_date'] = night_dates
        df = df[df['_night_date'].notna()]
    else:
        df['_sleep'] = is_sleep(df['datetime_local'], sleep_start_hr, sleep_end_hr)
        # Shift time back by sleep_end hours so the entire night groups under one date
        shift_h = sleep_end_hr
        df['_night_date'] = (
            (df['datetime_local'] - pd.Timedelta(hours=shift_h))
            .dt.date
        )

    nights = []
    for night_date, grp in df.groupby('_night_date', sort=True):
        sleep_grp = grp[grp['_sleep']]
        wake_grp  = grp[~grp['_sleep']]

        if len(sleep_grp) < 2 or len(wake_grp) < 2:
            continue

        mean_wake_sbp  = float(wake_grp['sbp_rel'].mean())
        mean_sleep_sbp = float(sleep_grp['sbp_rel'].mean())
        dip_mmhg       = mean_wake_sbp - mean_sleep_sbp

        if abs(mean_wake_sbp) > 1e-9:
            dip_pct = dip_mmhg / abs(mean_wake_sbp) * 100.0
        else:
            dip_pct = float('nan')

        nights.append({
            'date':             str(night_date),
            'n_sleep_windows':  int(len(sleep_grp)),
            'n_wake_windows':   int(len(wake_grp)),
            'mean_wake_sbp':    round(mean_wake_sbp,  3),
            'mean_sleep_sbp':   round(mean_sleep_sbp, 3),
            'dip_mmhg':         round(dip_mmhg, 3),
            'dip_pct':          round(dip_pct, 2) if not np.isnan(dip_pct) else None,
            'is_dipper':        (dip_pct >= DIPPER_THRESHOLD_PCT
                                 if not np.isnan(dip_pct) else None),
        })

    if not nights:
        summary = {}
    else:
        valid_dips   = [n['dip_mmhg'] for n in nights]
        valid_dip_pc = [n['dip_pct']  for n in nights if n['dip_pct'] is not None]
        dipper_flags = [n['is_dipper'] for n in nights if n['is_dipper'] is not None]
        summary = {
            'n_nights':             len(nights),
            'mean_dip_mmhg':        round(float(np.mean(valid_dips)),   2),
            'std_dip_mmhg':         round(float(np.std(valid_dips)),    2),
            'mean_dip_pct':         round(float(np.mean(valid_dip_pc)), 2) if valid_dip_pc else None,
            'pct_dipper_nights':    round(100.0 * sum(dipper_flags) / len(dipper_flags), 1)
                                    if dipper_flags else None,
        }

    return nights, summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    npz_path   = Path(args.features_npz)
    model_dir  = Path(args.model_dir)
    seq_len    = args.seq_len

    # ── Load participant data ──
    print('=' * 60)
    if not npz_path.exists():
        print(f'ERROR: {npz_path} not found. Run preprocess_leap.py first.')
        sys.exit(1)

    d = np.load(npz_path, allow_pickle=True)
    features   = d['features'].astype(np.float32)      # already normalised
    ts_ms      = d['timestamps_utc'].astype(np.float64)
    subject_id = str(d['subject_id'])

    print(f'Subject:  {subject_id}')
    print(f'Windows:  {len(features)}')
    print(f'Features: {features.shape[1]}')

    # ── Load sleep windows (optional) ──
    sleep_windows_data = None
    if args.sleep_windows_json:
        swj = Path(args.sleep_windows_json)
        if not swj.exists():
            print(f'ERROR: --sleep-windows-json {swj} not found.')
            sys.exit(1)
        with open(swj) as fj:
            sleep_windows_data = json.load(fj)
        n_nights = len(sleep_windows_data.get('nights', []))
        print(f'\nSleep windows: {swj}  ({n_nights} night(s), '
              f'timezone={sleep_windows_data.get("timezone", "?")})')
        for night in sleep_windows_data.get('nights', []):
            print(f'  {night["date"]}  {night["sleep_start_local"]} -> '
                  f'{night["sleep_end_local"]}  [{night["source"]}]')

    # ── Load model ──
    model_path = model_dir / 'best_model.keras'
    if not model_path.exists():
        print(f'ERROR: {model_path} not found. Run train_lstm.py (or sbatch slurm_train.sh) first.')
        sys.exit(1)

    print(f'\nLoading model from {model_path} ...')
    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={'WeightedMSE': WeightedMSE},
    )
    print('  Model loaded.')
    model.summary(print_fn=lambda s: None)   # suppress verbose output

    # ── Build sequences ──
    print(f'\nBuilding sequences (len={seq_len}, step=1) ...')
    X, seq_end_idx = build_sequences(features, seq_len)
    if X is None:
        print(f'ERROR: Not enough windows ({len(features)}) for seq_len={seq_len}. '
              f'Need at least {seq_len}.')
        sys.exit(1)
    print(f'  Sequences: {len(X)}')

    # ── Predict ──
    print(f'Running inference (batch_size={args.batch_size}) ...')
    preds_all = model.predict(X, batch_size=args.batch_size, verbose=1)
    # preds_all: (N_seqs, seq_len, 3)
    # Use last-step prediction to avoid double-counting overlapping windows
    preds_last = preds_all[:, -1, :]          # (N_seqs, 3)
    pred_sbp   = preds_last[:, 0].astype(np.float64)
    pred_dbp   = preds_last[:, 1].astype(np.float64)
    pred_map   = preds_last[:, 2].astype(np.float64)

    # Timestamps for each prediction = timestamp of the last window in the sequence
    pred_ts_ms = ts_ms[seq_end_idx]

    print(f'  Predictions: {len(pred_sbp)}')
    print(f'  SBP  mean={pred_sbp.mean():.2f}  std={pred_sbp.std():.2f} mmHg (relative)')
    print(f'  DBP  mean={pred_dbp.mean():.2f}  std={pred_dbp.std():.2f} mmHg (relative)')
    print(f'  MAP  mean={pred_map.mean():.2f}  std={pred_map.std():.2f} mmHg (relative)')

    # ── Determine is_sleep and sleep detection method ──
    local_dt = _to_local(pred_ts_ms, args.timezone)

    nights_list = (sleep_windows_data or {}).get('nights') or []
    if nights_list:
        is_sleep_col        = _is_sleep_from_windows(pred_ts_ms, nights_list)
        sleep_detection_str = 'temperature'
        sleep_window_str    = (f'temperature_detected '
                               f'(fallback {args.sleep_start:02d}:00-{args.sleep_end:02d}:00)')
    else:
        is_sleep_col        = is_sleep(
            local_dt.to_series().reset_index(drop=True),
            args.sleep_start, args.sleep_end
        ).values
        sleep_detection_str = 'fixed_clock'
        sleep_window_str    = f'{args.sleep_start:02d}:00-{args.sleep_end:02d}:00'

    # ── Build output DataFrame ──
    df = pd.DataFrame({
        'timestamp_utc_ms': pred_ts_ms.astype(np.int64),
        'datetime_utc':     pd.to_datetime(pred_ts_ms, unit='ms', utc=True)
                            .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'datetime_local':   local_dt,
        'sbp_rel_mmhg':     np.round(pred_sbp, 3),
        'dbp_rel_mmhg':     np.round(pred_dbp, 3),
        'map_rel_mmhg':     np.round(pred_map, 3),
        'is_sleep':         is_sleep_col,
    })

    # ── Nocturnal dip ──
    print('\nComputing nocturnal dip ...')
    dip_df = df[['timestamp_utc_ms', 'datetime_local', 'sbp_rel_mmhg']].copy()
    dip_df.columns = ['timestamp_utc_ms', 'datetime_local', 'sbp_rel']
    nights, dip_summary = compute_nocturnal_dip(
        dip_df, args.sleep_start, args.sleep_end, args.timezone,
        sleep_windows=nights_list if nights_list else None,
    )

    if nights:
        mean_dip   = dip_summary['mean_dip_mmhg']
        pct_dipper = dip_summary.get('pct_dipper_nights')
        print(f'  Sleep detection:             {sleep_detection_str}')
        print(f'  Nights with sufficient data: {dip_summary["n_nights"]}')
        print(f'  Mean nocturnal SBP dip:      {mean_dip:.2f} mmHg')
        if pct_dipper is not None:
            print(f'  Dipper nights (>=10% dip):   {pct_dipper:.1f}%')
    else:
        print('  WARNING: No nights with sufficient sleep+wake windows. '
              'Check --sleep-start/--sleep-end, --sleep-windows-json, and timezone.')

    # ── Save outputs ──
    out_dir = Path(args.out_dir) / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # bp_trend.csv — drop datetime_local (tz-aware) to keep CSV portable
    csv_df = df.drop(columns=['datetime_local'])
    csv_path = out_dir / 'bp_trend.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}  ({csv_path.stat().st_size / 1e3:.0f} KB)')

    # nocturnal_dip.json
    dip_out = {
        'subject_id':           subject_id,
        'timezone':             args.timezone,
        'sleep_window':         sleep_window_str,
        'sleep_detection':      sleep_detection_str,
        'dipper_threshold_pct': DIPPER_THRESHOLD_PCT,
        'nights':               nights,
        'summary':              dip_summary,
    }
    dip_path = out_dir / 'nocturnal_dip.json'
    with open(dip_path, 'w') as f:
        json.dump(dip_out, f, indent=2)
    print(f'Saved: {dip_path}')

    # summary.json
    t_start = datetime.fromtimestamp(pred_ts_ms[0]  / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    t_end   = datetime.fromtimestamp(pred_ts_ms[-1] / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    n_days  = int(np.ceil((pred_ts_ms[-1] - pred_ts_ms[0]) / 86_400_000))

    summary = {
        'subject_id':         subject_id,
        'model':              str(model_path),
        'features_npz':       str(npz_path),
        'seq_len':            seq_len,
        'n_predictions':      int(len(pred_sbp)),
        'time_range_start':   t_start,
        'time_range_end':     t_end,
        'n_days_covered':     n_days,
        'sbp_rel': {
            'mean':  round(float(pred_sbp.mean()), 3),
            'std':   round(float(pred_sbp.std()),  3),
            'min':   round(float(pred_sbp.min()),  3),
            'max':   round(float(pred_sbp.max()),  3),
        },
        'dbp_rel': {
            'mean':  round(float(pred_dbp.mean()), 3),
            'std':   round(float(pred_dbp.std()),  3),
            'min':   round(float(pred_dbp.min()),  3),
            'max':   round(float(pred_dbp.max()),  3),
        },
        'map_rel': {
            'mean':  round(float(pred_map.mean()), 3),
            'std':   round(float(pred_map.std()),  3),
            'min':   round(float(pred_map.min()),  3),
            'max':   round(float(pred_map.max()),  3),
        },
        'nocturnal_dip': dip_summary,
        'note': (
            'BP values are RELATIVE mmHg (within-person deviations from '
            'personal baseline). Absolute calibration via Model 2 '
            '(ridge regression on Qualtrics cuff readings) required for '
            'absolute mmHg interpretation.'
        ),
    }
    sum_path = out_dir / 'summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {sum_path}')

    print(f'\n{"="*60}')
    print(f'Inference complete for {subject_id}')
    print(f'Output directory: {out_dir}')
    print(f'{"="*60}')
    print('\nNext step: Model 2 (ridge regression calibration).')
    print('  Align Qualtrics home cuff timestamps to bp_trend.csv windows,')
    print('  then run:  python calibrate_bp.py --subject-id', subject_id)


if __name__ == '__main__':
    main()
