#!/usr/bin/env python3
"""
Calibrate LSTM relative BP predictions to absolute mmHg using home cuff readings.

This is Model 2 from Radha et al. 2019: per-participant Ridge regression.
For each cuff reading at time T, all LSTM windows within ±15 min are taken
and their median used as the predicted relative BP at that moment.
Ridge regression then maps relative -> absolute for SBP and DBP separately.
MAP is derived as DBP + (SBP - DBP) / 3 rather than fit independently.

Inputs:
  --inference-dir  directory containing {subject_id}/bp_trend.csv and
                   {subject_id}/nocturnal_dip.json  (from run_inference.py)
  --cuff-csv       Qualtrics cuff readings CSV with at minimum:
                   subject_id, timestamp_utc, sbp_mmhg, dbp_mmhg
  --subject-id     participant ID, e.g. MK01

Outputs: out_dir/{subject_id}/
  bp_trend_calibrated.csv       bp_trend.csv + sbp_abs_mmhg, dbp_abs_mmhg, map_abs_mmhg
  nocturnal_dip_calibrated.json nocturnal dip recomputed on calibrated absolute SBP
  calibration.json              Ridge coefficients, LOO RMSE, n calibration points,
                                warning flag if fewer than 5 cuff readings

Usage:
  python calibrate_bp.py --subject-id MK01 --cuff-csv qualtrics_cuff_readings.csv

  python calibrate_bp.py \\
      --subject-id MK01 \\
      --cuff-csv   qualtrics_cuff_readings.csv \\
      --inference-dir "D:\\results" \\
      --out-dir       "D:\\calibrated"

Requirements: pip install numpy pandas scikit-learn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

MATCH_WINDOW_MS      = 15 * 60 * 1000   # ±15 minutes in milliseconds
RIDGE_ALPHA          = 1.0
MIN_CALIB_POINTS     = 3
WARN_CALIB_POINTS    = 5
DIPPER_THRESHOLD_PCT = 10.0


def parse_args():
    p = argparse.ArgumentParser(description='Ridge regression BP calibration (Model 2)')
    p.add_argument('--subject-id', required=True,
                   help='Participant ID, e.g. MK01')
    p.add_argument('--cuff-csv', required=True,
                   help='Qualtrics cuff readings CSV')
    p.add_argument('--inference-dir',
                   default=r'inference_results',
                   help='Directory with {subject_id}/bp_trend.csv and nocturnal_dip.json')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: same as --inference-dir)')
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cuff_readings(csv_path, subject_id):
    """
    Load and filter cuff readings for subject_id.
    Flexible column matching for Qualtrics export variations.
    Returns DataFrame with columns: timestamp_utc_ms (int64), sbp_mmhg, dbp_mmhg.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    sid_col = next((c for c in df.columns if 'subject' in c or c == 'id'), None)
    if sid_col is None:
        raise ValueError(f'Cannot find subject_id column in {csv_path}. '
                         f'Columns: {list(df.columns)}')
    df = df[df[sid_col].astype(str).str.strip() == str(subject_id)].copy()
    if len(df) == 0:
        raise ValueError(f'No rows for subject_id={subject_id} in {csv_path}')

    ts_col = next((c for c in df.columns
                   if 'timestamp' in c or 'time' in c or 'date' in c), None)
    if ts_col is None:
        raise ValueError(f'Cannot find timestamp column in {csv_path}')
    ts_parsed = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    ts_ms = (ts_parsed.values
             .astype('datetime64[ms]')
             .astype(np.int64))

    sbp_col = next((c for c in df.columns if 'sbp' in c), None)
    dbp_col = next((c for c in df.columns if 'dbp' in c), None)
    if sbp_col is None or dbp_col is None:
        raise ValueError(f'Cannot find sbp/dbp columns in {csv_path}. '
                         f'Columns: {list(df.columns)}')

    result = pd.DataFrame({
        'timestamp_utc_ms': ts_ms,
        'sbp_mmhg':         pd.to_numeric(df[sbp_col].values, errors='coerce'),
        'dbp_mmhg':         pd.to_numeric(df[dbp_col].values, errors='coerce'),
    }).dropna().reset_index(drop=True)
    result['timestamp_utc_ms'] = result['timestamp_utc_ms'].astype(np.int64)

    print(f'  Cuff readings loaded: {len(result)} for {subject_id}')
    return result


# ── Calibration matching and fitting ─────────────────────────────────────────

def match_cuff_to_predictions(cuff_df, pred_df):
    """
    For each cuff reading, find LSTM windows within ±15 min and take their
    median. Returns DataFrame with sbp_cuff, dbp_cuff, sbp_pred_med, dbp_pred_med.
    Cuff readings with no matching windows are skipped.
    """
    pred_ts  = pred_df['timestamp_utc_ms'].values.astype(np.int64)
    pred_sbp = pred_df['sbp_rel_mmhg'].values
    pred_dbp = pred_df['dbp_rel_mmhg'].values

    rows = []
    for _, row in cuff_df.iterrows():
        t    = int(row['timestamp_utc_ms'])
        mask = np.abs(pred_ts - t) <= MATCH_WINDOW_MS
        if mask.sum() == 0:
            ts_str = pd.to_datetime(t, unit='ms', utc=True).strftime('%Y-%m-%dT%H:%M:%SZ')
            print(f'  WARNING: no LSTM windows within ±15 min of cuff reading '
                  f'at {ts_str}. Skipping.')
            continue
        rows.append({
            'sbp_cuff':     float(row['sbp_mmhg']),
            'dbp_cuff':     float(row['dbp_mmhg']),
            'sbp_pred_med': float(np.median(pred_sbp[mask])),
            'dbp_pred_med': float(np.median(pred_dbp[mask])),
        })

    if not rows:
        return pd.DataFrame(columns=['sbp_cuff', 'dbp_cuff', 'sbp_pred_med', 'dbp_pred_med'])
    return pd.DataFrame(rows)


def fit_ridge_with_loo(X, y):
    """
    Fit Ridge(alpha=RIDGE_ALPHA) on X (n,1) -> y (n,).
    Returns (alpha_coef, beta_intercept, loo_rmse).
    LOO RMSE is NaN when n < 2.
    """
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y)
    alpha_coef = float(model.coef_[0])
    beta       = float(model.intercept_)

    if len(X) < 2:
        return alpha_coef, beta, float('nan')

    errs = []
    for train_idx, test_idx in LeaveOneOut().split(X):
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X[train_idx], y[train_idx])
        errs.append((m.predict(X[test_idx])[0] - y[test_idx][0]) ** 2)
    return alpha_coef, beta, float(np.sqrt(np.mean(errs)))


# ── Nocturnal dip on calibrated absolute values ───────────────────────────────

def _parse_sleep_window(sleep_window_str):
    """'22:00-06:00' -> (22, 6)"""
    parts = sleep_window_str.split('-')
    return int(parts[0].split(':')[0]), int(parts[1].split(':')[0])


def _to_local(ts_ms_arr, tz_str):
    utc_idx = pd.to_datetime(ts_ms_arr, unit='ms', utc=True)
    try:
        return utc_idx.tz_convert(tz_str)
    except Exception as e:
        print(f'  WARNING: timezone "{tz_str}" not recognised ({e}). Using UTC.')
        return utc_idx


def _is_sleep(local_dt_series, sleep_start_hr, sleep_end_hr):
    h = local_dt_series.dt.hour
    if sleep_start_hr > sleep_end_hr:
        return (h >= sleep_start_hr) | (h < sleep_end_hr)
    return (h >= sleep_start_hr) & (h < sleep_end_hr)


def compute_nocturnal_dip_abs(df_in, sleep_start_hr, sleep_end_hr, tz_str):
    """
    Recompute nocturnal dip using calibrated absolute SBP.
    df_in must have: timestamp_utc_ms, sbp_abs_mmhg.
    Mirrors run_inference.compute_nocturnal_dip; returns (nights list, summary dict).
    """
    df = df_in[['timestamp_utc_ms', 'sbp_abs_mmhg']].copy()
    local_dt = _to_local(df['timestamp_utc_ms'].values, tz_str)
    df['datetime_local'] = local_dt
    df['_sleep']         = _is_sleep(df['datetime_local'], sleep_start_hr, sleep_end_hr)
    df['_night_date']    = (
        (df['datetime_local'] - pd.Timedelta(hours=sleep_end_hr))
        .dt.date
    )

    nights = []
    for night_date, grp in df.groupby('_night_date', sort=True):
        sleep_grp = grp[grp['_sleep']]
        wake_grp  = grp[~grp['_sleep']]
        if len(sleep_grp) < 2 or len(wake_grp) < 2:
            continue

        mean_wake_sbp  = float(wake_grp['sbp_abs_mmhg'].mean())
        mean_sleep_sbp = float(sleep_grp['sbp_abs_mmhg'].mean())
        dip_mmhg       = mean_wake_sbp - mean_sleep_sbp
        dip_pct = (dip_mmhg / abs(mean_wake_sbp) * 100.0
                   if abs(mean_wake_sbp) > 1e-9 else float('nan'))

        nights.append({
            'date':            str(night_date),
            'n_sleep_windows': int(len(sleep_grp)),
            'n_wake_windows':  int(len(wake_grp)),
            'mean_wake_sbp':   round(mean_wake_sbp,  3),
            'mean_sleep_sbp':  round(mean_sleep_sbp, 3),
            'dip_mmhg':        round(dip_mmhg, 3),
            'dip_pct':         round(dip_pct, 2) if not np.isnan(dip_pct) else None,
            'is_dipper':       (dip_pct >= DIPPER_THRESHOLD_PCT
                                if not np.isnan(dip_pct) else None),
        })

    if not nights:
        return nights, {}

    valid_dips   = [n['dip_mmhg'] for n in nights]
    valid_dip_pc = [n['dip_pct']  for n in nights if n['dip_pct'] is not None]
    dipper_flags = [n['is_dipper'] for n in nights if n['is_dipper'] is not None]
    summary = {
        'n_nights':          len(nights),
        'mean_dip_mmhg':     round(float(np.mean(valid_dips)),   2),
        'std_dip_mmhg':      round(float(np.std(valid_dips)),    2),
        'mean_dip_pct':      round(float(np.mean(valid_dip_pc)), 2) if valid_dip_pc else None,
        'pct_dipper_nights': (round(100.0 * sum(dipper_flags) / len(dipper_flags), 1)
                              if dipper_flags else None),
    }
    return nights, summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    subj    = args.subject_id
    inf_dir = Path(args.inference_dir) / subj
    out_dir = (Path(args.out_dir) / subj) if args.out_dir else inf_dir

    print('=' * 60)
    print(f'BP Calibration (Model 2) — {subj}')
    print('=' * 60)

    # ── Load LSTM predictions ──
    trend_path = inf_dir / 'bp_trend.csv'
    if not trend_path.exists():
        print(f'ERROR: {trend_path} not found. Run run_inference.py first.')
        sys.exit(1)
    pred_df = pd.read_csv(trend_path)
    pred_df['timestamp_utc_ms'] = pred_df['timestamp_utc_ms'].astype(np.int64)
    print(f'\nLoaded bp_trend.csv: {len(pred_df)} windows')

    # ── Load nocturnal_dip.json for timezone / sleep window metadata ──
    dip_path = inf_dir / 'nocturnal_dip.json'
    if not dip_path.exists():
        print(f'ERROR: {dip_path} not found. Run run_inference.py first.')
        sys.exit(1)
    with open(dip_path) as f:
        dip_meta = json.load(f)
    tz_str = dip_meta.get('timezone', 'UTC')
    sleep_start, sleep_end = _parse_sleep_window(
        dip_meta.get('sleep_window', '22:00-06:00'))

    # ── Load cuff readings ──
    print(f'\nLoading cuff readings from {args.cuff_csv} ...')
    try:
        cuff_df = load_cuff_readings(args.cuff_csv, subj)
    except ValueError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    # ── Match cuff readings to LSTM windows ──
    print('Matching cuff readings to LSTM windows (±15 min) ...')
    calib_df = match_cuff_to_predictions(cuff_df, pred_df)
    n_calib  = len(calib_df)
    print(f'  Calibration pairs available: {n_calib}')

    low_calib_warning = n_calib < WARN_CALIB_POINTS
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_calib < MIN_CALIB_POINTS:
        print(f'\nWARNING: only {n_calib} matched calibration point(s) '
              f'(minimum {MIN_CALIB_POINTS}). Skipping calibration.')
        cal_path = out_dir / 'calibration.json'
        with open(cal_path, 'w') as f:
            json.dump({
                'subject_id':           subj,
                'n_calibration_points': n_calib,
                'calibration_skipped':  True,
                'reason':               f'Fewer than {MIN_CALIB_POINTS} matched cuff readings',
                'low_calib_warning':    True,
            }, f, indent=2)
        print(f'Saved: {cal_path}')
        sys.exit(0)

    # ── Fit Ridge regression ──
    print(f'\nFitting Ridge regression (alpha={RIDGE_ALPHA}) ...')
    X_sbp = calib_df['sbp_pred_med'].values.reshape(-1, 1)
    y_sbp = calib_df['sbp_cuff'].values
    X_dbp = calib_df['dbp_pred_med'].values.reshape(-1, 1)
    y_dbp = calib_df['dbp_cuff'].values

    sbp_a, sbp_b, sbp_loo = fit_ridge_with_loo(X_sbp, y_sbp)
    dbp_a, dbp_b, dbp_loo = fit_ridge_with_loo(X_dbp, y_dbp)

    print(f'  SBP: alpha={sbp_a:.4f}  beta={sbp_b:.4f}  LOO-RMSE={sbp_loo:.2f} mmHg')
    print(f'  DBP: alpha={dbp_a:.4f}  beta={dbp_b:.4f}  LOO-RMSE={dbp_loo:.2f} mmHg')

    # ── Apply calibration ──
    pred_df['sbp_abs_mmhg'] = np.round(sbp_a * pred_df['sbp_rel_mmhg'] + sbp_b, 3)
    pred_df['dbp_abs_mmhg'] = np.round(dbp_a * pred_df['dbp_rel_mmhg'] + dbp_b, 3)
    pred_df['map_abs_mmhg'] = np.round(
        pred_df['dbp_abs_mmhg'] + (pred_df['sbp_abs_mmhg'] - pred_df['dbp_abs_mmhg']) / 3.0, 3)

    # ── Recompute nocturnal dip on calibrated absolute SBP ──
    print('\nComputing calibrated nocturnal dip ...')
    nights_cal, dip_summary_cal = compute_nocturnal_dip_abs(
        pred_df, sleep_start, sleep_end, tz_str)

    # ── Save outputs ──
    cal_csv_path = out_dir / 'bp_trend_calibrated.csv'
    pred_df.to_csv(cal_csv_path, index=False)
    print(f'\nSaved: {cal_csv_path}  ({cal_csv_path.stat().st_size / 1e3:.0f} KB)')

    dip_cal_path = out_dir / 'nocturnal_dip_calibrated.json'
    with open(dip_cal_path, 'w') as f:
        json.dump({
            'subject_id':           subj,
            'timezone':             tz_str,
            'sleep_window':         dip_meta.get('sleep_window',
                                                  f'{sleep_start:02d}:00-{sleep_end:02d}:00'),
            'dipper_threshold_pct': DIPPER_THRESHOLD_PCT,
            'calibrated':           True,
            'nights':               nights_cal,
            'summary':              dip_summary_cal,
        }, f, indent=2)
    print(f'Saved: {dip_cal_path}')

    cal_json_path = out_dir / 'calibration.json'
    with open(cal_json_path, 'w') as f:
        json.dump({
            'subject_id':           subj,
            'n_calibration_points': n_calib,
            'low_calib_warning':    low_calib_warning,
            'ridge_alpha':          RIDGE_ALPHA,
            'sbp': {'alpha': sbp_a, 'beta': sbp_b, 'loo_rmse': sbp_loo},
            'dbp': {'alpha': dbp_a, 'beta': dbp_b, 'loo_rmse': dbp_loo},
            'map_note': 'MAP = DBP + (SBP - DBP) / 3, not a separate Ridge fit',
        }, f, indent=2)
    print(f'Saved: {cal_json_path}')

    # ── Summary ──
    print(f'\n{"=" * 60}')
    print(f'Calibration summary — {subj}')
    print(f'  Calibration points used : {n_calib}')
    if low_calib_warning:
        print(f'  WARNING: fewer than {WARN_CALIB_POINTS} cuff readings — '
              f'calibration may be unreliable')
    print(f'  LOO RMSE  SBP           : {sbp_loo:.2f} mmHg')
    print(f'  LOO RMSE  DBP           : {dbp_loo:.2f} mmHg')
    if dip_summary_cal:
        print(f'\n  Calibrated nocturnal SBP dip:')
        print(f'    Nights with data : {dip_summary_cal["n_nights"]}')
        if dip_summary_cal.get('mean_dip_mmhg') is not None:
            print(f'    Mean dip         : {dip_summary_cal["mean_dip_mmhg"]:.2f} mmHg')
        if dip_summary_cal.get('mean_dip_pct') is not None:
            print(f'    Mean dip %       : {dip_summary_cal["mean_dip_pct"]:.2f}%')
        if dip_summary_cal.get('pct_dipper_nights') is not None:
            print(f'    Dipper nights    : {dip_summary_cal["pct_dipper_nights"]:.1f}%')
    else:
        print('\n  WARNING: no nights with sufficient sleep+wake windows for dip calculation.')
    print('=' * 60)


if __name__ == '__main__':
    main()
