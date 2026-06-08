#!/usr/bin/env python3
"""
Preprocess ActiGraph LEAP raw PPG CSV into the same 13-feature vectors
used by the MIMIC-trained LSTM, so the model can run inference on study data.

Input (from CentrePoint Data Access Files endpoint):
  ppg_csv    -- raw ppg-green CSV (25 Hz or 100 Hz)
               Columns: Ambient, Channel, Green, MonitorSerial,
                        Sensor, StudyId, SubjectId, Timestamp
  ibi_csv    -- (optional) CentrePoint IBI export
               Columns: INTERBEAT_INTERVAL, Timestamp (or similar)
               If provided, HRV features are computed from IBI instead of
               PPG-detected peaks (more accurate for RMSSD/SDNN/pNN50).

Output: participant_features/
  {subject_id}.npz
      features        float32 (W, 13)  -- same order as MIMIC model
      timestamps_utc  float64 (W,)     -- Unix ms, centre of each window
      feature_names   str array (13,)

  feature_names order:
    mean_rr, sdnn, rmssd, pnn50, mean_hr, lf_hf,
    pulse_amp, rise_time_ms, fall_time_ms, pulse_width_ms,
    auc, notch_depth, notch_time_norm

Processing steps:
  1. Load CSV, separate channels, flag ADC saturation as NaN
  2. Reconstruct uniform time axis from first valid timestamp + sample index
  3. Ambient-subtract Green channel (removes slow interference)
  4. Bandpass filter 0.5-4 Hz (matches MIMIC training)
  5. 30-second windows, 15-second step (matches MIMIC training)
  6. Per window: extract HRV (from IBI or PPG peaks) + PPG morphology
  7. Apply LSTM normalization (feat_mean.npy / feat_std.npy from training)
  8. Save features for LSTM inference

Usage:
  python preprocess_leap.py \\
      --ppg-csv  "downloads/25 hz.csv" \\
      --subject-id MK01 \\
      --out-dir   participant_features/

  # With IBI for better HRV (preferred when CentrePoint export available):
  python preprocess_leap.py \\
      --ppg-csv  "downloads/25 hz.csv" \\
      --ibi-csv  "downloads/MK01_ibi.csv" \\
      --subject-id MK01 \\
      --out-dir   participant_features/

Requirements: pip install numpy scipy pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Config ────────────────────────────────────────────────────────────────────
FS_25   = 25     # ppg-green 25 Hz stream
FS_100  = 100    # ppg-green-100-hz stream

SATURATION_VAL = 2_096_900   # ADC clipping sentinel (|Green| >= this -> NaN)
CHANNEL_USE    = 1            # which LED channel to use (0 or 1)
                              # Channel 1 is typically the primary PPG signal

BP_BAND     = (0.5, 4.0)     # bandpass Hz — matches MIMIC training
WINDOW_SEC  = 30             # window length (seconds)
STEP_SEC    = 15             # step between windows (50% overlap)

# PPG peak detection
PEAK_DIST_PPG = None         # set dynamically per FS (0.4 s minimum)
PEAK_PROM_PPG = 0.05         # 5% of signal range (normalized)

# HRV
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
MIN_RR_LF = 12

# Window quality gates
MAX_NAN_PPG = 0.10
MIN_BEATS   = 5

FEATURE_NAMES = [
    'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr', 'lf_hf',
    'pulse_amp', 'rise_time_ms', 'fall_time_ms', 'pulse_width_ms',
    'auc', 'notch_depth', 'notch_time_norm',
]
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ppg-csv',    required=True,
                   help='Path to raw ppg-green CSV from CentrePoint')
    p.add_argument('--ibi-csv',    default=None,
                   help='(Optional) CentrePoint IBI export CSV')
    p.add_argument('--subject-id', required=True,
                   help='Participant ID, e.g. MK01')
    p.add_argument('--out-dir',    default='participant_features')
    p.add_argument('--model-dir',
                   default=r'lstm_model',
                   help='Directory with feat_mean.npy and feat_std.npy')
    p.add_argument('--channel',    type=int, default=CHANNEL_USE,
                   help='Which LED channel to use (0 or 1)')
    return p.parse_args()


# ── Load and clean PPG CSV ────────────────────────────────────────────────────

def load_ppg_csv(csv_path, channel=CHANNEL_USE):
    """
    Load raw LEAP PPG CSV. Returns (ppg_signal, t_ms_start, fs_hz).
    - Filters to one channel
    - Flags ADC saturation as NaN
    - Subtracts ambient channel to remove slow light interference
    - Returns signal in ADC units (will be bandpass-filtered next)
    """
    # ActiGraph files have a multi-line text header before the CSV data.
    # Find the row that contains the actual column names.
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    header_row = next(i for i, l in enumerate(lines)
                      if l.strip().startswith('Timestamp'))
    df = pd.read_csv(csv_path, skiprows=header_row)

    # Parse datetime timestamps (M/d/yyyy HH:MM:SS.fff) to Unix milliseconds.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                     format='%m/%d/%Y %H:%M:%S.%f',
                                     errors='coerce')
    # Explicitly cast to datetime64[ms] so int64 view is always Unix milliseconds,
    # regardless of whether pandas stores datetime64 internally as ns or us.
    ts_col = df['Timestamp'].values.astype('datetime64[ms]').astype(np.int64).astype(np.float64)

    # Separate into per-channel dataframes (rows alternate 0,1,0,1)
    df_ch = df[df['Channel'] == channel].reset_index(drop=True)
    n_samples = len(df_ch)

    # Infer fs: count how many rows appear per unique timestamp value
    # The timestamp repeats for batches; step = 1/fs seconds per channel sample
    unique_ts, counts = np.unique(ts_col[df['Channel'] == channel], return_counts=True)
    if len(unique_ts) > 1:
        # Each unique timestamp block has ~counts[i] samples at 1/fs intervals
        # The gap between block timestamps (in ms) / count gives period
        ts_diffs = np.diff(unique_ts)
        median_gap_ms = np.median(ts_diffs[ts_diffs > 0])
        median_count  = np.median(counts)
        fs = round(1000.0 * median_count / median_gap_ms) if median_gap_ms > 0 else FS_25
        fs = int(fs)
    else:
        # Single timestamp block — infer from file name or default to 25
        fs = FS_25

    # Validate fs is one of the expected rates
    if fs not in (25, 100):
        print(f"  Inferred fs={fs} Hz (unusual — defaulting to 25 Hz)")
        fs = FS_25

    print(f"  Detected sampling rate: {fs} Hz  |  {n_samples} samples  "
          f"~{n_samples / fs / 3600:.1f} hrs")

    # Build uniform time axis (ms) starting at first valid timestamp
    first_ts_ms = float(unique_ts[0]) if len(unique_ts) > 0 else 0.0
    t_ms = first_ts_ms + np.arange(n_samples) * (1000.0 / fs)

    # Green signal
    green = df_ch['Green'].values.astype(np.float64)

    # Ambient signal (on same channel rows)
    ambient = df_ch['Ambient'].values.astype(np.float64)

    # Flag saturation
    sat_mask = np.abs(green) >= SATURATION_VAL
    green[sat_mask] = np.nan

    # Ambient subtract to reduce slow interference (DC + respiration band artefact)
    # Clamp ambient saturation too
    ambient_sat = np.abs(ambient) >= SATURATION_VAL
    ambient[ambient_sat] = np.nan
    ppg = green - ambient

    nan_frac = np.mean(np.isnan(ppg))
    print(f"  NaN (saturation) fraction: {nan_frac*100:.1f}%")

    return ppg.astype(np.float32), t_ms, fs


# ── Load IBI CSV (optional) ───────────────────────────────────────────────────

def load_ibi_csv(ibi_path):
    """
    Load CentrePoint IBI export.
    Expected column: INTERBEAT_INTERVAL (seconds) and a timestamp column.
    Returns DataFrame with columns ['ibi_s', 'timestamp_ms'].
    """
    df = pd.read_csv(ibi_path)
    df.columns = [c.upper() for c in df.columns]

    ibi_col = next((c for c in df.columns if 'INTERBEAT' in c or 'IBI' in c), None)
    ts_col  = next((c for c in df.columns if 'TIMESTAMP' in c or 'TIME' in c), None)

    if ibi_col is None:
        raise ValueError(f"Cannot find IBI column in {ibi_path}. "
                         f"Columns: {list(df.columns)}")

    result = pd.DataFrame()
    result['ibi_s'] = df[ibi_col].astype(float)

    if ts_col:
        ts_parsed = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
        result['timestamp_ms'] = (ts_parsed.values
                                  .astype('datetime64[ms]')
                                  .astype(np.int64)
                                  .astype(np.float64))
    else:
        result['timestamp_ms'] = np.arange(len(df)) * result['ibi_s'].median() * 1000

    # Filter physiologically implausible RR (HR 25-200 bpm)
    valid = (result['ibi_s'] >= 0.30) & (result['ibi_s'] <= 2.40)
    result = result[valid].reset_index(drop=True)
    print(f"  IBI beats loaded: {len(result)}  "
          f"(mean HR = {60.0 / result['ibi_s'].mean():.0f} bpm)")
    return result


# ── Bandpass filter ───────────────────────────────────────────────────────────

def apply_bandpass(sig, fs):
    sos = butter(4, BP_BAND, btype='bandpass', fs=fs, output='sos')
    out = sig.copy()
    finite = np.isfinite(out)
    if finite.sum() < 100:
        return out
    if not finite.all():
        idx = np.arange(len(out))
        out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    out = sosfiltfilt(sos, out).astype(np.float32)
    out[~finite] = np.nan
    return out


# ── PPG peak detection ────────────────────────────────────────────────────────

def detect_ppg_peaks(ppg_filt, fs):
    sig = ppg_filt.copy()
    sig[np.isnan(sig)] = 0.0
    sig_range = np.nanmax(sig) - np.nanmin(sig)
    if sig_range < 1e-6:
        return np.array([], dtype=int)
    sig_norm = (sig - np.nanmin(sig)) / sig_range
    min_dist = int(0.4 * fs)
    peaks, _ = find_peaks(sig_norm, distance=min_dist, prominence=PEAK_PROM_PPG)
    return peaks


# ── HRV features ─────────────────────────────────────────────────────────────

def _lf_hf(rr_ms):
    if len(rr_ms) < MIN_RR_LF:
        return 0.0
    rr_s = rr_ms / 1000.0
    t    = np.cumsum(rr_s)
    t   -= t[0]
    if t[-1] < 10.0:
        return 0.0
    fs_rr = 4.0
    t_uni = np.arange(0, t[-1], 1.0 / fs_rr)
    try:
        rr_uni = interp1d(t, rr_s, kind='linear',
                          bounds_error=False, fill_value='extrapolate')(t_uni)
    except Exception:
        return 0.0
    rr_uni -= rr_uni.mean()
    n     = len(rr_uni)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_rr)
    psd   = np.abs(np.fft.rfft(rr_uni)) ** 2 / n
    lf    = _trapz(psd[(freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])],
                   freqs[(freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])])
    hf    = _trapz(psd[(freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])],
                   freqs[(freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])])
    return float(lf / hf) if hf > 1e-12 else 0.0


def compute_hrv(rr_ms):
    if len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    return {
        'mean_rr': float(np.mean(rr_ms)),
        'sdnn':    float(np.std(rr_ms, ddof=1)),
        'rmssd':   float(np.sqrt(np.mean(diff ** 2))),
        'pnn50':   float(np.mean(np.abs(diff) > 50)),
        'mean_hr': float(60000.0 / np.mean(rr_ms)),
        'lf_hf':   _lf_hf(rr_ms),
    }


# ── PPG morphology features ───────────────────────────────────────────────────

def compute_morphology(ppg_win, peak_idx, fs):
    lists = {k: [] for k in [
        'pulse_amp', 'rise_time_ms', 'fall_time_ms',
        'pulse_width_ms', 'auc', 'notch_depth', 'notch_time_norm',
    ]}

    for i in range(len(peak_idx) - 1):
        pk     = int(peak_idx[i])
        nxt_pk = int(peak_idx[i + 1])

        foot_start = (int(peak_idx[i-1]) + 2 if i > 0
                      else max(0, pk - int(0.8 * (nxt_pk - pk))))
        if foot_start >= pk:
            continue
        foot_seg = ppg_win[foot_start:pk]
        if not np.any(np.isfinite(foot_seg)):
            continue
        foot_idx = foot_start + int(np.nanargmin(foot_seg))

        next_search = pk + max(1, int(0.1 * (nxt_pk - pk)))
        if next_search >= nxt_pk:
            continue
        next_seg = ppg_win[next_search:nxt_pk]
        if not np.any(np.isfinite(next_seg)):
            continue
        next_foot_idx = next_search + int(np.nanargmin(next_seg))

        peak_val = ppg_win[pk]
        foot_val = ppg_win[foot_idx]
        if np.isnan(peak_val) or np.isnan(foot_val):
            continue
        amp = peak_val - foot_val
        if amp < 1e-6:
            continue

        rise_ms = (pk - foot_idx) / fs * 1000
        fall_ms = (next_foot_idx - pk) / fs * 1000
        if rise_ms <= 0 or fall_ms <= 0:
            continue

        half = foot_val + 0.5 * amp
        rc   = np.where(ppg_win[foot_idx:pk + 1] >= half)[0]
        fc   = np.where(ppg_win[pk:next_foot_idx + 1] <= half)[0]
        if len(rc) == 0 or len(fc) == 0:
            continue
        pw_ms = ((pk + fc[0]) - (foot_idx + rc[0])) / fs * 1000

        beat = ppg_win[foot_idx:next_foot_idx + 1]
        if not np.all(np.isfinite(beat)):
            continue
        auc  = float(_trapz(beat - foot_val)) / fs

        fall_full  = ppg_win[pk:next_foot_idx]
        search_end = max(6, int(0.6 * len(fall_full)))
        notch_cands, _ = find_peaks(-fall_full[:search_end], distance=3)
        if len(notch_cands) == 0:
            notch_depth = 0.0
            notch_time_norm = 0.5
        else:
            nr = notch_cands[np.argmin(fall_full[notch_cands])]
            notch_depth     = float((peak_val - fall_full[nr]) / amp)
            notch_time_norm = float(nr / len(fall_full))

        lists['pulse_amp'].append(float(amp))
        lists['rise_time_ms'].append(float(rise_ms))
        lists['fall_time_ms'].append(float(fall_ms))
        lists['pulse_width_ms'].append(float(pw_ms))
        lists['auc'].append(float(auc))
        lists['notch_depth'].append(float(notch_depth))
        lists['notch_time_norm'].append(float(notch_time_norm))

    if len(lists['pulse_amp']) < 2:
        return None
    return {k: float(np.median(v)) for k, v in lists.items()}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(ppg_filt, t_ms, fs, ibi_df=None):
    """
    Slide 30-second windows over the PPG signal.
    HRV: from ibi_df (CentrePoint) if provided, else from PPG peaks.
    Morphology: always from PPG peaks.
    Returns (features (W, 13), window_timestamps_ms (W,)).
    """
    win_samp  = WINDOW_SEC * fs
    step_samp = STEP_SEC   * fs

    all_ppg_peaks = detect_ppg_peaks(ppg_filt, fs)

    n          = len(ppg_filt)
    n_windows  = max(0, (n - win_samp) // step_samp + 1)
    feat_rows  = []
    win_ts     = []

    for w in range(n_windows):
        s = w * step_samp
        e = s + win_samp
        ppg_win = ppg_filt[s:e]

        if np.mean(np.isnan(ppg_win)) > MAX_NAN_PPG:
            continue

        # Window centre time (ms)
        win_centre_ms = float(t_ms[s] + (t_ms[e - 1] - t_ms[s]) / 2)

        # PPG peaks in window (local indices)
        mask_ppg   = (all_ppg_peaks >= s) & (all_ppg_peaks < e)
        win_peaks  = all_ppg_peaks[mask_ppg] - s

        # ── HRV ──
        if ibi_df is not None:
            # Use CentrePoint IBI: select beats whose timestamp falls in window
            win_start_ms = float(t_ms[s])
            win_end_ms   = float(t_ms[e - 1])
            in_win = ibi_df[
                (ibi_df['timestamp_ms'] >= win_start_ms) &
                (ibi_df['timestamp_ms'] <  win_end_ms)
            ]
            rr_ms = in_win['ibi_s'].values * 1000.0
        else:
            # Derive RR from PPG peaks
            if len(win_peaks) < MIN_BEATS:
                continue
            rr_ms = np.diff(win_peaks) / fs * 1000.0
            rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2400)]

        if len(rr_ms) < 2:
            continue

        hrv = compute_hrv(rr_ms)
        if hrv is None:
            continue

        # ── Morphology ──
        if len(win_peaks) < MIN_BEATS:
            continue
        morph = compute_morphology(ppg_win, win_peaks, fs)
        if morph is None:
            continue

        feat_rows.append([
            hrv['mean_rr'],  hrv['sdnn'],          hrv['rmssd'],
            hrv['pnn50'],    hrv['mean_hr'],        hrv['lf_hf'],
            morph['pulse_amp'],      morph['rise_time_ms'],  morph['fall_time_ms'],
            morph['pulse_width_ms'], morph['auc'],           morph['notch_depth'],
            morph['notch_time_norm'],
        ])
        win_ts.append(win_centre_ms)

    if not feat_rows:
        return None, None

    return (np.array(feat_rows, dtype=np.float32),
            np.array(win_ts,    dtype=np.float64))


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(features, model_dir):
    """Apply training-set z-score normalization. Returns normalized features."""
    mean_path = Path(model_dir) / 'feat_mean.npy'
    std_path  = Path(model_dir) / 'feat_std.npy'

    if not mean_path.exists() or not std_path.exists():
        print(f"  WARNING: Normalization stats not found in {model_dir}. "
              f"Returning unnormalized features. Run train_lstm.py first.")
        return features

    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)
    return (features - mean) / std


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'LEAP Preprocessing — Subject {args.subject_id}')
    print('=' * 60)

    # ── Load PPG ──
    print(f'\nLoading PPG: {args.ppg_csv}')
    ppg_raw, t_ms, fs = load_ppg_csv(args.ppg_csv, channel=args.channel)

    # ── Load IBI (optional) ──
    ibi_df = None
    if args.ibi_csv:
        print(f'\nLoading IBI: {args.ibi_csv}')
        ibi_df = load_ibi_csv(args.ibi_csv)
    else:
        print('\nNo IBI CSV provided — HRV will be derived from PPG peaks.')
        print('(For better HRV accuracy, export IBI from CentrePoint and '
              'pass with --ibi-csv)')

    # ── Bandpass filter ──
    print('\nApplying bandpass filter (0.5–4 Hz) ...')
    ppg_filt = apply_bandpass(ppg_raw, fs)

    # ── Extract features ──
    print(f'Extracting features ({WINDOW_SEC}s windows, {STEP_SEC}s step) ...')
    features, win_ts = extract_features(ppg_filt, t_ms, fs, ibi_df)

    if features is None:
        print('ERROR: No valid windows extracted. Check signal quality.')
        return

    print(f'  Windows extracted: {len(features)}  '
          f'(~{len(features) * STEP_SEC / 3600:.1f} hrs coverage)')

    # ── Normalize ──
    features_norm = normalize(features, args.model_dir)

    # ── Save ──
    out_path = out_dir / f'{args.subject_id}.npz'
    np.savez_compressed(
        out_path,
        features       = features_norm,
        features_raw   = features,          # unnormalized, for diagnostics
        timestamps_utc = win_ts,
        subject_id     = args.subject_id,
        feature_names  = np.array(FEATURE_NAMES),
        fs             = fs,
        channel_used   = args.channel,
        ibi_source     = 'centrepoint' if ibi_df is not None else 'ppg_peaks',
    )

    print(f'\nSaved -> {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB)')
    print(f'Features shape: {features_norm.shape}')
    print(f'Time range: {_ts_str(win_ts[0])} to {_ts_str(win_ts[-1])}')
    print('=' * 60)
    print('\nNext step: run LSTM inference on this file.')
    print('  python run_inference.py --features-npz', out_path)


def _ts_str(ts_ms):
    """Format Unix ms timestamp as readable datetime string."""
    import datetime
    return datetime.datetime.fromtimestamp(
        ts_ms / 1000, tz=datetime.timezone.utc
    ).strftime('%Y-%m-%d %H:%M UTC')


if __name__ == '__main__':
    main()
