#!/usr/bin/env python3
"""
Preprocess MIMIC-III .npz records into HRV + PPG morphology feature vectors
with relative BP labels for LSTM training (Radha et al. 2019 architecture).

Per record:
  1. Bandpass filter PLETH 0.5-4 Hz at 125 Hz
  2. Detect PPG peaks -> compute RR intervals
  3. Extract ABP beat-by-beat SBP/DBP/MAP
  4. Slice into 30-second windows (15-second step, 50% overlap)
  5. Per window: HRV features (6) + PPG morphology features (7) = 13 total
  6. Subtract per-record median BP baseline -> relative labels
  7. Quality-filter windows

HRV features (6):
  mean_rr, sdnn, rmssd, pnn50, mean_hr, lf_hf

PPG morphology features (7, per-beat medians over window):
  pulse_amp, rise_time_ms, fall_time_ms, pulse_width_ms,
  auc, notch_depth, notch_time_norm

Output: C:\\Users\\ONEMI\\Desktop\\Kobe\\mimic_preprocessed\\
  {record_id}.npz per record:
      features      float32 (W, 13)
      sbp           float32 (W,)   relative SBP mmHg (delta from record median)
      dbp           float32 (W,)   relative DBP mmHg
      map_bp        float32 (W,)   relative MAP mmHg
      record        str
      feature_names str array (13,)
  preprocess_log.json  checkpoint (resumable)

Requirements: pip install numpy scipy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Config ────────────────────────────────────────────────────────────────────
IN_DIR  = Path(r'mimic_downloads')
OUT_DIR = Path(r'mimic_preprocessed')
LOG     = OUT_DIR / 'preprocess_log.json'

FS          = 125           # MIMIC-III waveform rate (Hz)
BP_BAND     = (0.5, 4.0)   # PPG bandpass (Hz)

WINDOW_SEC  = 30            # window length
STEP_SEC    = 15            # 50% overlap
WIN_SAMP    = WINDOW_SEC * FS    # 3750 samples per window
STEP_SAMP   = STEP_SEC   * FS   # 1875 samples per step

# PPG peak detection
PEAK_DIST_PPG = int(0.4 * FS)   # min 0.4 s between peaks (max ~150 bpm)
# ABP peak detection
PEAK_DIST_ABP = int(0.3 * FS)   # min 0.3 s between peaks (max ~200 bpm)
PEAK_PROM_ABP = 8.0              # mmHg prominence for ABP systolic peaks

# BP plausibility bounds
SBP_RANGE = (60, 260)
DBP_RANGE = (20, 150)

# Window quality gates
MIN_BEATS   = 5     # minimum PPG and ABP beats per window
MAX_NAN_PPG = 0.10  # maximum NaN fraction in PPG window

# LF/HF spectral bands (Hz) and minimum RR count
LF_BAND     = (0.04, 0.15)
HF_BAND     = (0.15, 0.40)
MIN_RR_LF   = 12   # minimum RR intervals to attempt LF/HF estimation

FEATURE_NAMES = [
    'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'mean_hr', 'lf_hf',
    'pulse_amp', 'rise_time_ms', 'fall_time_ms', 'pulse_width_ms',
    'auc', 'notch_depth', 'notch_time_norm',
]
# ─────────────────────────────────────────────────────────────────────────────


def make_bandpass():
    return butter(4, BP_BAND, btype='bandpass', fs=FS, output='sos')


def apply_bandpass(sig, sos):
    """Bandpass filter, NaN-aware (interpolates over NaN before filtering)."""
    out = sig.copy()
    finite = np.isfinite(out)
    if finite.sum() < 50:
        return out
    if not finite.all():
        idx = np.arange(len(out))
        out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    out = sosfiltfilt(sos, out).astype(np.float32)
    out[~finite] = np.nan
    return out


# ── ABP ───────────────────────────────────────────────────────────────────────

def detect_abp_beats(abp):
    """Return (peak_idx, sbp_vals, dbp_vals) from ABP waveform."""
    clean = abp.copy()
    med = np.nanmedian(clean)
    clean[np.isnan(clean)] = med if np.isfinite(med) else 80.0
    clean = np.clip(clean, 0, 300)

    peaks, _ = find_peaks(clean, distance=PEAK_DIST_ABP, prominence=PEAK_PROM_ABP)
    if len(peaks) < 2:
        return np.array([]), np.array([]), np.array([])

    sbp = abp[peaks]
    dbp = np.array([
        np.nanmin(abp[peaks[i]:peaks[i + 1]])
        for i in range(len(peaks) - 1)
    ])
    return peaks[:-1], sbp[:-1], dbp


# ── PPG ───────────────────────────────────────────────────────────────────────

def detect_ppg_peaks(ppg_filt):
    """Detect systolic peaks in bandpass-filtered PPG. Returns peak indices."""
    sig = ppg_filt.copy()
    sig[np.isnan(sig)] = 0.0
    sig_range = np.nanmax(sig) - np.nanmin(sig)
    if sig_range < 1e-6:
        return np.array([], dtype=int)
    sig_norm = (sig - np.nanmin(sig)) / sig_range
    peaks, _ = find_peaks(sig_norm, distance=PEAK_DIST_PPG, prominence=0.05)
    return peaks


# ── HRV features ──────────────────────────────────────────────────────────────

def _lf_hf(rr_ms):
    """LF/HF ratio from RR series (ms). Returns 0.0 if insufficient data."""
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
    lf = _trapz(psd[(freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])],
                freqs[(freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])])
    hf = _trapz(psd[(freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])],
                freqs[(freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])])
    return float(lf / hf) if hf > 1e-12 else 0.0


def compute_hrv(rr_ms):
    """Return dict of 6 HRV features from RR intervals in ms."""
    if len(rr_ms) < 2:
        return None
    diff  = np.diff(rr_ms)
    return {
        'mean_rr': float(np.mean(rr_ms)),
        'sdnn':    float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
        'rmssd':   float(np.sqrt(np.mean(diff ** 2))),
        'pnn50':   float(np.mean(np.abs(diff) > 50)),
        'mean_hr': float(60000.0 / np.mean(rr_ms)),
        'lf_hf':   _lf_hf(rr_ms),
    }


# ── PPG morphology features ───────────────────────────────────────────────────

def compute_morphology(ppg_win, peak_idx):
    """
    Per-beat PPG morphology features, returned as window-level medians.
    Returns dict of 7 features, or None if < 2 valid beats.
    """
    lists = {k: [] for k in [
        'pulse_amp', 'rise_time_ms', 'fall_time_ms',
        'pulse_width_ms', 'auc', 'notch_depth', 'notch_time_norm',
    ]}

    for i in range(len(peak_idx) - 1):
        pk     = int(peak_idx[i])
        nxt_pk = int(peak_idx[i + 1])

        # Diastolic foot: minimum in the region before this peak
        foot_start = int(peak_idx[i - 1]) + 2 if i > 0 else max(0, pk - int(0.8 * (nxt_pk - pk)))
        if foot_start >= pk:
            continue
        foot_seg = ppg_win[foot_start:pk]
        if not np.any(np.isfinite(foot_seg)):
            continue
        foot_idx = foot_start + int(np.nanargmin(foot_seg))

        # Next diastolic foot: minimum between this peak and the next
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

        rise_ms = (pk - foot_idx) / FS * 1000
        fall_ms = (next_foot_idx - pk) / FS * 1000
        if rise_ms <= 0 or fall_ms <= 0:
            continue

        # Pulse width at 50% amplitude
        half = foot_val + 0.5 * amp
        rise_seg = ppg_win[foot_idx:pk + 1]
        fall_seg = ppg_win[pk:next_foot_idx + 1]
        rc = np.where(rise_seg >= half)[0]
        fc = np.where(fall_seg <= half)[0]
        if len(rc) == 0 or len(fc) == 0:
            continue
        pw_ms = ((pk + fc[0]) - (foot_idx + rc[0])) / FS * 1000

        # AUC: foot-subtracted area under one beat
        beat = ppg_win[foot_idx:next_foot_idx + 1]
        if not np.all(np.isfinite(beat)):
            continue
        auc  = float(_trapz(beat - foot_val)) / FS

        # Dicrotic notch: local minimum in first 60% of falling edge
        fall_full  = ppg_win[pk:next_foot_idx]
        search_end = max(6, int(0.6 * len(fall_full)))
        notch_cands, _ = find_peaks(-fall_full[:search_end], distance=3)
        if len(notch_cands) == 0:
            notch_depth     = 0.0
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


# ── Record pipeline ───────────────────────────────────────────────────────────

def process_record(npz_path, sos):
    d      = np.load(npz_path, allow_pickle=True)
    pleth  = d['pleth'].astype(np.float32)
    abp    = d['abp'].astype(np.float32)
    record = str(d['record'])

    n     = min(len(pleth), len(abp))
    pleth = pleth[:n]
    abp   = abp[:n]

    ppg_filt = apply_bandpass(pleth, sos)

    # Per-record ABP baseline (all valid beats)
    peak_abp, sbp_all, dbp_all = detect_abp_beats(abp)
    if len(peak_abp) < 20:
        return None
    valid = (
        (sbp_all >= SBP_RANGE[0]) & (sbp_all <= SBP_RANGE[1]) &
        (dbp_all >= DBP_RANGE[0]) & (dbp_all <= DBP_RANGE[1]) &
        (sbp_all > dbp_all)
    )
    if valid.sum() < 10:
        return None
    peak_abp = peak_abp[valid]
    sbp_all  = sbp_all[valid]
    dbp_all  = dbp_all[valid]
    map_all  = dbp_all + (sbp_all - dbp_all) / 3.0
    sbp_base = float(np.median(sbp_all))
    dbp_base = float(np.median(dbp_all))
    map_base = float(np.median(map_all))

    # All PPG peaks (for windowed extraction)
    all_ppg_peaks = detect_ppg_peaks(ppg_filt)

    n_windows = max(0, (n - WIN_SAMP) // STEP_SAMP + 1)
    feat_rows, sbp_rows, dbp_rows, map_rows = [], [], [], []

    for w in range(n_windows):
        s = w * STEP_SAMP
        e = s + WIN_SAMP
        ppg_win = ppg_filt[s:e]

        if np.mean(np.isnan(ppg_win)) > MAX_NAN_PPG:
            continue

        # PPG peaks in window (shifted to window-local indices)
        mask_ppg  = (all_ppg_peaks >= s) & (all_ppg_peaks < e)
        win_peaks = all_ppg_peaks[mask_ppg] - s

        if len(win_peaks) < MIN_BEATS:
            continue

        # RR intervals (filter physiologically implausible)
        rr_ms = np.diff(win_peaks) / FS * 1000.0
        rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2400)]
        if len(rr_ms) < 2:
            continue

        hrv   = compute_hrv(rr_ms)
        morph = compute_morphology(ppg_win, win_peaks)
        if hrv is None or morph is None:
            continue

        # ABP label for this window
        mask_abp = (peak_abp >= s) & (peak_abp < e)
        if mask_abp.sum() < MIN_BEATS:
            continue
        win_sbp = float(np.median(sbp_all[mask_abp])) - sbp_base
        win_dbp = float(np.median(dbp_all[mask_abp])) - dbp_base
        win_map = float(np.median(map_all[mask_abp])) - map_base

        feat_rows.append([
            hrv['mean_rr'],  hrv['sdnn'],          hrv['rmssd'],
            hrv['pnn50'],    hrv['mean_hr'],        hrv['lf_hf'],
            morph['pulse_amp'],     morph['rise_time_ms'],  morph['fall_time_ms'],
            morph['pulse_width_ms'], morph['auc'],  morph['notch_depth'],
            morph['notch_time_norm'],
        ])
        sbp_rows.append(win_sbp)
        dbp_rows.append(win_dbp)
        map_rows.append(win_map)

    if len(feat_rows) < 5:
        return None

    return record, {
        'features': np.array(feat_rows, dtype=np.float32),
        'sbp':      np.array(sbp_rows,  dtype=np.float32),
        'dbp':      np.array(dbp_rows,  dtype=np.float32),
        'map_bp':   np.array(map_rows,  dtype=np.float32),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(IN_DIR))
    p.add_argument('--out-dir',  default=str(OUT_DIR))
    args = p.parse_args()

    in_dir  = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    log_file = out_dir / 'preprocess_log.json'

    out_dir.mkdir(parents=True, exist_ok=True)

    log = {}
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
        print(f"Resuming -- {len(log)} records already processed\n")

    npz_files = sorted(in_dir.glob('*.npz'))
    print(f"Input records : {len(npz_files)}")
    print(f"Output dir    : {out_dir}")
    print(f"Window        : {WINDOW_SEC}s  step {STEP_SEC}s  ({WIN_SAMP} samples)")
    print(f"Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}\n")

    sos = make_bandpass()
    total_windows = 0
    t0 = time.time()

    for i, path in enumerate(npz_files):
        key = path.stem
        if key in log:
            total_windows += log[key].get('n_windows', 0)
            continue

        print(f"[{i+1}/{len(npz_files)}] {path.name}", end=' ... ', flush=True)
        try:
            result = process_record(path, sos)
        except Exception as ex:
            print(f"ERROR: {ex}")
            log[key] = {'status': 'error', 'n_windows': 0}
            continue

        if result is None:
            print("SKIPPED (quality)")
            log[key] = {'status': 'skipped', 'n_windows': 0}
            continue

        record_id, arrays = result
        n_win    = len(arrays['sbp'])
        out_path = out_dir / f"{key}.npz"

        np.savez_compressed(
            out_path,
            features      = arrays['features'],
            sbp           = arrays['sbp'],
            dbp           = arrays['dbp'],
            map_bp        = arrays['map_bp'],
            record        = record_id,
            feature_names = np.array(FEATURE_NAMES),
        )

        total_windows += n_win
        mb = out_path.stat().st_size / 1e6
        print(f"{n_win} windows  ({mb:.2f} MB)")
        log[key] = {'status': 'ok', 'n_windows': n_win, 'file': str(out_path)}

        if (i + 1) % 20 == 0:
            with open(log_file, 'w') as f:
                json.dump(log, f)
            elapsed = time.time() - t0
            rate    = elapsed / max(i + 1, 1)
            eta     = (len(npz_files) - i - 1) * rate / 60
            done    = sum(1 for v in log.values() if v['status'] == 'ok')
            print(f"  -- checkpoint: {done} ok | ~{eta:.0f} min remaining\n")

    with open(log_file, 'w') as f:
        json.dump(log, f)

    ok      = sum(1 for v in log.values() if v['status'] == 'ok')
    skipped = sum(1 for v in log.values() if v['status'] != 'ok')
    print(f"\n{'='*60}")
    print(f"Done. {ok} records -> {total_windows} total windows  ({skipped} skipped)")
    print(f"Output: {out_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
