#!/usr/bin/env python3
"""
Step 1: Filter MIMIC-III matched records by session duration,
then download PLETH (PPG) + ABP waveforms for LSTM training.

Two phases:
  Phase 1 (fast)  — reads headers only, checks duration, no data downloaded
  Phase 2 (slow)  — downloads filtered records, saves as compressed .npz

Output: C:\\Users\\ONEMI\\Desktop\\Kobe\\mimic_downloads\\
  {patient}_{record}.npz   one file per record
  download_log.json         checkpoint (resumable)
  download_manifest.csv     summary of all downloaded records

Requirements: pip install wfdb pandas numpy
"""

import concurrent.futures
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# ── Config ───────────────────────────────────────────────────────────────────
PHYSIONET_USER   = os.environ.get('PHYSIONET_USER', '')
PHYSIONET_PASS   = os.environ.get('PHYSIONET_PASS', '')

CSV_IN           = Path(r'mimic3_pleth_abp_records.csv')
OUT_DIR          = Path(r'mimic_downloads')
LOG_FILE         = OUT_DIR / 'download_log.json'
MANIFEST_FILE    = OUT_DIR / 'download_manifest.csv'
DURATION_LOG     = OUT_DIR / 'duration_check_log.json'

MIN_DURATION_HRS = 6.0    # minimum session length to include
MAX_HOURS_PER_RECORD = 8  # cap each download at 8 hours (keeps files ~30 MB each)
N_RECORDS        = 500    # randomly sample this many qualifying records
RANDOM_SEED      = 42
SAVE_EVERY       = 10     # checkpoint every N downloads
# ─────────────────────────────────────────────────────────────────────────────


def setup_auth():
    """Write PhysioNet credentials to netrc for automatic wfdb auth."""
    home     = Path.home()
    filename = '_netrc' if platform.system() == 'Windows' else '.netrc'
    netrc    = home / filename
    entry    = (f'machine physionet.org\n'
                f'login {PHYSIONET_USER}\n'
                f'password {PHYSIONET_PASS}\n\n')
    existing = netrc.read_text() if netrc.exists() else ''
    if 'physionet.org' not in existing:
        with open(netrc, 'a') as f:
            f.write(entry)
        print(f"Credentials written to {netrc}")
    else:
        print(f"Credentials already in {netrc}")


def get_duration_hrs(record, pn_dir):
    """
    Read master header and return session duration in hours.
    Returns None on error.
    """
    try:
        hdr = wfdb.rdheader(record, pn_dir=pn_dir)
        if hdr.sig_len and hdr.fs:
            return hdr.sig_len / hdr.fs / 3600
        return None
    except Exception:
        return None


def download_record(record, pn_dir, out_path, max_hours=MAX_HOURS_PER_RECORD):
    """
    Download PLETH and ABP channels for one record, capped at max_hours.
    Saves as .npz. Returns dict with metadata, or None on failure.
    """
    sampto = int(max_hours * 3600 * 125)  # 125 Hz, cap at max_hours
    try:
        hdr = wfdb.rdheader(record, pn_dir=pn_dir)
        if hdr.sig_len:
            sampto = min(sampto, hdr.sig_len)
    except Exception:
        pass
    try:
        rec = wfdb.rdrecord(
            record,
            pn_dir=pn_dir,
            channel_names=['PLETH', 'ABP'],
            sampto=sampto,
        )
    except Exception as e:
        print(f"    Download error: {e}")
        return None

    if rec is None or rec.p_signal is None:
        return None

    sig_names = [s.upper() for s in rec.sig_name]
    if 'PLETH' not in sig_names or 'ABP' not in sig_names:
        print(f"    Signal missing after download: {rec.sig_name}")
        return None

    pleth_idx = sig_names.index('PLETH')
    abp_idx   = sig_names.index('ABP')
    pleth     = rec.p_signal[:, pleth_idx].astype(np.float32)
    abp       = rec.p_signal[:, abp_idx].astype(np.float32)
    fs        = int(rec.fs)
    dur_hrs   = len(pleth) / fs / 3600

    # Quality check — skip records with >50% NaN in either channel
    pleth_valid = np.mean(~np.isnan(pleth))
    abp_valid   = np.mean(~np.isnan(abp))
    if pleth_valid < 0.5 or abp_valid < 0.5:
        print(f"    Skipped — too much missing data "
              f"(PLETH {pleth_valid*100:.0f}% valid, ABP {abp_valid*100:.0f}% valid)")
        return None
    print(f"    Signal quality: PLETH {pleth_valid*100:.0f}% valid, "
          f"ABP {abp_valid*100:.0f}% valid")

    np.savez_compressed(
        out_path,
        pleth       = pleth,
        abp         = abp,
        fs          = fs,
        dur_hrs     = dur_hrs,
        record      = record,
        pn_dir      = pn_dir,
    )

    return {
        'record':    record,
        'pn_dir':    pn_dir,
        'fs':        fs,
        'dur_hrs':   round(dur_hrs, 2),
        'n_samples': len(pleth),
        'file':      str(out_path),
    }


# ── Phase 1: Duration check ───────────────────────────────────────────────────

def phase1_duration_check(df):
    """
    Check session duration for every record via header read (no data download).
    Returns DataFrame with duration column, filtered to >= MIN_DURATION_HRS.
    """
    # Load prior duration checks if available
    dur_log = {}
    if DURATION_LOG.exists():
        with open(DURATION_LOG) as f:
            dur_log = json.load(f)
        print(f"  Resuming duration check — {len(dur_log)} already checked\n")

    durations = []
    t0 = time.time()

    for i, row in df.iterrows():
        key = row['record']
        if key in dur_log:
            durations.append(dur_log[key])
            continue

        dur = get_duration_hrs(row['record'], row['pn_dir'])
        dur_log[key] = dur
        durations.append(dur)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed if elapsed > 0 else 1
            eta_min = (len(df) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(df)} checked | ~{eta_min:.0f} min remaining]")
            with open(DURATION_LOG, 'w') as f:
                json.dump(dur_log, f)

    with open(DURATION_LOG, 'w') as f:
        json.dump(dur_log, f)

    df = df.copy()
    df['duration_hrs'] = durations
    df = df[df['duration_hrs'] >= MIN_DURATION_HRS].reset_index(drop=True)
    return df


# ── Phase 2: Download ─────────────────────────────────────────────────────────

def phase2_download(filtered_df):
    """Download PLETH + ABP for all filtered records."""
    # Load checkpoint
    log = {}
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
        print(f"  Resuming — {len(log)} records already downloaded\n")

    manifest = [v for v in log.values() if v]
    t0       = time.time()
    new_dl   = 0

    for i, row in filtered_df.iterrows():
        key = row['record']
        if key in log:
            continue

        patient = row['patient_dir'].replace('/', '_')
        fname   = f"{patient}_{row['record']}.npz"
        out_path = OUT_DIR / fname

        print(f"  [{i+1}/{len(filtered_df)}] {row['record']} "
              f"({row['duration_hrs']:.1f} hrs)")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(download_record, row['record'], row['pn_dir'], out_path)
        try:
            result = future.result(timeout=900)  # 15-minute cap per record
        except concurrent.futures.TimeoutError:
            print(f"    Timed out after 10 min — skipping")
            result = None
        finally:
            executor.shutdown(wait=False)  # don't block on hung thread

        if result:
            log[key] = result
            manifest.append(result)
            print(f"    Saved -> {fname}  "
                  f"({out_path.stat().st_size / 1e6:.1f} MB)")
        else:
            log[key] = None
            print(f"    FAILED — skipped")

        new_dl += 1

        if new_dl % SAVE_EVERY == 0:
            _save_log(log, manifest)
            elapsed  = time.time() - t0
            rate     = new_dl / elapsed if elapsed > 0 else 1
            eta_min  = (len(filtered_df) - len(log)) / rate / 60
            free_gb  = shutil.disk_usage('C:/').free / 1e9
            print(f"  [{len(log)}/{len(filtered_df)} done | "
                  f"~{eta_min:.0f} min left | C: free: {free_gb:.0f} GB]")

    _save_log(log, manifest)
    return manifest


def _save_log(log, manifest):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f)
    if manifest:
        pd.DataFrame(manifest).to_csv(MANIFEST_FILE, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MIMIC-III Download Script — PLETH + ABP Waveforms")
    print("=" * 60)

    # Auth
    setup_auth()

    # Disk space
    free_gb = shutil.disk_usage('C:/').free / 1e9
    print(f"\nC: drive free space: {free_gb:.0f} GB\n")

    # Load CSV
    df = pd.read_csv(CSV_IN)
    print(f"Records in CSV: {len(df)}\n")

    # ── Phase 1 ──
    print(f"Phase 1: Checking session durations (>= {MIN_DURATION_HRS} hrs)...")
    filtered = phase1_duration_check(df)

    est_size_gb = len(filtered) * 0.025  # ~25 MB avg per record compressed
    print(f"\n  Records >= {MIN_DURATION_HRS} hrs: {len(filtered)}")
    print(f"  Estimated download size: ~{est_size_gb:.0f} GB")
    print(f"  C: free space: {free_gb:.0f} GB")

    if free_gb < est_size_gb * 1.2:
        print("\n  WARNING: May not have enough space. Consider raising "
              "MIN_DURATION_HRS at the top of this script.")

    # Randomly sample N_RECORDS from qualifying records
    if len(filtered) > N_RECORDS:
        filtered = filtered.sample(n=N_RECORDS, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  Randomly sampled: {N_RECORDS} records (seed={RANDOM_SEED})")

    # Save filtered list
    filtered_csv = OUT_DIR / 'filtered_records.csv'
    filtered.to_csv(filtered_csv, index=False)
    print(f"\n  Filtered list saved -> {filtered_csv}")

    # Confirm before downloading
    print(f"\nReady to download {len(filtered)} records.")
    confirm = input("Start download? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled. Re-run when ready.")
        return

    # ── Phase 2 ──
    print(f"\nPhase 2: Downloading waveforms to {OUT_DIR} ...\n")
    manifest = phase2_download(filtered)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(manifest)} records downloaded.")
    print(f"Manifest -> {MANIFEST_FILE}")
    used_gb = sum(
        Path(m['file']).stat().st_size for m in manifest
        if Path(m['file']).exists()
    ) / 1e9
    print(f"Total disk used: {used_gb:.1f} GB")
    print("=" * 60)


if __name__ == '__main__':
    main()
