#!/usr/bin/env python3
"""
Search MIMIC-III matched waveform database for records with both
PLETH (PPG) and ABP (arterial blood pressure) signals.

Approach: HTTP directory listing per patient folder + wfdb header reads.
Resumable — re-run after any interruption to continue from checkpoint.

Requirements: pip install wfdb pandas requests
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import wfdb

# ── Config ───────────────────────────────────────────────────────────────────
DB      = 'mimic3wdb-matched'
VERSION = '1.0'
BASE    = f'https://physionet.org/files/{DB}/{VERSION}/'

PHYSIONET_USER = os.environ.get('PHYSIONET_USER', '')
PHYSIONET_PASS = os.environ.get('PHYSIONET_PASS', '')

OUT_DIR  = Path(r'mimic_search_output')
RESULTS  = OUT_DIR / 'mimic3_pleth_abp_records.csv'
PROGRESS = OUT_DIR / 'mimic3_search_progress.json'

SAVE_EVERY = 50   # checkpoint every N patient dirs
# ─────────────────────────────────────────────────────────────────────────────

AUTH = (PHYSIONET_USER, PHYSIONET_PASS)


def list_master_records(pdir_clean):
    """
    Fetch patient directory listing via HTTP and return master record names.
    Master records: e.g. 'p000020-2183-04-28-17-47'
    Excludes: segment files (_0001), layout files (_layout), numerics (n.hea)
    """
    url = BASE + pdir_clean + '/'
    try:
        r = requests.get(url, auth=AUTH, timeout=20)
        if r.status_code != 200:
            return []
        all_hea = re.findall(r'href="([^/"]+\.hea)"', r.text)
        masters = [
            f.replace('.hea', '') for f in all_hea
            if not re.search(r'_\d+\.hea$|_layout\.hea$|n\.hea$', f)
        ]
        return masters
    except Exception:
        return []


def get_signals(record_name, pdir_clean):
    """
    Return signal name list for a multi-segment record.
    Reads the layout segment header to get all possible signals.
    """
    pn_dir = f'{DB}/{pdir_clean}'
    try:
        hdr = wfdb.rdheader(record_name, pn_dir=pn_dir)
    except Exception:
        return []

    # Single-segment
    if not hasattr(hdr, 'seg_name') or not hdr.seg_name:
        return getattr(hdr, 'sig_name', []) or []

    # Multi-segment — read layout segment
    for seg in hdr.seg_name:
        if seg and seg != '~':
            try:
                layout = wfdb.rdheader(seg, pn_dir=pn_dir)
                return getattr(layout, 'sig_name', []) or []
            except Exception:
                continue
    return []


def has_pleth_and_abp(sig_names):
    upper = [s.upper() for s in sig_names]
    return (
        any('PLETH' in s for s in upper),
        any('ABP' in s for s in upper),
    )


def main():
    # Load checkpoint
    progress = {}
    if PROGRESS.exists():
        with open(PROGRESS) as f:
            progress = json.load(f)
        n_matches = sum(1 for v in progress.values() if v)
        print(f"Resuming — {len(progress):,} dirs already checked, "
              f"{n_matches} matches so far.\n")

    print(f"Fetching patient directory list from {DB}...")
    patient_dirs = [p.rstrip('/') for p in wfdb.get_record_list(DB)]
    print(f"Total patient dirs: {len(patient_dirs):,}\n")

    matches = [v for plist in progress.values() if plist for v in plist]
    t0 = time.time()
    new_checked = 0

    for pdir in patient_dirs:
        if pdir in progress:
            continue

        # List master records in this patient directory
        master_records = list_master_records(pdir)

        patient_matches = []
        for record in master_records:
            sigs = get_signals(record, pdir)
            has_pleth, has_abp = has_pleth_and_abp(sigs)

            if has_pleth and has_abp:
                entry = {
                    'patient_dir': pdir,
                    'record':      record,
                    'pn_dir':      f'{DB}/{pdir}',
                    'signals':     ','.join(sigs),
                }
                patient_matches.append(entry)
                matches.append(entry)
                print(f"  MATCH [{len(matches)}]: {pdir}/{record}  {sigs}")

        progress[pdir] = patient_matches if patient_matches else None
        new_checked += 1

        if new_checked % SAVE_EVERY == 0:
            elapsed  = time.time() - t0
            rate     = new_checked / elapsed if elapsed > 0 else 1
            eta_min  = (len(patient_dirs) - len(progress)) / rate / 60
            print(f"  [{len(progress):,}/{len(patient_dirs):,} checked | "
                  f"{len(matches)} matches | ~{eta_min:.0f} min remaining]")
            _save(progress, matches)

    _save(progress, matches)
    print(f"\nDone. Records with PLETH + ABP: {len(matches)}")
    print(f"Results → {RESULTS}")


def _save(progress, matches):
    with open(PROGRESS, 'w') as f:
        json.dump(progress, f)
    if matches:
        pd.DataFrame(matches).to_csv(RESULTS, index=False)


if __name__ == '__main__':
    main()
