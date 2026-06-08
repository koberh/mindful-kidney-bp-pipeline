#!/usr/bin/env python3
"""
Diagnostic v3 — tests direct HTTP directory listing approach.
Checks first 2 patient dirs. Takes ~30 seconds.
"""
import os
import re
import requests
import wfdb

DB      = 'mimic3wdb-matched'
VERSION = '1.0'
BASE    = f'https://physionet.org/files/{DB}/{VERSION}/'

# ── Enter your PhysioNet credentials here ───────────────────────────────────
PHYSIONET_USER = os.environ.get('PHYSIONET_USER', '')
PHYSIONET_PASS = os.environ.get('PHYSIONET_PASS', '')
# ─────────────────────────────────────────────────────────────────────────────

auth = (PHYSIONET_USER, PHYSIONET_PASS) if PHYSIONET_USER else None

print("Step 1: Get patient dir list...")
patient_dirs = wfdb.get_record_list(DB)
print(f"  Total: {len(patient_dirs)}  |  First 2: {patient_dirs[:2]}\n")

for pdir in patient_dirs[:2]:
    pdir_clean = pdir.rstrip('/')
    url = BASE + pdir_clean + '/'
    print(f"Step 2: HTTP GET {url}")

    r = requests.get(url, auth=auth, timeout=15)
    print(f"  Status: {r.status_code}")

    if r.status_code == 200:
        # Show raw snippet
        print(f"  Response snippet:\n    {r.text[:500]}\n")

        # Extract .hea filenames
        all_hea = re.findall(r'href="([^/"]+\.hea)"', r.text)
        print(f"  All .hea files found: {all_hea}")

        # Master records only (no _layout, no _0001 etc.)
        masters = [f.replace('.hea', '') for f in all_hea
                   if not re.search(r'_\d+\.hea$|_layout\.hea$|n\.hea$', f)]
        print(f"  Master records: {masters}\n")

        # Try reading header for first master record
        if masters:
            record = masters[0]
            pn_dir = f'{DB}/{pdir_clean}'
            print(f"  Trying wfdb.rdheader('{record}', pn_dir='{pn_dir}')")
            try:
                hdr = wfdb.rdheader(record, pn_dir=pn_dir)
                print(f"  Header type: {type(hdr).__name__}")
                if hasattr(hdr, 'seg_name') and hdr.seg_name:
                    print(f"  Multi-segment. Segs: {hdr.seg_name[:4]}")
                    for seg in hdr.seg_name:
                        if seg and seg != '~':
                            try:
                                layout = wfdb.rdheader(seg, pn_dir=pn_dir)
                                print(f"  Layout signals: {layout.sig_name}")
                            except Exception as e:
                                print(f"  Layout error: {e}")
                            break
                else:
                    print(f"  Signals: {getattr(hdr, 'sig_name', 'N/A')}")
            except Exception as e:
                print(f"  rdheader error: {e}")
    elif r.status_code == 401:
        print("  AUTH REQUIRED — fill in PHYSIONET_USER and PHYSIONET_PASS above")
    else:
        print(f"  Unexpected status. Body: {r.text[:300]}")
    print()

print("Done.")
