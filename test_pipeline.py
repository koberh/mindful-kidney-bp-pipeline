#!/usr/bin/env python3
"""
Smoke test for mindful-kidney-bp-pipeline.

Creates minimal synthetic data and runs each pipeline step.
Does NOT require MIMIC-III access, Aurora-BP, or a trained model upfront.
Runtime: ~30-60 seconds on CPU.

Usage:
    python test_pipeline.py
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
REPO             = Path(__file__).parent
TMP              = REPO / '_smoke_test'
FS_MIMIC         = 125
FS_LEAP          = 100
HR_HZ            = 1.2       # 72 bpm
N_RECORDS        = 10        # synthetic MIMIC records
MIMIC_DURATION_S = 15 * 60  # 15 min each (needs >21 quality windows)
LEAP_DURATION_S  = 10 * 60  # 10 min LEAP data
# ---------------------------------------------------------------------------


def _run(cmd_list, label):
    """Run a script via subprocess; return (success, output_tail)."""
    result = subprocess.run(
        [sys.executable] + cmd_list,
        capture_output=True, text=True, cwd=REPO,
    )
    ok    = result.returncode == 0
    badge = "[PASS]" if ok else "[FAIL]"
    print(f"  {badge}  {label}")
    if not ok:
        tail = (result.stdout + result.stderr).strip().split("\n")
        for line in tail[-12:]:
            print(f"         {line}")
    return ok


# ── Synthetic data generators ────────────────────────────────────────────────

def _ppg(n, fs):
    t   = np.arange(n) / fs
    sig = np.sin(2*np.pi*HR_HZ*t) + 0.3*np.sin(2*np.pi*2*HR_HZ*t)
    sig += np.random.randn(n).astype(np.float32) * 0.05
    return sig.astype(np.float32)

def _abp(n, fs):
    t   = np.arange(n) / fs
    sig = 120 + 20 * np.sin(2*np.pi*HR_HZ*t)
    sig += np.random.randn(n).astype(np.float32) * 1.0
    return sig.astype(np.float32)


def make_mimic_raws(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = FS_MIMIC * MIMIC_DURATION_S
    for i in range(N_RECORDS):
        np.savez(out_dir / f"record_{i:03d}.npz",
                 pleth=_ppg(n, FS_MIMIC),
                 abp=_abp(n, FS_MIMIC),
                 record=f"synthetic_{i:03d}")
    print(f"    {N_RECORDS} MIMIC records ({MIMIC_DURATION_S//60} min each, 125 Hz)")


def make_leap_csv(path: Path, t0: datetime):
    n = FS_LEAP * LEAP_DURATION_S
    ppg = _ppg(n, FS_LEAP)
    green   = (12000 + ppg * 500).astype(int)
    ambient = np.full(n, 100, dtype=int)
    with open(path, "w") as f:
        f.write("ActiGraph CentrePoint Export\n")
        f.write("Timestamp,Channel,Green,Ambient\n")
        for i in range(n):
            ts = t0 + timedelta(milliseconds=i * (1000 // FS_LEAP))
            ts_str = ts.strftime("%m/%d/%Y %H:%M:%S.") + f"{ts.microsecond//1000:03d}"
            for ch in (0, 1):
                f.write(f"{ts_str},{ch},{green[i]},{ambient[i]}\n")
    print(f"    LEAP CSV: {LEAP_DURATION_S//60} min @ {FS_LEAP} Hz, {n*2} rows")


def make_cuff_csv(path: Path, t0: datetime, n=6):
    rows = [{"subject_id": "SUBJ01",
             "timestamp_utc": (t0 + timedelta(minutes=5+i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "sbp_mmhg": 120 + int(np.random.randint(-8, 8)),
             "dbp_mmhg":  78 + int(np.random.randint(-5, 5))}
            for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    Cuff CSV: {n} readings (t0+5min .. t0+{5+n-1}min)")


def make_temp_csv(path: Path):
    # 3 hours at 4-second intervals, starting 20:00 UTC
    t0  = datetime(2026, 4, 27, 20, 0, 0, tzinfo=timezone.utc)
    n   = 3 * 3600 // 4
    with open(path, "w") as f:
        f.write("ActiGraph Temperature Export\n")
        f.write("Timestamp,Temperature\n")
        for i in range(n):
            ts   = t0 + timedelta(seconds=i * 4)
            hour = ts.hour
            temp = (33.5 if (hour >= 22 or hour < 2) else 31.8) + np.random.randn() * 0.15
            f.write(ts.strftime("%m/%d/%Y %H:%M:%S.000") + f",{temp:.2f}\n")
    print(f"    Temp CSV: 3 hours @ 4 s intervals ({n} rows)")


# ── Import / package check ───────────────────────────────────────────────────

def check_imports():
    needed = {
        "numpy":      "numpy",
        "scipy":      "scipy",
        "pandas":     "pandas",
        "sklearn":    "scikit-learn",
        "wfdb":       "wfdb",
        "requests":   "requests",
        "tensorflow": "tensorflow  (needed by train_lstm.py / run_inference.py)",
        "torch":      "torch       (needed by path_a_radha/ Phase 5 scripts)",
        "antropy":    "antropy     (needed by path_a_radha/ Phase 5 scripts)",
    }
    ok_all = True
    for mod, label in needed.items():
        try:
            __import__(mod)
            print(f"  [OK]      {label}")
        except ImportError:
            print(f"  [MISSING] {label}")
            if mod == "tensorflow":
                ok_all = False     # blocks in-repo training+inference
    return ok_all


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 62)
    print("mindful-kidney-bp-pipeline -- smoke test")
    print("=" * 62)

    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()

    results = {}

    # 0 ── Package check ──────────────────────────────────────────────────────
    print("\n[0] Package imports")
    imports_ok = check_imports()

    # ── Synthetic data ────────────────────────────────────────────────────────
    print("\n[data] Creating synthetic inputs")
    mimic_raw = TMP / "mimic_raw"
    mimic_out = TMP / "mimic_features"
    model_dir = TMP / "lstm_model"
    leap_csv  = TMP / "SUBJ01_leap.csv"
    feat_dir  = TMP / "participant_features"
    inf_dir   = TMP / "inference_results"
    cuff_csv  = TMP / "cuff_readings.csv"
    temp_csv  = TMP / "temperature.csv"
    sleep_dir = TMP / "sleep_output"

    t0_leap = datetime(2026, 4, 28, 10, 0, 0, tzinfo=timezone.utc)

    make_mimic_raws(mimic_raw)
    make_leap_csv(leap_csv, t0_leap)
    make_cuff_csv(cuff_csv, t0_leap)
    make_temp_csv(temp_csv)

    # 1 ── preprocess_mimic.py ────────────────────────────────────────────────
    print("\n[1] preprocess_mimic.py  (numpy/scipy only - no TF needed)")
    ok = _run(["preprocessing/preprocess_mimic.py",
               "--data-dir", str(mimic_raw),
               "--out-dir",  str(mimic_out)],
              "preprocess_mimic.py")
    results["preprocess_mimic"] = ok
    if ok:
        npzs = list(mimic_out.glob("*.npz"))
        print(f"         -> {len(npzs)} feature .npz files written")
        if npzs:
            d = np.load(npzs[0], allow_pickle=True)
            print(f"         -> features shape: {d['features'].shape}  "
                  f"(expect (N, 13))")

    # 2 ── train_lstm.py  (TensorFlow) ───────────────────────────────────────
    print("\n[2] train_lstm.py  (TensorFlow - expects tensorflow installed)")
    ok = _run(["training/train_lstm.py",
               "--data-dir", str(mimic_out),
               "--out-dir",  str(model_dir),
               "--epochs",   "1"],
              "train_lstm.py  (1 epoch)")
    results["train_lstm"] = ok
    if ok:
        keras_path = model_dir / "best_model.keras"
        print(f"         -> best_model.keras: {'exists' if keras_path.exists() else 'MISSING'}")

    # 3 ── preprocess_leap.py ─────────────────────────────────────────────────
    print("\n[3] preprocess_leap.py  (numpy/scipy/pandas - no TF needed)")
    ok = _run(["preprocessing/preprocess_leap.py",
               "--ppg-csv",    str(leap_csv),
               "--subject-id", "SUBJ01",
               "--model-dir",  str(model_dir),
               "--out-dir",    str(feat_dir)],
              "preprocess_leap.py")
    results["preprocess_leap"] = ok
    if ok:
        npz = feat_dir / "SUBJ01.npz"
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            print(f"         -> features shape: {d['features'].shape}  "
                  f"(expect (N, 13))")

    # 4 ── run_inference.py  (TensorFlow) ────────────────────────────────────
    print("\n[4] run_inference.py  (TensorFlow - expects tensorflow installed)")
    ok = _run(["inference/run_inference.py",
               "--features-npz", str(feat_dir / "SUBJ01.npz"),
               "--model-dir",    str(model_dir),
               "--out-dir",      str(inf_dir),
               "--timezone",     "UTC"],
              "run_inference.py")
    results["run_inference"] = ok
    if ok:
        bp_csv = inf_dir / "SUBJ01" / "bp_trend.csv"
        if bp_csv.exists():
            n = len(pd.read_csv(bp_csv))
            print(f"         -> {n} prediction rows in bp_trend.csv")

    # 5 ── calibrate_bp.py  (no TF - runs on inference output) ───────────────
    print("\n[5] calibrate_bp.py  (numpy/pandas/sklearn - no TF needed)")
    if results.get("run_inference"):
        ok = _run(["inference/calibrate_bp.py",
                   "--subject-id",    "SUBJ01",
                   "--cuff-csv",      str(cuff_csv),
                   "--inference-dir", str(inf_dir)],
                  "calibrate_bp.py")
        results["calibrate_bp"] = ok
        if ok:
            cal = inf_dir / "SUBJ01" / "calibration.json"
            if cal.exists():
                data = json.loads(cal.read_text())
                sbp_loo = data.get("sbp", {}).get("loo_rmse")
                print(f"         -> calibration.json written  "
                      f"(SBP LOO-RMSE: {sbp_loo:.1f} mmHg)" if sbp_loo else "")
    else:
        print("  [SKIP]  calibrate_bp.py - skipped (run_inference failed)")
        results["calibrate_bp"] = None

    # 6 ── detect_sleep.py  (no TF) ──────────────────────────────────────────
    print("\n[6] detect_sleep.py  (numpy/pandas - no TF needed)")
    ok = _run(["inference/detect_sleep.py",
               "--temp-csv",   str(temp_csv),
               "--subject-id", "SUBJ01",
               "--out-dir",    str(sleep_dir),
               "--timezone",   "UTC"],
              "detect_sleep.py")
    results["detect_sleep"] = ok
    if ok:
        sw = sleep_dir / "SUBJ01" / "sleep_windows.json"
        if sw.exists():
            data  = json.loads(sw.read_text())
            nights = data.get("nights", [])
            print(f"         -> {len(nights)} night(s) detected "
                  f"(source: {nights[0]['source'] if nights else 'none'})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SMOKE TEST SUMMARY")
    print("=" * 62)

    status_map = {True: "[PASS]", False: "[FAIL]", None: "[SKIP]"}
    for name, ok in results.items():
        print(f"  {status_map[ok]}  {name}")

    tf_scripts = {"train_lstm", "run_inference"}
    tf_failed  = any(not results.get(s) for s in tf_scripts)

    print()
    if tf_failed and not imports_ok:
        print("  NOTE: TF scripts failed because tensorflow is not installed.")
        print("        This machine has PyTorch (Phase 5 environment).")
        print("        To run the in-repo TF scripts:  pip install tensorflow")
        print("        To run Phase 5 end-to-end:      add path_a_radha/ to the repo")
    elif all(v for v in results.values() if v is not None):
        print("  All tested scripts PASSED.")
    else:
        print("  Some scripts FAILED - see details above.")

    print("=" * 62)

    # Cleanup (ignore lock errors on Windows)
    try:
        shutil.rmtree(TMP)
    except Exception:
        pass
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
