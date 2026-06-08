# Mindful Kidney: Wrist PPG Blood Pressure Pipeline

Research code from the **Mindful Kidney Study** — a Yale IRB-approved prospective pilot trial (Protocol #2000042104) evaluating the Calm Health mindfulness app in adults with chronic kidney disease (CKD; n=26). PI: Garrett Ash, PhD (Yale School of Medicine); Co-I: Menaka Sarav, MD (Yale Medicine Nephrology).

---

## What this is

A full end-to-end pipeline that estimates **within-person blood pressure trends** from wrist photoplethysmography (PPG) collected by ActiGraph LEAP wearable devices, and integrates that signal into automated personalized health reports for CKD trial participants.

**The core problem:** Ambulatory BP monitoring is burdensome for CKD patients. Can a passive wrist wearable detect meaningful BP fluctuations (e.g., nocturnal dip patterns) over a multi-week clinical trial?

---

## Pipeline overview

```
MIMIC-III waveform data (training)
    └─► data_collection/          search + download ICU PPG+ABP pairs
    └─► preprocessing/preprocess_mimic.py   extract 13-feature windows + relative BP labels
    └─► training/train_lstm.py              train Radha et al. 2019 LSTM

ActiGraph LEAP participant data (inference)
    └─► preprocessing/preprocess_leap.py    convert raw 100 Hz PPG → same 13 features
    └─► inference/run_inference.py          LSTM inference → relative BP trend + nocturnal dip
    └─► inference/calibrate_bp.py           ridge regression on home cuff readings → absolute mmHg
    └─► inference/detect_sleep.py           wrist temperature → per-night sleep windows

Biometric report
    └─► reporting/compute_hr_hrv_from_ppg.py   derive HR + RMSSD from raw PPG
    └─► reporting/generate_full_report.py      8-page PDF report (Calm + actigraphy + BP + AHA staging)
```

---

## Features

- **13-feature PPG representation** — 6 HRV features (mean RR, SDNN, RMSSD, pNN50, mean HR, LF/HF) + 7 PPG morphology features (pulse amplitude, rise time, fall time, pulse width, AUC, dicrotic notch depth/time) extracted in 30-second windows with 50% overlap
- **LSTM architecture** — Radha et al. (2019) sequence-to-sequence model: `Dense(8, relu) → LSTM(32) → Dense(3)`, predicting relative SBP/DBP/MAP at every timestep; weighted MSE loss upweights deviant BP values to improve nocturnal dip sensitivity
- **Domain mismatch resolution** — Phase 3 (trained on MIMIC-III ICU corpus) showed near-zero correlation on ambulatory data; Phase 5 (retrained on Microsoft Aurora-BP, n=518 ambulatory subjects) improved median SBP r from −0.10 → +0.48 on held-out ambulatory validation set
- **Wrist temperature sleep detection** — peripheral vasodilation signal smoothed with 10-minute rolling median; per-night adaptive threshold on IQR; fallback to fixed clock window
- **Automated PDF report** — 8 official pages + weekly daily snapshots; integrates Calm Health app engagement, actigraphy-derived sleep/steps, PPG-derived HR/HRV, LSTM BP trend with AHA 2017 hypertension staging; delivered to each participant as a tailored health summary
- **CentrePoint V3 API integration** — `preprocess_leap.py` reads raw gzip-compressed CSV exports from the ActiGraph CentrePoint REST API (100 Hz green PPG, dual-channel interleaved, Unix-ms timestamps); handles ADC saturation flagging, ambient subtraction, and per-file sample rate detection

---

## Five-phase development

| Phase | Training data | Val SBP r | Note |
|-------|--------------|-----------|------|
| 1–2   | MIMIC-III ICU (n=188 subjects) | — | Baseline architecture |
| 3     | MIMIC-III ICU (n=188) | −0.10 | Deployed on LEAP data; near-mean predictions |
| 4     | MIMIC-III + PulseDB augmentation (n=1,015) | +0.04 gain | Data augmentation alone insufficient |
| **5** | **Aurora-BP ambulatory (n=518)** | **+0.48** | Domain-matched retraining; current deployment |

Key finding: the gap between phases 3 and 5 confirms a **training-domain mismatch hypothesis** — ICU-trained models learn BP variation driven by acute illness and vasopressor response, not the physiological variation present in ambulatory wrist PPG.

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

For MIMIC-III data access, set your PhysioNet credentials as environment variables:
```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password
```

### 1. Download MIMIC-III training data

```bash
# Find records with both PPG and arterial BP
python data_collection/find_mimic_pleth_abp.py

# Download filtered records (interactive confirmation required)
python data_collection/download_mimic_records.py
```

### 2. Preprocess and train

```bash
python preprocessing/preprocess_mimic.py --data-dir mimic_downloads --out-dir mimic_preprocessed
python training/train_lstm.py --data-dir mimic_preprocessed --out-dir lstm_model
```

### 3. Run inference on LEAP participant data

```bash
# Preprocess raw PPG
python preprocessing/preprocess_leap.py \
    --ppg-csv  "participant_data/SUBJ01_ppg_100hz.csv" \
    --subject-id SUBJ01 \
    --model-dir lstm_model \
    --out-dir   participant_features/

# LSTM inference
python inference/run_inference.py \
    --features-npz participant_features/SUBJ01.npz \
    --model-dir    lstm_model \
    --timezone     US/Eastern

# Calibrate to absolute mmHg using home cuff readings
python inference/calibrate_bp.py \
    --subject-id SUBJ01 \
    --cuff-csv   cuff_readings.csv
```

### 4. Derive HR/HRV and generate report

```bash
python reporting/compute_hr_hrv_from_ppg.py \
    --ppg-dir data/ppg_day_files \
    --out-dir data/hr_hrv_output

python reporting/generate_full_report.py \
    --subject-id   SUBJ01 \
    --inference-dir inference_results \
    --data-dir      data/hr_hrv_output \
    --timezone      US/Eastern \
    --study-start   2026-04-28
```

---

## Repository structure

```
data_collection/
    find_mimic_pleth_abp.py     search MIMIC-III for PPG+ABP records
    download_mimic_records.py   download + checkpoint (resumable)
    diagnose_mimic.py           connectivity/auth diagnostic

preprocessing/
    preprocess_mimic.py         MIMIC .npz → 13-feature windows + relative BP labels
    preprocess_leap.py          LEAP raw PPG CSV → same 13-feature format

training/
    train_lstm.py               Radha et al. LSTM; WeightedMSE loss; eval + checkpointing

inference/
    run_inference.py            LSTM inference; nocturnal dip analysis; JSON + CSV output
    calibrate_bp.py             Ridge regression (Model 2) to absolute mmHg; LOO RMSE
    detect_sleep.py             Sleep onset/wake from wrist temperature (vasodilation)

reporting/
    compute_hr_hrv_from_ppg.py  HR + RMSSD from raw 100 Hz PPG; 1-min HR / 5-min HRV windows
    generate_full_report.py     8-page PDF biometric report
```

---

## Data

No participant data is included in this repository. MIMIC-III access requires credentialed PhysioNet account with completed CITI training. Aurora-BP is available through Microsoft Research.

---

## References

- Radha, M. et al. (2019). Estimating blood pressure trends and the nocturnal dip from photoplethysmography. *Physiological Measurement*, 40(2), 025006.
- Johnson, A. E. W. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.
- Zhu, T. et al. (2022). Feasibility of cuffless ambulatory blood pressure monitoring using wrist photoplethysmography (Aurora-BP). Microsoft Research.

---

*Research code — not a medical device. All BP values are research estimates not intended for clinical use.*
