# Mindful Kidney: Wrist PPG Blood Pressure Pipeline

Research code from the **Mindful Kidney Study** — a Yale IRB-approved prospective pilot trial (Protocol #2000042104) evaluating the Calm Health mindfulness app in adults with chronic kidney disease (CKD; n=26). PI: Garrett Ash, PhD (Yale School of Medicine); Co-I: Menaka Sarav, MD (Yale Medicine Nephrology).

---

## What this is

A full end-to-end pipeline that estimates **within-person blood pressure trends** from wrist photoplethysmography (PPG) collected by ActiGraph LEAP wearable devices, and integrates that signal into automated personalized health reports for CKD trial participants.

**The core problem:** Ambulatory BP monitoring is burdensome for CKD patients. Can a passive wrist wearable detect meaningful BP fluctuations (e.g., nocturnal dip patterns) over a multi-week clinical trial?

---

## Pipeline overview

```
MIMIC-III ICU data (Phase 1–4 training — in this repo)
    └─► data_collection/                search + download ICU PPG+ABP pairs (PhysioNet)
    └─► preprocessing/preprocess_mimic.py   extract 13-feature windows + relative BP labels
    └─► training/train_lstm.py              train Radha et al. 2019 LSTM (TensorFlow)

Aurora-BP ambulatory data (Phase 5 training — scripts not yet in repo, see below)
    └─► path_a_radha/aurora_bp_adapter.py   download + convert 15s wrist PPG snippets → 176-feature NPZ
    └─► path_a_radha/train_phase5.py        train bidirectional LSTM on ambulatory data (PyTorch)

ActiGraph LEAP / CentrePoint participant data (inference)
    └─► preprocessing/preprocess_leap.py    convert raw 100 Hz wrist PPG → 13-feature NPZ
    └─► inference/run_inference.py          LSTM inference → relative BP trend + nocturnal dip
    └─► inference/calibrate_bp.py           ridge regression on cuff readings → absolute mmHg
    └─► inference/detect_sleep.py           wrist temperature → per-night sleep windows

Biometric report
    └─► reporting/compute_hr_hrv_from_ppg.py   derive HR + RMSSD from raw PPG
    └─► reporting/generate_full_report.py      8-page PDF (Calm + actigraphy + BP + AHA staging)
```

---

## Features

**Scripts in this repo (Phases 1–4, TensorFlow):**
- **13-feature PPG representation** — 6 HRV features (mean RR, SDNN, RMSSD, pNN50, mean HR, LF/HF) + 7 PPG morphology features (pulse amplitude, rise time, fall time, pulse width, AUC, dicrotic notch depth/time); extracted in 30-second windows with 50% overlap from raw PPG at 125 Hz (MIMIC) or 25/100 Hz (LEAP).
- **LSTM architecture (TensorFlow/Keras)** — Radha et al. (2019) sequence-to-sequence model: `TimeDistributed Dense(8, ReLU) → LSTM(32) → TimeDistributed Dense(3)`, predicting relative SBP/DBP/MAP at every timestep; amplified MSE loss upweights deviant BP windows to improve nocturnal dip sensitivity.

**Phase 5 implementation (PyTorch, path_a_radha/ — not yet in repo):**
- **176-feature PPG representation** — 174 handcrafted Radha et al. features (5 HRV + 40 Elgendi + 60 Gaussian-fit + 69 Monte-Moreno) + 2 position features; 153 features survive zero-variance filtering on Aurora-BP's 15-second windows.
- **Bidirectional LSTM (PyTorch)** — `Linear(176→8, ReLU) → Dropout(0.4) → BiLSTM(8→16) → Dropout(0.4) → Linear(32→3)`, ~9,600 parameters. Monte-Carlo dropout (30 forward passes) at inference time significantly improves per-participant r vs. deterministic inference (Gal & Ghahramani 2016).
- **Hardened reliability gate** — participant classified `reliable` only if SBP r ≥ 0.40, n ≥ 5 cuff readings, p < 0.05, and bootstrap 95% CI lower > 0. Prevents false-positive classifications seen with r-only gating at small n.

**Shared across all phases:**
- **Wrist temperature sleep detection** — peripheral vasodilation smoothed with 10-minute rolling median; per-night adaptive IQR threshold; fallback to fixed clock window.
- **Automated PDF report** — 8 official pages; integrates Calm Health engagement, actigraphy-derived sleep/steps, HR/HRV, LSTM BP trend with AHA 2017 hypertension staging.
- **CentrePoint V3 API integration** — `preprocess_leap.py` reads gzip-compressed CSV exports from the ActiGraph CentrePoint REST API (100 Hz green PPG, dual-channel interleaved, Unix-ms timestamps); handles ADC saturation, ambient subtraction, per-file sample rate detection.

---

## Five-phase development

| Phase | Training data | Aurora val SBP r | Note |
|-------|--------------|------------------|------|
| 1 | MIMIC-III ICU, CKD-only (n=50) | — | Collapsed to predict-mean; too little data |
| 2 | MIMIC-III ICU, mixed (n=188) | — | Overfit; validation loss diverged |
| 3 | MIMIC-III ICU, mixed (n=188) | −0.10 | Tighter regularization; deployed until 2026-05-28 |
| 4 | MIMIC-III + PulseDB (n=1,015) | not eval'd | +0.04 SBP r on MIMIC val; no ambulatory improvement |
| **5** | **Aurora-BP ambulatory (n=518)** | **+0.48** | **Domain-matched retraining; current deployment** |

**Key finding:** Phases 3 and 4 establish that neither architecture changes nor 5× more ICU training data improve ambulatory performance. The +0.57 SBP r gain in Phase 5 — with identical architecture and hyperparameters — confirms the bottleneck is the **ICU → ambulatory domain shift**, not data volume or model capacity. Phase 5 is the first model whose training distribution matches its deployment distribution.

Within-person SBP Pearson r on Aurora-BP held-out validation (n=103 subjects):

| Model | Median r | Q25 | Q75 |
|-------|----------|-----|-----|
| Phase 3 (ICU-trained) | −0.096 | −0.274 | +0.099 |
| **Phase 5 (Aurora-trained)** | **+0.477** | **+0.235** | **+0.671** |

---

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Phase 5 training path (current deployment — Aurora-BP)

> **Note:** The `path_a_radha/` scripts (Aurora-BP adapter, PyTorch training, head-to-head eval) are not yet included in this public repo. The steps below document the process; the MIMIC/TF training path below is fully runnable from this repo.

**Step 1: Download Aurora-BP data from Zenodo**

By downloading, you agree to the [Aurora-BP Data Use Agreement](https://zenodo.org/records/19099166) (research use only, no redistribution, publications must cite Mieloszyk et al. 2022).

```bash
mkdir -p data/aurora_bp && cd data/aurora_bp

# Metadata files (~35 MB total)
curl -L -o participants.tsv "https://zenodo.org/records/19099166/files/participants.tsv"
curl -L -o measurements_oscillometric.tsv "https://zenodo.org/records/19099166/files/measurements_oscillometric.tsv"

# Waveform archive (4.7 GB — use -C - to resume if interrupted)
curl -L -C - -o measurements_oscillometric.zip \
    "https://zenodo.org/records/19099166/files/measurements_oscillometric.zip"
```

**Step 2: Run the adapter (convert Aurora-BP → training NPZ files)**

```bash
python -m path_a_radha.aurora_bp_adapter \
    --tsv        data/aurora_bp/measurements_oscillometric.tsv \
    --zip        data/aurora_bp/measurements_oscillometric.zip \
    --output-dir data/processed/path_a_aurora
# ~66 min on CPU; output: ~977 .npz files (9.5 MB total)
```

**Step 3: Train Phase 5**

```bash
python -m path_a_radha.train_phase5
# ~53 seconds on CPU; early stop epoch 123/2000
# Output: benchmark/results/path_a_phase5/best_model.pt
```

---

### MIMIC-III training path (Phases 1–4, fully runnable)

For MIMIC-III access, set PhysioNet credentials:
```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password
```

```bash
python data_collection/find_mimic_pleth_abp.py
python data_collection/download_mimic_records.py
python preprocessing/preprocess_mimic.py --data-dir mimic_downloads --out-dir mimic_preprocessed
python training/train_lstm.py --data-dir mimic_preprocessed --out-dir lstm_model
# Output: lstm_model/best_model.keras
```

> MIMIC-trained models produce near-zero correlation on ambulatory data (see Phase 3 row in table above).

---

### Inference on LEAP / CentrePoint participant data

```bash
# Preprocess raw 100 Hz wrist PPG → 13-feature NPZ
python preprocessing/preprocess_leap.py \
    --ppg-csv    "participant_data/SUBJ01_ppg_100hz.csv" \
    --subject-id SUBJ01 \
    --model-dir  lstm_model \
    --out-dir    participant_features/

# LSTM inference
python inference/run_inference.py \
    --features-npz participant_features/SUBJ01.npz \
    --model-dir    lstm_model \
    --timezone     US/Eastern

# Calibrate to absolute mmHg using oscillometric cuff readings
python inference/calibrate_bp.py \
    --subject-id SUBJ01 \
    --cuff-csv   cuff_readings.csv
```

### Derive HR/HRV and generate biometric report

```bash
python reporting/compute_hr_hrv_from_ppg.py \
    --ppg-dir data/ppg_day_files \
    --out-dir data/hr_hrv_output

python reporting/generate_full_report.py \
    --subject-id    SUBJ01 \
    --inference-dir inference_results \
    --data-dir      data/hr_hrv_output \
    --timezone      US/Eastern \
    --study-start   2026-04-28
```

---

## Repository structure

```
data_collection/
    find_mimic_pleth_abp.py         search MIMIC-III for PPG+ABP records
    download_mimic_records.py       download + checkpoint (resumable)
    diagnose_mimic.py               connectivity/auth diagnostic

preprocessing/
    preprocess_mimic.py             MIMIC .npz → 13-feature windows + relative BP labels
    preprocess_leap.py              LEAP/CentrePoint raw PPG CSV → same 13-feature format

training/
    train_lstm.py                   Phases 1–4 MIMIC training (TensorFlow)

inference/
    run_inference.py                LSTM inference; nocturnal dip analysis; JSON + CSV output
    calibrate_bp.py                 Absolute mmHg calibration (ridge regression); LOO RMSE
    detect_sleep.py                 Sleep onset/wake from wrist temperature (vasodilation)

reporting/
    compute_hr_hrv_from_ppg.py      HR + RMSSD from raw 100 Hz PPG; 1-min HR / 5-min HRV windows
    generate_full_report.py         8-page PDF biometric report

path_a_radha/                       [not yet in repo — Phase 5 PyTorch scripts]
    aurora_bp_adapter.py            Aurora-BP → 176-feature training NPZ
    train_phase5.py                 Phase 5 training (PyTorch, bidirectional LSTM)
    evaluate_phase3_vs_phase5.py    2×2 head-to-head evaluation
    calibrate_from_predictions.py   Hardened reliability gate
    finetune_participant.py         LOO per-participant fine-tune prototype
    run_centrepoint_inference.py    CentrePoint inference with Phase 5 checkpoint
    model.py                        Shared RadhaLSTM architecture
```

---

## Data

No participant data is included in this repository.

**Aurora-BP (Phase 5 training data):** Available at [Zenodo record 19099166](https://zenodo.org/records/19099166) under a data use agreement requiring research-only use, no redistribution, and citation of Mieloszyk et al. 2022. Ambulatory phase: 548 subjects, wrist MAX30101 PPG + oscillometric cuff readings every ~30 min over 24 hours. Download size: 4.74 GB.

**MIMIC-III (Phase 1–4 training data):** Requires a credentialed PhysioNet account with completed CITI training. Access at [physionet.org/content/mimiciii](https://physionet.org/content/mimiciii/).

---

## Limitations

- **Not a BP measurement device.** Phase 5 predicts *relative* BP trend and must be anchored to oscillometric cuff readings. The reliability gate suppresses results for participants who cannot be tracked.
- **Aurora-BP is not CKD-specific.** The training cohort is general-population ambulatory adults (~42% with elevated BP, no kidney disease enrichment). CKD-specific physiology is not represented.
- **Performance ceiling.** Median SBP r ≈ 0.4–0.6 is consistent with independent reports (Mieloszyk et al. 2022, Radha et al. 2019) and likely reflects the physical limits of wrist green PPG for BP estimation, not model inadequacy.
- **Per-participant fine-tuning requires LOO evaluation.** A prototype LOO ridge fine-tune on n=12 cuff readings showed in-sample r = +0.64 but LOO r = −0.30 — a classic overfit artifact. Any deployed per-participant adaptation must report leave-one-out metrics.

---

## References

- Radha, M. et al. (2019). Estimating blood pressure trends and the nocturnal dip from photoplethysmography. *Physiological Measurement*, 40(2), 025006.
- Mieloszyk, R. J. et al. (2022). A comparison of wearable tonometry, photoplethysmography, and electrocardiography for cuffless measurement of blood pressure in an ambulatory setting. *IEEE Journal of Biomedical and Health Informatics*, 26(7), 2864–2875. [Aurora-BP dataset]
- Johnson, A. E. W. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML*.

---

*Research code — not a medical device. All BP values are research estimates not intended for clinical use.*
