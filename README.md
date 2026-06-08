# Mindful Kidney: Wrist PPG Blood Pressure Pipeline

Research code from the **Mindful Kidney Study** — a Yale IRB-approved prospective pilot trial (Protocol #2000042104) evaluating the Calm Health mindfulness app in adults with chronic kidney disease (CKD; n=26). PI: Garrett Ash, PhD (Yale School of Medicine); Co-I: Menaka Sarav, MD (Yale Medicine Nephrology).

---

## What this is

A full end-to-end pipeline that estimates **within-person blood pressure trends** from wrist photoplethysmography (PPG) collected by ActiGraph LEAP wearable devices, and integrates that signal into automated personalized health reports for CKD trial participants.

**The core problem:** Ambulatory BP monitoring is burdensome for CKD patients. Can a passive wrist wearable detect meaningful BP fluctuations (e.g., nocturnal dip patterns) over a multi-week clinical trial?

---

## Pipeline overview

```
Aurora-BP ambulatory data (Phase 5 training — current deployment)
    └─► aurora_bp_adapter.py            download + convert 15s wrist PPG snippets → 176-feature NPZ
    └─► train_phase5.py                 train bidirectional LSTM on ambulatory wrist data

MIMIC-III ICU data (Phase 1–4 training — reference only)
    └─► data_collection/                search + download ICU PPG+ABP pairs (PhysioNet)
    └─► preprocessing/preprocess_mimic.py   extract feature windows + relative BP labels

ActiGraph LEAP / CentrePoint participant data (inference)
    └─► preprocessing/preprocess_leap.py    convert raw 100 Hz wrist PPG → 176 features
    └─► inference/run_inference.py          LSTM inference → relative BP trend + nocturnal dip
    └─► inference/calibrate_bp.py           ridge regression on cuff readings → absolute mmHg
    └─► inference/detect_sleep.py           wrist temperature → per-night sleep windows

Biometric report
    └─► reporting/compute_hr_hrv_from_ppg.py   derive HR + RMSSD from raw PPG
    └─► reporting/generate_full_report.py      8-page PDF (Calm + actigraphy + BP + AHA staging)
```

---

## Features

- **176-feature PPG representation** — 174 handcrafted features following Radha et al. (2019): 5 HRV features, 40 Elgendi morphology features, 60 Gaussian-fit features, 69 Monte-Moreno features; plus 2 position features (window start time, gap to previous measurement). After zero-variance filtering on Aurora-BP's 15-second windows, 153 features survive into training.
- **LSTM architecture (PyTorch)** — Radha et al. (2019) bidirectional sequence-to-sequence model: `Linear(176→8, ReLU) → Dropout(0.4) → BiLSTM(8→16) → Dropout(0.4) → Linear(32→3)`, predicting relative SBP/DBP/MAP at every timestep; amplified MSE loss upweights deviant BP windows to improve nocturnal dip sensitivity. ~9,600 parameters total.
- **Monte-Carlo dropout inference** — 30 stochastic forward passes averaged at inference time (Gal & Ghahramani 2016); meaningfully improves single-participant r vs. deterministic inference.
- **Domain mismatch resolution** — Phase 3 (trained on MIMIC-III ICU corpus) showed r = −0.10 on ambulatory data. Phase 5 (retrained on Aurora-BP, n=518 ambulatory subjects) improved median SBP r from −0.10 → +0.48 and doubled range compression (0.32 → 0.67) on the same held-out ambulatory validation set. The architecture, feature extractor, and hyperparameters are identical across phases — the only change is the training domain.
- **Hardened reliability gate** — a participant is classified `reliable` only if all hold: SBP r ≥ 0.40, n ≥ 5 cuff readings, two-sided p < 0.05, and bootstrap 95% CI lower bound > 0. Participants below this threshold have their BP panel suppressed in the report rather than shown as a misleading flat line.
- **Wrist temperature sleep detection** — peripheral vasodilation signal smoothed with 10-minute rolling median; per-night adaptive threshold on IQR; fallback to fixed clock window.
- **Automated PDF report** — 8 official pages + weekly daily snapshots; integrates Calm Health app engagement, actigraphy-derived sleep/steps, PPG-derived HR/HRV, LSTM BP trend with AHA 2017 hypertension staging.
- **CentrePoint V3 API integration** — `preprocess_leap.py` reads raw gzip-compressed CSV exports from the ActiGraph CentrePoint REST API (100 Hz green PPG, dual-channel interleaved, Unix-ms timestamps); handles ADC saturation flagging, ambient subtraction, and per-file sample rate detection.

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

### Phase 5 training path (current deployment)

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

Verify the download:
```bash
python -c "import zipfile; z=zipfile.ZipFile('data/aurora_bp/measurements_oscillometric.zip'); print(f'OK: {len(z.namelist())} files')"
# Expected: OK: ~12000+ files
```

**Step 2: Run the adapter (convert Aurora-BP → training NPZ files)**

```bash
python -m path_a_radha.aurora_bp_adapter \
    --tsv        data/aurora_bp/measurements_oscillometric.tsv \
    --zip        data/aurora_bp/measurements_oscillometric.zip \
    --output-dir data/processed/path_a_aurora
# Runtime: ~66 min on CPU. Output: ~977 .npz files (9.5 MB total)
```

The adapter filters to `optical_quality ≥ 0.65`, downsamples 500 Hz → 100 Hz, extracts 176 features per window, and writes one `.npz` per participant-day.

**Step 3: Train Phase 5**

```bash
python -m path_a_radha.train_phase5
# Runtime: ~53 seconds on CPU (early stop at epoch 123/2000)
# Output: benchmark/results/path_a_phase5/best_model.pt
```

**Step 4: Evaluate (Phase 3 vs Phase 5 head-to-head)**

```bash
python -m path_a_radha.evaluate_phase3_vs_phase5
# Output: benchmark/results/phase3_vs_phase5/comparison_table.md
```

---

### Inference on LEAP / CentrePoint participant data

```bash
# Preprocess raw 100 Hz wrist PPG
python preprocessing/preprocess_leap.py \
    --ppg-csv    "participant_data/SUBJ01_ppg_100hz.csv" \
    --subject-id SUBJ01 \
    --out-dir    participant_features/

# LSTM inference (Monte-Carlo dropout, 30 passes)
python inference/run_inference.py \
    --features-npz participant_features/SUBJ01.npz \
    --checkpoint   benchmark/results/path_a_phase5/best_model.pt \
    --timezone     US/Eastern

# Calibrate to absolute mmHg using oscillometric cuff readings
python inference/calibrate_bp.py \
    --subject-id SUBJ01 \
    --cuff-csv   cuff_readings.csv
# Reports reliability gate classification: reliable / weak / unreliable
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

### MIMIC-III training path (Phases 1–4, reference)

For MIMIC-III access, set PhysioNet credentials:
```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password
```

```bash
# Find records with both PPG and arterial BP
python data_collection/find_mimic_pleth_abp.py
python data_collection/download_mimic_records.py

# Preprocess and train (Phase 3-equivalent)
python preprocessing/preprocess_mimic.py --data-dir mimic_downloads --out-dir mimic_preprocessed
python training/train_lstm.py --data-dir mimic_preprocessed --out-dir lstm_model
```

Note: MIMIC-trained models produce near-zero correlation on ambulatory wrist data. See five-phase development table above.

---

## Repository structure

```
data_collection/
    find_mimic_pleth_abp.py         search MIMIC-III for PPG+ABP records
    download_mimic_records.py       download + checkpoint (resumable)
    diagnose_mimic.py               connectivity/auth diagnostic

preprocessing/
    preprocess_mimic.py             MIMIC .npz → 176-feature windows + relative BP labels
    preprocess_leap.py              LEAP/CentrePoint raw PPG CSV → same 176-feature format

training/
    train_lstm.py                   Phase 1–4 MIMIC training entry point

path_a_radha/
    aurora_bp_adapter.py            Aurora-BP zip → per-participant-day .npz (Phase 5 data prep)
    train_phase5.py                 Phase 5 training entry point (Aurora-BP)
    evaluate_phase3_vs_phase5.py    2×2 head-to-head evaluation (MIMIC val + Aurora val)
    calibrate_from_predictions.py   Hardened reliability gate (r + p + CI + n)
    finetune_participant.py         LOO per-participant ridge fine-tune (prototype; not deployed)
    run_centrepoint_inference.py    CentrePoint API inference with Phase 3 or Phase 5 checkpoint
    model.py                        Shared RadhaLSTM architecture (PyTorch)

inference/
    run_inference.py                LSTM inference; nocturnal dip analysis; JSON + CSV output
    calibrate_bp.py                 Absolute mmHg calibration; reliability gate; LOO RMSE
    detect_sleep.py                 Sleep onset/wake from wrist temperature (vasodilation)

reporting/
    compute_hr_hrv_from_ppg.py      HR + RMSSD from raw 100 Hz PPG; 1-min HR / 5-min HRV windows
    generate_full_report.py         8-page PDF biometric report
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
