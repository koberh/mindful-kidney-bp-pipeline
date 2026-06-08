"""Evaluation harness shared by both model paths.

Both paths implement BPEstimator and are evaluated identically.
Primary metric (per PI Garrett Ash): within-person BP trend correlation.
Secondary: nocturnal dip magnitude error, population MAE/RMSE, BHS/AAMI.

All compute_* functions accept plain numpy arrays; no pandas dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ── Protocol ──────────────────────────────────────────────────────────────────

class BPEstimator(Protocol):
    def predict(
        self,
        ppg: np.ndarray,    # (window_samples,) or (n_windows, window_samples)
        accel: np.ndarray,  # (window_samples, 3) or (n_windows, window_samples, 3)
        fs: int,
    ) -> dict[str, float | np.ndarray]:
        """Return dict with keys: sbp (mmHg), dbp (mmHg), confidence (0–1)."""
        ...


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class BPMetrics:
    """All evaluation metrics for one model on one test set."""

    # Population-level (mmHg)
    sbp_mae: float
    sbp_rmse: float
    sbp_me: float           # signed bias (predicted − reference)
    dbp_mae: float
    dbp_rmse: float
    dbp_me: float

    # Bland-Altman (mmHg)
    sbp_ba_mean: float      # mean difference
    sbp_ba_loa_upper: float # +1.96 SD
    sbp_ba_loa_lower: float # −1.96 SD
    dbp_ba_mean: float
    dbp_ba_loa_upper: float
    dbp_ba_loa_lower: float

    # BHS grading A/B/C/D
    sbp_bhs: str
    dbp_bhs: str

    # AAMI compliance (ME ≤5 mmHg, SDE ≤8 mmHg)
    sbp_aami_pass: bool
    dbp_aami_pass: bool

    # Within-person trend — PRIMARY METRIC per PI Garrett Ash
    # mean and SD across participants
    within_person_pearson_sbp_mean: float
    within_person_pearson_sbp_std: float
    within_person_spearman_sbp_mean: float
    within_person_pearson_dbp_mean: float
    within_person_pearson_dbp_std: float
    within_person_spearman_dbp_mean: float

    # Nocturnal dip (mmHg; Radha §3.5 definition)
    dip_mae: float          # |predicted_dip − reference_dip|, averaged across participants
    dip_pearson: float      # Pearson r of predicted vs reference dip across participants
    n_participants: int     # number of participants with both predictions and references


# ── BHS grading ───────────────────────────────────────────────────────────────

# BHS criteria (O'Brien et al. 2001, J Hypertens 19:507–516)
_BHS_THRESHOLDS = [
    ("A", 0.60, 0.85, 0.95),
    ("B", 0.50, 0.75, 0.90),
    ("C", 0.40, 0.65, 0.85),
]


def bhs_grade(errors: np.ndarray) -> str:
    """Return BHS grade (A/B/C/D) from array of signed or absolute errors.

    BHS Grade A: ≥60% within 5 mmHg, ≥85% within 10, ≥95% within 15.
    Grade B:     ≥50% within 5 mmHg, ≥75% within 10, ≥90% within 15.
    Grade C:     ≥40% within 5 mmHg, ≥65% within 10, ≥85% within 15.
    Grade D:     below Grade C thresholds.

    Reference: O'Brien et al. 2001, J Hypertens 19:507–516.
    """
    abs_err = np.abs(errors[np.isfinite(errors)])
    if len(abs_err) == 0:
        return "D"
    p5  = float((abs_err <=  5).mean())
    p10 = float((abs_err <= 10).mean())
    p15 = float((abs_err <= 15).mean())
    for grade, t5, t10, t15 in _BHS_THRESHOLDS:
        if p5 >= t5 and p10 >= t10 and p15 >= t15:
            return grade
    return "D"


# ── Bland-Altman ──────────────────────────────────────────────────────────────

def bland_altman(
    pred: np.ndarray,
    ref: np.ndarray,
) -> tuple[float, float, float]:
    """Return (mean_difference, loa_upper, loa_lower).

    Difference = predicted − reference (positive = overestimate).
    Limits of agreement = mean ± 1.96 × SD (Bland & Altman 1986).
    """
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    valid = diff[np.isfinite(diff)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    mean_diff = float(np.mean(valid))
    sd = float(np.std(valid, ddof=1))
    return mean_diff, mean_diff + 1.96 * sd, mean_diff - 1.96 * sd


# ── Within-person trend correlation ──────────────────────────────────────────

def within_person_correlations(
    pred: np.ndarray,
    ref: np.ndarray,
    participant_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson and Spearman r per participant, returned as arrays.

    Returns (pearsons, spearmans) — one value per valid participant.
    Participants with fewer than 3 valid paired samples are skipped.
    """
    from scipy.stats import pearsonr, spearmanr

    unique_pids = np.unique(participant_ids)
    pearsons: list[float] = []
    spearmans: list[float] = []
    for pid in unique_pids:
        mask = participant_ids == pid
        p, r = pred[mask].astype(np.float64), ref[mask].astype(np.float64)
        valid = np.isfinite(p) & np.isfinite(r)
        if valid.sum() < 3:
            continue
        pr, _ = pearsonr(p[valid], r[valid])
        sr, _ = spearmanr(p[valid], r[valid])
        if np.isfinite(pr):
            pearsons.append(float(pr))
        if np.isfinite(sr):
            spearmans.append(float(sr))
    return np.array(pearsons), np.array(spearmans)


# ── Nocturnal dip ─────────────────────────────────────────────────────────────

def nocturnal_dip(
    sbp_pred: np.ndarray,
    sbp_ref: np.ndarray,
    participant_ids: np.ndarray,
    sleep_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and reference nocturnal SBP dip per participant.

    Dip = mean_wake_SBP − mean_sleep_SBP  (positive = nocturnal dip).
    Sleep windows identified by sleep_flags (True = sleep).

    Returns (dip_pred_per_participant, dip_ref_per_participant) — same length,
    one entry per participant with ≥1 wake and ≥1 sleep window.

    Reference: Radha et al. 2019 §3.5 — dip = mean_awake − mean_sleep.
    """
    unique_pids = np.unique(participant_ids)
    dip_pred: list[float] = []
    dip_ref:  list[float] = []
    for pid in unique_pids:
        mask  = participant_ids == pid
        wake  = mask & ~sleep_flags
        sleep = mask & sleep_flags
        if wake.sum() < 1 or sleep.sum() < 1:
            continue
        dp = float(np.nanmean(sbp_pred[wake])) - float(np.nanmean(sbp_pred[sleep]))
        dr = float(np.nanmean(sbp_ref[wake]))  - float(np.nanmean(sbp_ref[sleep]))
        if np.isfinite(dp) and np.isfinite(dr):
            dip_pred.append(dp)
            dip_ref.append(dr)
    return np.array(dip_pred), np.array(dip_ref)


# ── Main metrics function ─────────────────────────────────────────────────────

def compute_metrics(
    sbp_pred: np.ndarray,
    sbp_ref: np.ndarray,
    dbp_pred: np.ndarray,
    dbp_ref: np.ndarray,
    participant_ids: np.ndarray,
    sleep_flags: np.ndarray,
) -> BPMetrics:
    """Compute all evaluation metrics from prediction and reference arrays.

    Args:
        sbp_pred, sbp_ref:    (N,) SBP in mmHg
        dbp_pred, dbp_ref:    (N,) DBP in mmHg
        participant_ids:      (N,) string or int participant labels
        sleep_flags:          (N,) bool — True = sleep window

    Returns:
        BPMetrics with all fields populated (NaN where insufficient data).
    """
    from scipy.stats import pearsonr

    # ── Population-level ─────────────────────────────────────────────────────
    def _pop(pred, ref):
        diff = pred.astype(np.float64) - ref.astype(np.float64)
        valid = np.isfinite(diff)
        if not valid.any():
            return float("nan"), float("nan"), float("nan")
        d = diff[valid]
        return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d**2))), float(np.mean(d))

    sbp_mae, sbp_rmse, sbp_me = _pop(sbp_pred, sbp_ref)
    dbp_mae, dbp_rmse, dbp_me = _pop(dbp_pred, dbp_ref)

    # ── BHS / AAMI ───────────────────────────────────────────────────────────
    sbp_errs = sbp_pred.astype(np.float64) - sbp_ref.astype(np.float64)
    dbp_errs = dbp_pred.astype(np.float64) - dbp_ref.astype(np.float64)

    sbp_bhs = bhs_grade(sbp_errs)
    dbp_bhs = bhs_grade(dbp_errs)

    sbp_sde = float(np.nanstd(sbp_errs))
    dbp_sde = float(np.nanstd(dbp_errs))
    sbp_aami_pass = bool(abs(sbp_me) <= 5.0 and sbp_sde <= 8.0)
    dbp_aami_pass = bool(abs(dbp_me) <= 5.0 and dbp_sde <= 8.0)

    # ── Bland-Altman ─────────────────────────────────────────────────────────
    sbp_ba = bland_altman(sbp_pred, sbp_ref)
    dbp_ba = bland_altman(dbp_pred, dbp_ref)

    # ── Within-person ────────────────────────────────────────────────────────
    sbp_pr, sbp_sr = within_person_correlations(sbp_pred, sbp_ref, participant_ids)
    dbp_pr, dbp_sr = within_person_correlations(dbp_pred, dbp_ref, participant_ids)

    def _safe_stat(arr, fn):
        return float(fn(arr)) if len(arr) >= 1 else float("nan")

    # ── Nocturnal dip ─────────────────────────────────────────────────────────
    dip_p, dip_r = nocturnal_dip(sbp_pred, sbp_ref, participant_ids, sleep_flags)
    if len(dip_p) >= 2:
        dip_mae_val = float(np.mean(np.abs(dip_p - dip_r)))
        dip_pear, _ = pearsonr(dip_p, dip_r)
        dip_pear = float(dip_pear) if np.isfinite(dip_pear) else float("nan")
    else:
        dip_mae_val = float("nan")
        dip_pear    = float("nan")

    n_participants = len(np.unique(participant_ids))

    return BPMetrics(
        sbp_mae=sbp_mae,   sbp_rmse=sbp_rmse,   sbp_me=sbp_me,
        dbp_mae=dbp_mae,   dbp_rmse=dbp_rmse,   dbp_me=dbp_me,
        sbp_ba_mean=sbp_ba[0], sbp_ba_loa_upper=sbp_ba[1], sbp_ba_loa_lower=sbp_ba[2],
        dbp_ba_mean=dbp_ba[0], dbp_ba_loa_upper=dbp_ba[1], dbp_ba_loa_lower=dbp_ba[2],
        sbp_bhs=sbp_bhs,   dbp_bhs=dbp_bhs,
        sbp_aami_pass=sbp_aami_pass, dbp_aami_pass=dbp_aami_pass,
        within_person_pearson_sbp_mean=_safe_stat(sbp_pr, np.mean),
        within_person_pearson_sbp_std=_safe_stat(sbp_pr, np.std),
        within_person_spearman_sbp_mean=_safe_stat(sbp_sr, np.mean),
        within_person_pearson_dbp_mean=_safe_stat(dbp_pr, np.mean),
        within_person_pearson_dbp_std=_safe_stat(dbp_pr, np.std),
        within_person_spearman_dbp_mean=_safe_stat(dbp_sr, np.mean),
        dip_mae=dip_mae_val,
        dip_pearson=dip_pear,
        n_participants=n_participants,
    )


def metrics_to_dict(m: BPMetrics) -> dict[str, float | str | bool | int]:
    """Convert BPMetrics to a flat dict suitable for MLflow logging."""
    import dataclasses
    return dataclasses.asdict(m)
