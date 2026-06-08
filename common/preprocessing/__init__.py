"""Shared PPG preprocessing pipeline.

Design decisions:
- Bandpass: 0.5–4 Hz Butterworth order 4 (Yao et al. 2022, Section C.1)
- Motion rejection: accelerometer magnitude threshold + two SQIs
  (skewness-based per PulseDB/Wang 2023; template-matching per Mejia-Mejia 2021)
- Windowing: 30 s default, 50% overlap (configurable)
- Normalization: z-score or min-max per window (configurable)
- Output tensor shape: (n_windows, window_samples, n_channels)

Calling convention: bandpass_filter and normalize_window propagate NaN/all-zero
naturally; callers should inspect SQI before using windows in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew as scipy_skew


class NormMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


class SQIMethod(str, Enum):
    SKEWNESS = "skewness"   # Wang et al. 2023 / PulseDB §2 SQI criterion
    TEMPLATE = "template"   # Mejia-Mejia 2021 §3.1.3 SQI₅


@dataclass
class PreprocessConfig:
    fs: int = 25
    bp_low_hz: float = 0.5       # Yao et al. 2022 §C.1
    bp_high_hz: float = 4.0      # Yao et al. 2022 §C.1; Nyquist satisfied at 25 Hz
    bp_order: int = 4             # Yao et al. 2022 §C.1
    window_seconds: float = 30.0
    overlap: float = 0.5
    norm_method: NormMethod = NormMethod.ZSCORE
    sqi_method: SQIMethod = SQIMethod.SKEWNESS
    sqi_threshold: float = 0.5
    accel_motion_threshold: float = 0.2  # g; van Hees 2013 rest/active boundary
    motion_use_peak_enmo: bool = True    # True = 95th-pct ENMO; False = mean ENMO


@dataclass
class WindowMetadata:
    participant_id: str
    timestamp_center: object   # pd.Timestamp, or None if not available
    sqi: float
    motion_flag: bool


# ── Bandpass filter ───────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass via second-order sections (sosfiltfilt).

    Minimum signal length = 3 × order × 2 + 1 samples; raises ValueError if shorter.
    Cites: Yao et al. 2022 (IEEE JBHI) §C.1 — order 4, 0.5–4 Hz.
    """
    min_len = 3 * order * 2 + 1
    if len(signal) < min_len:
        raise ValueError(
            f"Signal too short for order-{order} filter: need >={min_len}, got {len(signal)}"
        )
    nyq = 0.5 * fs
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal.astype(np.float64))


# ── SQI helpers ───────────────────────────────────────────────────────────────

def _sqi_skewness(ppg: np.ndarray) -> float:
    """Sigmoid(skewness). Returns 0–1; >0.5 ↔ positive skewness (clean signal).

    Reflectance PPG (systolic peaks up after negation) has positive skewness.
    Ref: Wang et al. 2023 PulseDB §2 — skewness ≥ 0 acceptance criterion.
    """
    if np.all(ppg == ppg[0]):
        return 0.5  # constant input: ambiguous quality
    s = float(scipy_skew(ppg))
    return float(1.0 / (1.0 + np.exp(-s))) if np.isfinite(s) else 0.0


def _sqi_template(ppg: np.ndarray, fs: int) -> float:
    """Mean Pearson r of individual pulses against the ensemble mean pulse.

    Ref: Mejia-Mejia et al. 2021 §3.1.3 SQI₅.
    """
    rng = float(np.ptp(ppg))
    if rng < 1e-8:
        return 0.0
    peaks, _ = find_peaks(ppg, distance=int(0.35 * fs), prominence=0.1 * rng)
    if len(peaks) < 3:
        return 0.0
    pulse_len = int(np.median(np.diff(peaks)))
    if pulse_len < 2:
        return 0.0
    pulses: list[np.ndarray] = []
    for p in peaks[:-1]:
        end = p + pulse_len
        if end > len(ppg):
            break
        pulses.append(
            np.interp(
                np.linspace(0.0, 1.0, pulse_len),
                np.linspace(0.0, 1.0, end - p),
                ppg[p:end],
            )
        )
    if len(pulses) < 2:
        return 0.0
    template = np.mean(pulses, axis=0)
    corrs: list[float] = []
    for p in pulses:
        if np.std(p) < 1e-8 or np.std(template) < 1e-8:
            continue
        r = float(np.corrcoef(p, template)[0, 1])
        if np.isfinite(r):
            corrs.append(max(0.0, r))
    return float(np.mean(corrs)) if corrs else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_sqi(ppg: np.ndarray, method: SQIMethod, fs: int = 25) -> float:
    """Signal quality index for a single (filtered) PPG window. Returns 0–1.

    Threshold of 0.5 separates acceptable/rejected windows by convention
    (configurable via PreprocessConfig.sqi_threshold).
    Returns 0.0 for NaN input.
    """
    if np.any(np.isnan(ppg)):
        return 0.0
    return _sqi_skewness(ppg) if method == SQIMethod.SKEWNESS else _sqi_template(ppg, fs)


def compute_motion_flag(
    accel: np.ndarray,
    threshold: float,
    use_peak: bool = True,
) -> bool:
    """True if window ENMO summary statistic exceeds threshold.

    ENMO = max(0, ||accel||₂ − 1 g).
    use_peak=True  → 95th-percentile ENMO (catches brief bursts, default).
    use_peak=False → mean ENMO (van Hees 2013; better for sustained motion).
    Threshold 0.2 g calibrated for moderate motion (van Hees et al. 2013).
    """
    enmo = np.maximum(0.0, np.linalg.norm(accel, axis=1) - 1.0)
    stat = float(np.percentile(enmo, 95)) if use_peak else float(enmo.mean())
    return bool(stat > threshold)


def normalize_window(window: np.ndarray, method: NormMethod) -> np.ndarray:
    """Per-window normalization. Returns all-zeros for degenerate (constant) input."""
    arr = window.astype(np.float64)
    if method == NormMethod.ZSCORE:
        sigma = arr.std()
        return (arr - arr.mean()) / sigma if sigma >= 1e-8 else np.zeros_like(arr)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi - lo >= 1e-8 else np.zeros_like(arr)


def window_signal(
    ppg: np.ndarray,
    accel: np.ndarray,
    timestamps,
    config: PreprocessConfig,
    participant_id: str,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    """Slide windows over a full recording; apply filter, SQI, and normalization.

    Args:
        ppg:           (N,) raw (negated) green PPG samples
        accel:         (N, 3) accelerometer in g, already resampled to ppg_fs
        timestamps:    sequence of pd.Timestamp, length N (or None)
        config:        PreprocessConfig
        participant_id: string label for metadata

    Returns:
        ppg_windows:   (n_windows, window_samples) bandpass-filtered, normalized
        accel_windows: (n_windows, window_samples, 3) raw accel (not normalized)
        metadata:      list[WindowMetadata] length n_windows
    """
    fs = config.fs
    n_win = int(config.window_seconds * fs)
    step = max(1, int(n_win * (1.0 - config.overlap)))
    n = min(len(ppg), len(accel))

    ppg_wins: list[np.ndarray] = []
    accel_wins: list[np.ndarray] = []
    metas: list[WindowMetadata] = []

    for start in range(0, n - n_win + 1, step):
        end = start + n_win
        ppg_raw = ppg[start:end].astype(np.float64)
        acc_win = accel[start:end].astype(np.float64)

        try:
            ppg_filt = bandpass_filter(ppg_raw, fs, config.bp_low_hz, config.bp_high_hz, config.bp_order)
        except ValueError:
            ppg_filt = ppg_raw.copy()

        sqi = compute_sqi(ppg_filt, config.sqi_method, fs=fs)
        motion = compute_motion_flag(acc_win, config.accel_motion_threshold, use_peak=config.motion_use_peak_enmo)
        ppg_norm = normalize_window(ppg_filt, config.norm_method)

        try:
            ts = timestamps[start + n_win // 2]
        except (IndexError, TypeError, KeyError):
            ts = None

        ppg_wins.append(ppg_norm)
        accel_wins.append(acc_win)
        metas.append(WindowMetadata(
            participant_id=participant_id,
            timestamp_center=ts,
            sqi=sqi,
            motion_flag=motion,
        ))

    if not ppg_wins:
        return (
            np.empty((0, n_win), dtype=np.float64),
            np.empty((0, n_win, 3), dtype=np.float64),
            [],
        )
    return np.stack(ppg_wins), np.stack(accel_wins), metas
