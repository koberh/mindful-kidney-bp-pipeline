"""Feature extraction for Path A — Radha et al. 2019.

Three feature families totalling 195 features, extracted from one 5-minute
filtered PPG window + simultaneous accelerometer window:

  Activity      (21): ENMO statistics in 4 time segments + rest probability
  HRV            (5): multi-scale entropy at temporal scales 6–10 on RR series
  Morphology   (169):
    Elgendi       (40): per-pulse amplitude/width/area + PPG'/PPG'' wave features
    Gaussian      (60): 4-Gaussian pulse decomposition parameters and ratios
    Monte-Moreno  (69): spectral, systolic-complex, non-linear, statistical

Output ordering:
  [0:21]    activity
  [21:26]   HRV
  [26:66]   Elgendi
  [66:126]  Gaussian
  [126:195] Monte-Moreno

Primary references:
  Radha et al. 2019, §3.2 — three morphology sub-families
  Elgendi et al. 2012, Curr. Cardiol. Rev. 8:14-25 — peak feature taxonomy
  van Hees et al. 2013, J Appl Physiol 115:1220 — ENMO threshold 0.04 g
  Richman & Moorman 2000, Am J Physiol 278:H2039 — sample entropy / MSE
  Monte-Moreno 2011, Comput. Biol. Med. 41:1092 — Gaussian decomposition
"""
from __future__ import annotations

import warnings

import antropy  # type: ignore[import-untyped]
import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
import scipy.stats as st

# ── Feature family counts (must sum to 195) ──────────────────────────────────
N_ACTIVITY: int = 21
N_HRV: int = 5
N_ELGENDI: int = 40
N_GAUSSIAN: int = 60
N_MONTE_MORENO: int = 69
N_TOTAL: int = N_ACTIVITY + N_HRV + N_ELGENDI + N_GAUSSIAN + N_MONTE_MORENO
assert N_TOTAL == 195

# ── Constants ────────────────────────────────────────────────────────────────
_ENMO_REST_G: float = 0.04       # van Hees et al. 2013 J Appl Physiol 115:1220
_MIN_PEAK_DIST_S: float = 0.35   # shortest allowable RR interval (≈170 bpm max)
_MAX_GAUSS_PULSES: int = 50      # cap for Gaussian fitting (speed)
_APP_MAX_N: int = 512            # max samples for O(n²) entropy methods


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


def _agg(v: np.ndarray) -> np.ndarray:
    """[mean, std] over finite values; [nan, nan] if fewer than 2 finite."""
    fin = v[np.isfinite(v)]
    if len(fin) < 2:
        return _nan(2)
    return np.array([float(np.mean(fin)), float(np.std(fin, ddof=1))], dtype=np.float64)


def _safe(x: object) -> float:
    try:
        f = float(x)  # type: ignore[arg-type]
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection and pulse segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _detect_peaks(ppg: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate systolic peak indices and pulse onset indices.

    Onset of pulse i = minimum of ppg between peaks[i-1] and peaks[i].
    Returns (peaks, onsets); either may be empty on failure.
    """
    min_dist = max(1, int(_MIN_PEAK_DIST_S * fs))
    sig_range = float(np.ptp(ppg))
    prominence = max(1e-9, 0.10 * sig_range)
    peaks_arr, _ = sig.find_peaks(ppg, distance=min_dist, prominence=prominence)
    if len(peaks_arr) < 2:
        return peaks_arr, np.empty(0, dtype=np.intp)
    onsets = np.empty(len(peaks_arr), dtype=np.intp)
    for i, pk in enumerate(peaks_arr):
        lo = int(peaks_arr[i - 1]) if i > 0 else 0
        onsets[i] = lo + int(np.argmin(ppg[lo : int(pk)]))
    return peaks_arr, onsets


def _segment_pulses(ppg: np.ndarray, onsets: np.ndarray) -> list[np.ndarray]:
    """Segment PPG into individual pulses: onset[i] to onset[i+1]."""
    pulses: list[np.ndarray] = []
    for i in range(len(onsets) - 1):
        pulse = ppg[int(onsets[i]) : int(onsets[i + 1])].astype(np.float64)
        if len(pulse) >= 6:
            pulses.append(pulse)
    return pulses


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _sum4g(x: np.ndarray, *p: float) -> np.ndarray:
    """Sum of 4 Gaussians; p = [A1,mu1,s1, A2,mu2,s2, A3,mu3,s3, A4,mu4,s4]."""
    y = np.zeros_like(x, dtype=np.float64)
    for k in range(4):
        A, mu, s = p[3 * k], p[3 * k + 1], p[3 * k + 2]
        if s <= 0.0:
            return np.full_like(x, np.inf)
        y += A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return y


def _fit_gaussians(pulse: np.ndarray) -> np.ndarray | None:
    """
    Fit normalized pulse to sum of 4 Gaussians.
    Returns (12,) params [A, mu, sigma × 4] in normalized coords, or None.
    Normalized: x in [0,1], y in [0,1].
    """
    if len(pulse) < 10:
        return None
    prange = float(pulse.max() - pulse.min())
    if prange < 1e-10:
        return None
    x = np.linspace(0.0, 1.0, len(pulse))
    y = (pulse - float(pulse.min())) / prange
    p0 = [v for k in range(4) for v in (0.5 / 4, (k + 0.5) / 4.0, 0.08)]
    lo = [0.0, 0.0, 0.01] * 4
    hi = [2.0, 1.0, 0.45] * 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = opt.curve_fit(
                _sum4g, x, y, p0=p0, bounds=(lo, hi), maxfev=300,
            )
            return np.asarray(popt, dtype=np.float64)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — Activity
# ─────────────────────────────────────────────────────────────────────────────

def extract_activity_features(
    accel: np.ndarray,
    fs: int,
    rest_prob: float,
) -> np.ndarray:
    """
    21 activity features from tri-axial accelerometer.

    ENMO statistics (mean, std, skew, kurtosis, IQR) over 4 equal time
    segments = 20 features, plus pre-computed rest probability = 1 feature.

    accel: (n_samples, 3), tri-axial accelerometer in g.
    rest_prob: smoothed rest probability for this window [0, 1]; computed at
               day level by caller via compute_rest_probabilities() (Q3 resolution).
    """
    out = _nan(N_ACTIVITY)
    try:
        enmo = np.maximum(0.0, np.linalg.norm(accel, axis=1) - 1.0)
        n = len(enmo)
        seg = max(1, n // 4)
        for s in range(4):
            e = enmo[s * seg : min((s + 1) * seg, n)]
            b = s * 5
            out[b + 0] = float(np.mean(e))
            out[b + 1] = float(np.std(e, ddof=1)) if len(e) > 1 else np.nan
            out[b + 2] = _safe(st.skew(e))
            out[b + 3] = _safe(st.kurtosis(e))
            out[b + 4] = float(np.percentile(e, 75) - np.percentile(e, 25))
        out[20] = float(np.clip(rest_prob, 0.0, 1.0))
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — HRV
# ─────────────────────────────────────────────────────────────────────────────

def extract_hrv_features(ppg: np.ndarray, fs: int) -> np.ndarray:
    """
    5 HRV features: multi-scale entropy at temporal scales 6–10.

    Coarse-grains the RR interval series by factor k, then computes sample
    entropy (order=2). Richman & Moorman 2000; Radha 2019 §3.2.2.
    """
    out = _nan(N_HRV)
    try:
        peaks, _ = _detect_peaks(ppg, fs)
        if len(peaks) < 25:
            return out
        rr = np.diff(peaks.astype(np.float64)) * (1000.0 / fs)
        rr = rr[(rr > 300) & (rr < 2000)]  # 30–200 bpm
        if len(rr) < 20:
            return out
        for i, scale in enumerate(range(6, 11)):
            n_cg = len(rr) // scale
            if n_cg < 5:
                continue
            cg = rr[: n_cg * scale].reshape(n_cg, scale).mean(axis=1)
            if float(np.std(cg, ddof=1)) <= 0:
                continue
            try:
                out[i] = _safe(antropy.sample_entropy(cg, order=2))
            except Exception:
                pass
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — Elgendi morphology
# ─────────────────────────────────────────────────────────────────────────────

def extract_elgendi_features(ppg: np.ndarray, fs: int) -> np.ndarray:
    """
    40 Elgendi-family morphology features.

    20 per-pulse scalars each aggregated to (mean, std) = 40 total:
      pulse-level (11): systolic/diastolic amplitude, their ratio, widths, areas, timing
      PPG' first derivative (3): max slope, max-slope timing, min slope
      PPG'' second derivative (6): a/b/c/d waves, b/a ratio, (b-c-d)/a ratio

    Elgendi et al. 2012, Curr. Cardiol. Rev. 8:14-25.
    """
    out = _nan(N_ELGENDI)
    try:
        peaks, onsets = _detect_peaks(ppg, fs)
        if len(peaks) < 3 or len(onsets) < 2:
            return out
        pulses = _segment_pulses(ppg, onsets)
        if len(pulses) < 2:
            return out

        ppg1 = np.gradient(ppg)
        ppg2 = np.gradient(ppg1)
        n_p = len(pulses)

        bufs: dict[str, np.ndarray] = {k: _nan(n_p) for k in (
            "sys_amp", "dia_amp", "syd_ratio", "pw", "sys_w", "dia_w",
            "p_area", "sys_area", "dia_area", "rise_t", "fall_t",
            "max_sl", "max_sl_t", "min_sl",
            "a_w", "b_w", "c_w", "d_w", "ba_r", "bcd_a_r",
        )}

        for i, pulse in enumerate(pulses):
            np_ = len(pulse)
            onset_v = float(pulse[0])
            sys_i = int(np.argmax(pulse[: max(1, np_ // 2)]))
            sys_v = float(pulse[sys_i])
            amp = sys_v - onset_v

            bufs["sys_amp"][i] = amp
            bufs["pw"][i] = np_ / fs
            bufs["sys_w"][i] = sys_i / fs
            bufs["dia_w"][i] = (np_ - sys_i) / fs
            bufs["rise_t"][i] = sys_i / fs
            bufs["fall_t"][i] = (np_ - sys_i) / fs

            bl = onset_v
            bufs["p_area"][i] = float(np.trapezoid(pulse - bl)) / fs
            bufs["sys_area"][i] = float(np.trapezoid(pulse[: sys_i + 1] - bl)) / fs
            bufs["dia_area"][i] = float(np.trapezoid(pulse[sys_i:] - bl)) / fs

            # Diastolic peak: max in diastolic phase, must be below systolic peak
            if np_ - sys_i > 4:
                dia_seg = pulse[sys_i:]
                d_val = float(dia_seg[int(np.argmax(dia_seg))])
                if d_val < sys_v:
                    bufs["dia_amp"][i] = d_val - onset_v
                    if bufs["dia_amp"][i] > 0 and amp > 0:
                        bufs["syd_ratio"][i] = amp / bufs["dia_amp"][i]

            # PPG' and PPG'' over the pulse interval
            g0 = int(onsets[i]) if i < len(onsets) else 0
            g1 = g0 + np_
            if g1 <= len(ppg1):
                sl = ppg1[g0:g1]
                bufs["max_sl"][i] = float(sl.max())
                bufs["max_sl_t"][i] = float(int(np.argmax(sl))) / max(np_, 1)
                bufs["min_sl"][i] = float(sl.min())

                sl2 = ppg2[g0:g1]
                if len(sl2) >= 8:
                    pp2, _ = sig.find_peaks(sl2)
                    mp2, _ = sig.find_peaks(-sl2)
                    if len(pp2) >= 1:
                        bufs["a_w"][i] = float(sl2[pp2[0]])
                    if len(mp2) >= 1:
                        bufs["b_w"][i] = float(sl2[mp2[0]])
                    if len(pp2) >= 2:
                        bufs["c_w"][i] = float(sl2[pp2[1]])
                    if len(mp2) >= 2:
                        bufs["d_w"][i] = float(sl2[mp2[1]])
                    a = bufs["a_w"][i]
                    b = bufs["b_w"][i]
                    c = bufs["c_w"][i]
                    d2 = bufs["d_w"][i]
                    if np.isfinite(a) and abs(a) > 1e-12:
                        if np.isfinite(b):
                            bufs["ba_r"][i] = b / a
                        if all(np.isfinite(v) for v in (b, c, d2)):
                            bufs["bcd_a_r"][i] = (b - c - d2) / a

        idx = 0
        for key in (
            "sys_amp", "dia_amp", "syd_ratio", "pw", "sys_w", "dia_w",
            "p_area", "sys_area", "dia_area", "rise_t", "fall_t",
            "max_sl", "max_sl_t", "min_sl",
            "a_w", "b_w", "c_w", "d_w", "ba_r", "bcd_a_r",
        ):
            out[idx : idx + 2] = _agg(bufs[key])
            idx += 2
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — Gaussian decomposition
# ─────────────────────────────────────────────────────────────────────────────

def extract_gaussian_features(ppg: np.ndarray, fs: int) -> np.ndarray:
    """
    60 Gaussian decomposition features.

    4-Gaussian fit per pulse → 12 raw params × 2 stats = 24.
    Derived (36 total):
      4 normalized amplitudes × 2 stats      =  8
      3 position differences × 2 stats       =  6
      3 width ratios × 2 stats               =  6
      fit RMSE (2) + R² (2)                  =  4
      4 Gaussian areas × 2 stats             =  8
      dominant-area fraction × 2 stats       =  2
      early-area fraction × 2 stats          =  2

    Monte-Moreno 2011, Comput. Biol. Med. 41:1092.
    """
    out = _nan(N_GAUSSIAN)
    try:
        _, onsets = _detect_peaks(ppg, fs)
        pulses = _segment_pulses(ppg, onsets)
        if len(pulses) < 3:
            return out

        if len(pulses) > _MAX_GAUSS_PULSES:
            rng = np.random.default_rng(42)
            sel = np.sort(rng.choice(len(pulses), _MAX_GAUSS_PULSES, replace=False))
            pulses = [pulses[j] for j in sel]

        n_f = len(pulses)
        raw = np.full((n_f, 12), np.nan)
        rmse = _nan(n_f)
        r2 = _nan(n_f)
        sqrt2pi = float(np.sqrt(2.0 * np.pi))

        for i, pulse in enumerate(pulses):
            p = _fit_gaussians(pulse)
            if p is None:
                continue
            raw[i] = p
            n = len(pulse)
            x = np.linspace(0.0, 1.0, n)
            prange = float(pulse.max() - pulse.min())
            if prange < 1e-10:
                continue
            y = (pulse - float(pulse.min())) / prange
            resid = y - _sum4g(x, *p)
            rmse[i] = float(np.sqrt(np.mean(resid**2)))
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((y - float(y.mean())) ** 2))
            r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        idx = 0
        # 12 raw params × 2 = 24
        for j in range(12):
            out[idx : idx + 2] = _agg(raw[:, j])
            idx += 2

        # Normalized amplitudes: 4 × 2 = 8
        A = raw[:, [0, 3, 6, 9]]
        with np.errstate(invalid="ignore", divide="ignore"):
            A_n = A / A.sum(axis=1, keepdims=True)
        for j in range(4):
            out[idx : idx + 2] = _agg(A_n[:, j])
            idx += 2

        # Position differences mu_i − mu_1: 3 × 2 = 6
        mu = raw[:, [1, 4, 7, 10]]
        for j in range(1, 4):
            out[idx : idx + 2] = _agg(mu[:, j] - mu[:, 0])
            idx += 2

        # Width ratios sigma_i / sigma_1: 3 × 2 = 6
        sigma = raw[:, [2, 5, 8, 11]]
        for j in range(1, 4):
            with np.errstate(invalid="ignore", divide="ignore"):
                out[idx : idx + 2] = _agg(sigma[:, j] / sigma[:, 0])
            idx += 2

        # Fit quality: RMSE and R² × 2 each = 4
        out[idx : idx + 2] = _agg(rmse)
        idx += 2
        out[idx : idx + 2] = _agg(r2)
        idx += 2

        # Gaussian areas A × sigma × √(2π): 4 × 2 = 8
        for j in range(4):
            area_j = raw[:, 3 * j] * raw[:, 3 * j + 2] * sqrt2pi
            out[idx : idx + 2] = _agg(area_j)
            idx += 2

        # Dominant Gaussian fraction: 2
        gareas = np.column_stack([
            raw[:, 3 * j] * raw[:, 3 * j + 2] * sqrt2pi for j in range(4)
        ])
        with np.errstate(invalid="ignore", divide="ignore"):
            dom_frac = gareas.max(axis=1) / gareas.sum(axis=1)
        out[idx : idx + 2] = _agg(dom_frac)
        idx += 2

        # Early-area fraction (A1+A2 area / total): 2
        with np.errstate(invalid="ignore", divide="ignore"):
            early = (gareas[:, 0] + gareas[:, 1]) / gareas.sum(axis=1)
        out[idx : idx + 2] = _agg(early)
        idx += 2

    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — Monte-Moreno
# ─────────────────────────────────────────────────────────────────────────────

def extract_monte_moreno_features(ppg: np.ndarray, fs: int) -> np.ndarray:
    """
    69 Monte-Moreno features across four sub-groups:

      Spectral         [0:18]  — band powers (6+6), entropy, centroid, peak freq,
                                 flatness, rolloff, HR estimate
      Systolic complex [18:39] — per-pulse shape descriptors (10 × 2 stats + 1 ratio)
      Non-linear       [39:54] — entropy + fractal + complexity measures (15)
      Statistical      [54:69] — percentiles, spread, moment features (15)

    Monte-Moreno 2011, Comput. Biol. Med. 41:1092.
    Radha 2019 §3.2.3 (third morphology sub-family).
    """
    out = _nan(N_MONTE_MORENO)

    # ── Spectral [0:18] ───────────────────────────────────────────────────
    try:
        freqs, psd = sig.welch(ppg, fs=fs, nperseg=min(256, len(ppg) // 2))
        bands = [
            (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
            (2.0, 2.5), (2.5, 3.0), (3.0, 4.0),
        ]
        bp = np.array([
            float(np.trapezoid(
                psd[(freqs >= lo) & (freqs < hi)],
                freqs[(freqs >= lo) & (freqs < hi)],
            ))
            for lo, hi in bands
        ])
        tot = float(bp.sum()) + 1e-30
        out[0:6] = bp                           # absolute band powers
        out[6:12] = bp / tot                    # normalized band powers
        probs = bp / tot
        out[12] = float(-np.sum(probs * np.log2(probs + 1e-30)))  # spectral entropy
        mask = (freqs >= 0.5) & (freqs <= 4.0)
        fb, pb = freqs[mask], psd[mask]
        psum = float(pb.sum()) + 1e-30
        out[13] = float(np.sum(fb * pb) / psum)                   # centroid
        out[14] = float(fb[int(np.argmax(pb))]) if len(pb) else np.nan  # peak freq
        gm = float(np.exp(np.mean(np.log(pb + 1e-30))))
        out[15] = gm / (float(np.mean(pb)) + 1e-30)              # flatness
        cum = np.cumsum(pb)
        ri = int(np.searchsorted(cum, 0.85 * float(cum[-1])))
        out[16] = float(fb[min(ri, len(fb) - 1)]) if len(fb) else np.nan  # rolloff 85%
        out[17] = out[14] * 60.0 if np.isfinite(out[14]) else np.nan      # HR from spectrum
    except Exception:
        pass

    # ── Systolic complex [18:39] ─────────────────────────────────────────
    try:
        _, onsets = _detect_peaks(ppg, fs)
        pulses = _segment_pulses(ppg, onsets)
        n_p = len(pulses)
        if n_p >= 2:
            g_min = float(ppg.min())
            grange = max(float(ppg.max()) - g_min, 1e-10)

            sys_t = _nan(n_p)
            hw50  = _nan(n_p); hw25  = _nan(n_p)
            sa_n  = _nan(n_p); n_pos = _nan(n_p); n_amp = _nan(n_p)
            aug   = _nan(n_p); ds_r  = _nan(n_p); pt_p  = _nan(n_p)
            pk_n  = _nan(n_p)

            for i, pulse in enumerate(pulses):
                np_ = len(pulse)
                if np_ < 6:
                    continue
                onset_v = float(pulse[0])
                sys_i = int(np.argmax(pulse[: max(1, np_ // 2)]))
                sys_v = float(pulse[sys_i])
                amp = sys_v - onset_v
                if amp < 1e-10:
                    continue

                sys_t[i] = sys_i / np_
                pk_n[i] = (sys_v - g_min) / grange
                pt_p[i] = sys_i / np_

                for thresh, buf in ((0.5, hw50), (0.25, hw25)):
                    lvl = onset_v + thresh * amp
                    above = np.where(pulse >= lvl)[0]
                    if len(above) >= 2:
                        buf[i] = (int(above[-1]) - int(above[0])) / fs

                p_area = float(np.trapezoid(pulse - onset_v))
                s_area = float(np.trapezoid(pulse[: sys_i + 1] - onset_v))
                if abs(p_area) > 1e-12:
                    sa_n[i] = s_area / p_area
                    ds_r[i] = 1.0 - s_area / p_area

                # Dicrotic notch via local min in PPG' after systolic peak
                ppg1_loc = np.gradient(pulse)
                notch_cands, _ = sig.find_peaks(-ppg1_loc[sys_i:])
                if len(notch_cands) > 0:
                    nc = int(notch_cands[0])
                    n_pos[i] = (sys_i + nc) / np_
                    n_val = float(pulse[sys_i + nc])
                    n_amp[i] = (n_val - onset_v) / amp
                    dia_seg = pulse[sys_i + nc :]
                    if len(dia_seg) > 1:
                        aug[i] = (float(dia_seg.max()) - n_val) / amp

            arrs10 = [sys_t, hw50, hw25, sa_n, n_pos, n_amp, aug, ds_r, pt_p, pk_n]
            for j, arr in enumerate(arrs10):
                out[18 + j * 2 : 20 + j * 2] = _agg(arr)
            # Width ratio hw50/hw25: scalar mean (index 38)
            valid = np.isfinite(hw50) & np.isfinite(hw25) & (hw25 > 1e-12)
            if valid.any():
                out[38] = float(np.nanmean((hw50 / hw25)[valid]))
    except Exception:
        pass

    # ── Non-linear [39:54] ───────────────────────────────────────────────
    try:
        nl: list[float] = []
        # O(n log n) — full signal
        try: nl.append(_safe(antropy.perm_entropy(ppg, normalize=True)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.spectral_entropy(ppg, sf=fs, method="fft", normalize=True)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.svd_entropy(ppg, normalize=True)))
        except Exception: nl.append(np.nan)
        # O(n²) — subsample
        sub = ppg[:: max(1, len(ppg) // _APP_MAX_N)][: _APP_MAX_N]
        try: nl.append(_safe(antropy.app_entropy(sub)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.sample_entropy(sub)))
        except Exception: nl.append(np.nan)
        # O(n) — full signal
        try:
            mob, comp = antropy.hjorth_params(ppg)
            nl.extend([_safe(mob), _safe(comp)])
        except Exception:
            nl.extend([np.nan, np.nan])
        try: nl.append(_safe(antropy.higuchi_fd(ppg)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.petrosian_fd(ppg)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.detrended_fluctuation(ppg)))
        except Exception: nl.append(np.nan)
        try: nl.append(_safe(antropy.num_zerocross(ppg, normalize=True)))
        except Exception: nl.append(np.nan)
        nl.append(_safe(st.kurtosis(ppg)))
        nl.append(_safe(st.skew(ppg)))
        nl.append(float(np.mean(np.abs(ppg))))
        nl.append(float(np.sqrt(np.mean(ppg**2))))
        out[39:54] = np.array(nl[:15], dtype=np.float64)
    except Exception:
        pass

    # ── Statistical [54:69] ──────────────────────────────────────────────
    try:
        pcts = np.percentile(ppg, [5, 10, 25, 50, 75, 90, 95])
        out[54:61] = pcts.astype(np.float64)
        out[61] = float(np.std(ppg, ddof=1))
        out[62] = float(pcts[4] - pcts[2])                            # IQR
        out[63] = float(float(ppg.max()) - float(ppg.min()))          # range
        mu_ppg = float(np.mean(ppg))
        sd_ppg = float(np.std(ppg, ddof=1))
        out[64] = sd_ppg / abs(mu_ppg) if abs(mu_ppg) > 1e-12 else np.nan  # CV
        out[65] = float(np.mean(np.abs(ppg - mu_ppg)))                # MAD
        out[66] = float(ppg.max())
        out[67] = float(ppg.min())
        out[68] = float(np.mean(ppg**2))                              # signal power
    except Exception:
        pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    ppg: np.ndarray,
    accel: np.ndarray,
    fs: int = 25,
    rest_prob: float = 0.0,
) -> np.ndarray:
    """
    Extract all 195 features from one 5-minute PPG + accelerometer window.

    ppg:       (n_samples,)   — bandpass-filtered PPG (0.5–4 Hz, Yao 2022)
    accel:     (n_samples, 3) — tri-axial accelerometer in g
    fs:        sampling rate in Hz (default 25, ActiGraph LEAP)
    rest_prob: ENMO-derived rest probability [0, 1] for this window,
               smoothed at day level by caller via compute_rest_probabilities.

    Returns float64 array (195,). NaN where extraction failed.
    Ordering: [activity(21) | hrv(5) | elgendi(40) | gaussian(60) | monte_moreno(69)]
    """
    return np.concatenate([
        extract_activity_features(accel, fs, rest_prob),
        extract_hrv_features(ppg, fs),
        extract_elgendi_features(ppg, fs),
        extract_gaussian_features(ppg, fs),
        extract_monte_moreno_features(ppg, fs),
    ]).astype(np.float64)


def compute_rest_probabilities(
    enmo_per_window: np.ndarray,
    threshold_g: float = _ENMO_REST_G,
    smooth_n: int = 3,
) -> np.ndarray:
    """
    Convert per-window mean ENMO into smoothed rest probabilities for a full day.

    enmo_per_window: (n_windows,) — mean ENMO in g for each 5-minute window.
    threshold_g: ENMO below this = rest (van Hees et al. 2013, default 0.04 g).
    smooth_n: rolling-mean width (default 3 windows = 15 min).

    Returns (n_windows,) array of rest probabilities in [0, 1].
    """
    binary = (enmo_per_window < threshold_g).astype(np.float64)
    kernel = np.ones(smooth_n) / smooth_n
    return np.clip(np.convolve(binary, kernel, mode="same"), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature name index (for model interpretability and ablations)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Activity (21)
    *[
        f"act_seg{s}_{stat}"
        for s in range(4)
        for stat in ("mean", "std", "skew", "kurt", "iqr")
    ],
    "act_rest_prob",
    # HRV (5)
    *[f"hrv_mse_scale{k}" for k in range(6, 11)],
    # Elgendi (40)
    *[
        f"elg_{name}_{stat}"
        for name in (
            "sys_amp", "dia_amp", "syd_ratio", "pulse_width", "sys_width",
            "dia_width", "pulse_area", "sys_area", "dia_area", "rise_time",
            "fall_time", "max_slope", "max_slope_t", "min_slope",
            "a_wave", "b_wave", "c_wave", "d_wave", "ba_ratio", "bcd_a_ratio",
        )
        for stat in ("mean", "std")
    ],
    # Gaussian (60)
    *[
        f"gauss_G{k // 3 + 1}_{'Ams'[k % 3]}_{stat}"
        for k in range(12)
        for stat in ("mean", "std")
    ],
    *[f"gauss_normA{j + 1}_{stat}" for j in range(4) for stat in ("mean", "std")],
    *[f"gauss_dmu{j + 2}1_{stat}" for j in range(3) for stat in ("mean", "std")],
    *[f"gauss_dsigma{j + 2}1_{stat}" for j in range(3) for stat in ("mean", "std")],
    "gauss_rmse_mean", "gauss_rmse_std",
    "gauss_r2_mean", "gauss_r2_std",
    *[f"gauss_areaG{j + 1}_{stat}" for j in range(4) for stat in ("mean", "std")],
    "gauss_dom_frac_mean", "gauss_dom_frac_std",
    "gauss_early_frac_mean", "gauss_early_frac_std",
    # Monte-Moreno spectral (18)
    *[f"mm_bp_abs_{lo}_{hi}" for lo, hi in [(5,10),(10,15),(15,20),(20,25),(25,30),(30,40)]],
    *[f"mm_bp_rel_{lo}_{hi}" for lo, hi in [(5,10),(10,15),(15,20),(20,25),(25,30),(30,40)]],
    "mm_spectral_entropy", "mm_centroid", "mm_peak_freq",
    "mm_flatness", "mm_rolloff85", "mm_hr_spectral",
    # Monte-Moreno systolic complex (21)
    *[
        f"mm_{name}_{stat}"
        for name in (
            "sys_t_norm", "hw50", "hw25", "sys_area_norm", "notch_pos",
            "notch_amp", "aug_idx", "ds_ratio", "pt_proxy", "peak_norm",
        )
        for stat in ("mean", "std")
    ],
    "mm_width_ratio",
    # Monte-Moreno non-linear (15)
    "mm_perm_ent", "mm_spec_ent_ant", "mm_svd_ent",
    "mm_app_ent", "mm_samp_ent",
    "mm_hjorth_mob", "mm_hjorth_comp",
    "mm_higuchi_fd", "mm_petrosian_fd", "mm_dfa",
    "mm_zerocross", "mm_kurtosis", "mm_skewness", "mm_mean_abs", "mm_rms",
    # Monte-Moreno statistical (15)
    "mm_p5", "mm_p10", "mm_p25", "mm_p50", "mm_p75", "mm_p90", "mm_p95",
    "mm_std", "mm_iqr", "mm_range", "mm_cv", "mm_mad", "mm_max", "mm_min", "mm_power",
]
assert len(FEATURE_NAMES) == N_TOTAL, (
    f"FEATURE_NAMES length {len(FEATURE_NAMES)} != {N_TOTAL}"
)
