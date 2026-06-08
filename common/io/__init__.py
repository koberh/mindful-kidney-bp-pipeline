"""LEAP file format readers and CentrePoint-derived data loaders.

Reads ActiGraph LEAP v2 CSV exports and CentrePoint algorithm outputs.
All timestamps are returned in UTC.

Confirmed schema (device STM2E24242014, firmware 2.0.1, 2024-12-28 recording):
  ppg25Hz.csv  — 25 Hz green PPG, two interleaved channels (0/1), local time
  RAW.csv      — 32 Hz tri-axial accelerometer, local time
  InterBeatInterval.csv — CentrePoint IBI (seconds), UTC
  HeartRateVar.csv      — CentrePoint 5-min HRV epochs, UTC
  HeartRate.csv         — CentrePoint 1-min HR, UTC

Timezone: LEAP stores local time; offset extracted from agsd/info.json or
provided explicitly (device timezone was -05:00 for test recording).

PPG sign convention: raw green ADC values are large negatives in normal
operation. Negating gives standard reflectance PPG (systolic peaks positive).
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class LEAPMetadata:
    participant_id: str
    serial_number: str
    firmware: str
    accel_fs: int                    # Hz, typically 32
    ppg_fs: int                      # Hz (25 or 100)
    start_utc: pd.Timestamp
    tz_offset_hours: float           # e.g. -5.0 for EST
    limb: str = "Wrist"


@dataclass
class LEAPRecording:
    """Aligned PPG + accelerometer data for one LEAP recording."""

    metadata: LEAPMetadata
    ppg_ch0: np.ndarray              # (n_samples,) negated green ADC, primary channel
    ppg_ch1: np.ndarray              # (n_samples,) negated green ADC, secondary channel
    ambient_ch0: np.ndarray          # (n_samples,) ambient light ch0
    accel: np.ndarray                # (n_samples, 3) XYZ in g, resampled to ppg_fs
    timestamps_utc: pd.DatetimeIndex  # length n_samples, UTC

    @property
    def ppg_green(self) -> np.ndarray:
        """Primary PPG channel (ch0), negated for standard peak-up convention."""
        return self.ppg_ch0

    @property
    def fs(self) -> int:
        return self.metadata.ppg_fs


@dataclass
class CentrePointIBI:
    """Beat-by-beat inter-beat intervals from CentrePoint algorithm."""
    timestamps_utc: pd.DatetimeIndex   # time of each beat detection
    ibi_seconds: np.ndarray           # (n_beats,) inter-beat intervals in seconds


@dataclass
class CentrePointHRV:
    """5-minute HRV epoch summaries from CentrePoint."""
    df: pd.DataFrame                  # full DataFrame with all HRV columns


@dataclass
class CuffReading:
    """Single cuff-based BP measurement."""
    participant_id: str
    timestamp_utc: pd.Timestamp
    sbp_mmhg: float
    dbp_mmhg: float
    notes: str = ""


@dataclass
class AlignedWindow:
    """PPG + accel window paired with the nearest cuff reading."""
    cuff: CuffReading
    ppg: np.ndarray      # (window_samples,)
    accel: np.ndarray    # (window_samples, 3)
    window_start_utc: pd.Timestamp
    window_end_utc: pd.Timestamp
    lag_seconds: float   # signed: cuff.timestamp_utc − window_center


# ── Header parsing ───────────────────────────────────────────────────────────

def _leap_skiprows(path: Path) -> int:
    """
    Return the number of lines to skip (through and including the column header).

    LEAP CSV files have a variable-length text header ending with a dashed
    separator line, followed by a comma-delimited column header starting with
    'Timestamp'. Different export types have slightly different header lengths
    (e.g. RAW.csv has an extra 'Epoch Period' line vs ppg25Hz.csv).
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            # Column header row starts with 'Timestamp' and contains commas
            if stripped.startswith("Timestamp") and "," in stripped:
                return i + 1  # skip through this line
    return 11  # safe fallback


def _parse_leap_header(path: Path) -> dict[str, str]:
    """Extract key-value pairs from the LEAP CSV text header."""
    meta: dict[str, str] = {}
    skiprows = _leap_skiprows(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= skiprows:
                break
            line = line.strip()
            for key, prefix in (
                ("serial_number", "Serial Number:"),
                ("start_time", "Start Time"),
                ("start_date", "Start Date"),
                ("firmware", "ActiGraph LEAP v2 ActiLife"),
            ):
                if prefix in line:
                    meta[key] = line.split(prefix)[-1].strip().strip(":").strip()
    return meta


def _tz_offset_from_agsd(agsd_path: Path) -> float:
    """Extract timezone offset in hours from agsd/info.json (e.g. -5.0 for EST)."""
    try:
        with zipfile.ZipFile(agsd_path) as z:
            info = json.loads(z.read("info.json"))
        tz_str: str = info.get("timeZone", "-00:00:00")
        sign = -1 if tz_str.startswith("-") else 1
        parts = tz_str.lstrip("+-").split(":")
        return sign * (int(parts[0]) + int(parts[1]) / 60)
    except Exception:
        return 0.0


# ── PPG reader ───────────────────────────────────────────────────────────────

def read_ppg_csv(
    path: Path,
    tz_offset_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Load a LEAP PPG CSV (25 Hz or 100 Hz) into a wide-format DataFrame.

    Columns returned: timestamp_utc (DatetimeIndex), Green_0, Green_1,
    Ambient_0, Ambient_1.

    Green values are NEGATED so systolic peaks are positive (standard
    reflectance PPG convention; raw values are large negatives in normal op).

    tz_offset_hours: local-to-UTC offset, e.g. -5.0 for EST.
    """
    skip = _leap_skiprows(path)
    df = pd.read_csv(
        path,
        skiprows=skip,
        names=["Timestamp", "Ambient", "Channel", "Green"],
        dtype={"Ambient": "int32", "Channel": "int8", "Green": "int64"},
        parse_dates=False,
    )

    df["ts_local"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M:%S.%f")
    offset = pd.Timedelta(hours=-tz_offset_hours)  # subtract offset to get UTC
    df["timestamp_utc"] = df["ts_local"] + offset

    # Separate channels and merge side-by-side
    ch0 = (
        df[df["Channel"] == 0][["timestamp_utc", "Green", "Ambient"]]
        .rename(columns={"Green": "Green_0", "Ambient": "Ambient_0"})
        .reset_index(drop=True)
    )
    ch1 = (
        df[df["Channel"] == 1][["Green", "Ambient"]]
        .rename(columns={"Green": "Green_1", "Ambient": "Ambient_1"})
        .reset_index(drop=True)
    )

    wide = pd.concat([ch0, ch1], axis=1)
    wide = wide.dropna(subset=["timestamp_utc"])
    wide = wide.sort_values("timestamp_utc").reset_index(drop=True)

    # Negate green channels: raw is negative, systolic = least negative
    wide["Green_0"] = -wide["Green_0"]
    wide["Green_1"] = -wide["Green_1"]

    wide = wide.set_index("timestamp_utc")
    return wide


# ── Accelerometer reader ─────────────────────────────────────────────────────

def read_accel_csv(
    path: Path,
    tz_offset_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Load a LEAP RAW accelerometer CSV (32 Hz) into a DataFrame.

    Columns returned: timestamp_utc (DatetimeIndex), Accel_X, Accel_Y, Accel_Z in g.
    """
    skip = _leap_skiprows(path)
    df = pd.read_csv(
        path,
        skiprows=skip,
        names=["Timestamp", "Accel_X", "Accel_Y", "Accel_Z"],
        dtype={"Accel_X": "float32", "Accel_Y": "float32", "Accel_Z": "float32"},
        parse_dates=False,
    )

    df["ts_local"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M:%S.%f")
    offset = pd.Timedelta(hours=-tz_offset_hours)
    df["timestamp_utc"] = df["ts_local"] + offset
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df.set_index("timestamp_utc")[["Accel_X", "Accel_Y", "Accel_Z"]]


# ── CentrePoint-derived data readers ─────────────────────────────────────────

def _centrepoint_to_utc(series: pd.Series, tz_offset_hours: float) -> pd.Series:
    """
    Convert CentrePoint timestamps to true UTC.

    CentrePoint stores timestamps in device local time but appends '+0000',
    making them appear to be UTC. We parse them as-is, then shift by the
    true UTC offset to get correct UTC values.

    tz_offset_hours: device local time offset, e.g. -5.0 for EST.
    UTC = CentrePoint_timestamp - tz_offset_hours hours
    """
    parsed = pd.to_datetime(series, format="ISO8601", utc=True)
    # Strip the (incorrect) timezone info, then re-localize as UTC after shifting
    naive = parsed.dt.tz_localize(None)
    corrected = naive - pd.Timedelta(hours=tz_offset_hours)
    return corrected


def read_ibi_csv(path: Path, tz_offset_hours: float = 0.0) -> CentrePointIBI:
    """
    Load CentrePoint InterBeatInterval.csv. IBI in seconds.

    tz_offset_hours: device local-time offset (e.g. -5.0 for EST).
    CentrePoint labels timestamps as +0000 but stores local time — this
    function corrects them to true UTC.
    """
    df = pd.read_csv(
        path,
        usecols=["timestamp", "interbeat_interval"],
        dtype={"interbeat_interval": "float32"},
    )
    df["timestamp"] = _centrepoint_to_utc(df["timestamp"], tz_offset_hours)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return CentrePointIBI(
        timestamps_utc=pd.DatetimeIndex(df["timestamp"]),
        ibi_seconds=df["interbeat_interval"].to_numpy(dtype=np.float32),
    )


def read_hrv_csv(path: Path, tz_offset_hours: float = 0.0) -> CentrePointHRV:
    """
    Load CentrePoint HeartRateVar.csv (5-minute HRV epochs).
    Columns: start, end, window_size_seconds, hrv_reliability, rmssd, sdnn,
    sdsd, mean_nn, pnn50, pnn20, lf_power, hf_power, breathing_rate, etc.

    tz_offset_hours: applied to correct the pseudo-UTC timestamps to true UTC.
    """
    df = pd.read_csv(path)
    df["start"] = _centrepoint_to_utc(df["start"], tz_offset_hours)
    df["end"] = _centrepoint_to_utc(df["end"], tz_offset_hours)
    # CentrePoint stores reliability as 0–100; normalize to 0–1
    if "hrv_reliability" in df.columns:
        df["hrv_reliability"] = df["hrv_reliability"] / 100.0
    if "breathing_rate_reliability" in df.columns:
        df["breathing_rate_reliability"] = df["breathing_rate_reliability"] / 100.0
    return CentrePointHRV(df=df.sort_values("start").reset_index(drop=True))


def read_heart_rate_csv(path: Path, tz_offset_hours: float = 0.0) -> pd.DataFrame:
    """
    Load CentrePoint HeartRate.csv (1-minute HR epochs).
    Returns DataFrame with columns: timestamp_utc, heart_rate_bpm.

    tz_offset_hours: applied to correct pseudo-UTC timestamps to true UTC.
    """
    df = pd.read_csv(path, usecols=["timestamp", "HeartRate"])
    df["timestamp"] = _centrepoint_to_utc(df["timestamp"], tz_offset_hours)
    df = df.rename(columns={"timestamp": "timestamp_utc", "HeartRate": "heart_rate_bpm"})
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ── Full recording loader ─────────────────────────────────────────────────────

def load_leap_recording(
    ppg_path: Path,
    accel_path: Path,
    participant_id: str,
    agsd_path: Path | None = None,
    tz_offset_hours: float | None = None,
    ppg_fs: int = 25,
    accel_fs: int = 32,
) -> LEAPRecording:
    """
    Load and align PPG + accelerometer for one LEAP participant recording.

    ppg_path:        path to *ppg25Hz.csv or *ppg100Hz.csv
    accel_path:      path to *RAW.csv (32 Hz accelerometer)
    agsd_path:       optional path to *.agsd (for timezone extraction)
    tz_offset_hours: local-to-UTC hours; if None, read from agsd or default 0
    ppg_fs:          nominal PPG sampling rate (25 or 100 Hz)
    accel_fs:        nominal accel sampling rate (32 Hz)

    The accelerometer is resampled to ppg_fs using linear interpolation so
    that accel[:, :] and ppg_ch0[:] share the same time axis.
    """
    # Resolve timezone
    if tz_offset_hours is None:
        if agsd_path is not None and agsd_path.exists():
            tz_offset_hours = _tz_offset_from_agsd(agsd_path)
        else:
            tz_offset_hours = 0.0

    raw_meta = _parse_leap_header(ppg_path)
    serial = raw_meta.get("serial_number", "unknown")

    # Load PPG
    ppg_df = read_ppg_csv(ppg_path, tz_offset_hours=tz_offset_hours)

    # Load accelerometer
    accel_df = read_accel_csv(accel_path, tz_offset_hours=tz_offset_hours)

    # Find common time window
    t_start = max(ppg_df.index[0], accel_df.index[0])
    t_end = min(ppg_df.index[-1], accel_df.index[-1])

    ppg_win = ppg_df.loc[t_start:t_end]
    accel_win = accel_df.loc[t_start:t_end]

    # Resample accel to PPG time axis using linear interpolation
    # Combine both time axes, interpolate, then select PPG times
    ppg_times = ppg_win.index
    accel_reindexed = (
        accel_win.reindex(accel_win.index.union(ppg_times))
        .interpolate(method="time", limit_direction="both")
        .reindex(ppg_times)
    )

    meta = LEAPMetadata(
        participant_id=participant_id,
        serial_number=serial,
        firmware=raw_meta.get("firmware", "unknown"),
        accel_fs=accel_fs,
        ppg_fs=ppg_fs,
        start_utc=ppg_times[0],
        tz_offset_hours=tz_offset_hours,
    )

    return LEAPRecording(
        metadata=meta,
        ppg_ch0=ppg_win["Green_0"].to_numpy(dtype=np.float64),
        ppg_ch1=ppg_win["Green_1"].to_numpy(dtype=np.float64),
        ambient_ch0=ppg_win["Ambient_0"].to_numpy(dtype=np.float64),
        accel=accel_reindexed[["Accel_X", "Accel_Y", "Accel_Z"]].to_numpy(dtype=np.float64),
        timestamps_utc=ppg_times,
    )


# ── Cuff BP ───────────────────────────────────────────────────────────────────

def read_cuff_bp(filepath: Path, participant_id: str | None = None) -> list[CuffReading]:
    """Load cuff-BP reference CSV for one participant.

    Expected CSV schema (header row, comma-separated):

      timestamp_utc   ISO-8601 UTC timestamp (e.g. 2026-05-13T14:35:00+0000)
      sbp_mmhg        systolic BP (mmHg, integer or float)
      dbp_mmhg        diastolic BP (mmHg)
      notes           optional free-text (arm, cuff size, posture, etc.)
      participant_id  optional — used only if `participant_id` arg is None to
                      filter to one participant from a multi-participant file

    Notes:
    - Timestamps assumed UTC. If they're local-time pre-conversion, the
      caller is responsible for shifting before passing in.
    - Missing optional columns are tolerated.
    - One CSV row per cuff reading. The file should typically contain only
      the baseline-visit readings used for per-participant calibration.

    Schema source: study-defined; this is the format we'll request from the
    REDCap export of the IRB-approved cuff measurement form. Update this
    docstring (and the test fixture) when the final REDCap field names are
    confirmed.
    """
    df = pd.read_csv(filepath)
    required = {"timestamp_utc", "sbp_mmhg", "dbp_mmhg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cuff CSV {filepath} missing required columns: {sorted(missing)}. "
            f"Expected at minimum {sorted(required)}."
        )
    if participant_id is not None and "participant_id" in df.columns:
        df = df[df["participant_id"] == participant_id]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    readings: list[CuffReading] = []
    for _, row in df.iterrows():
        if not np.isfinite(row["sbp_mmhg"]) or not np.isfinite(row["dbp_mmhg"]):
            continue
        # LEAPRecording.timestamps_utc is tz-naive (UTC values without offset
        # metadata); align CuffReading the same way so set-comparison works.
        ts = pd.Timestamp(row["timestamp_utc"])
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        readings.append(CuffReading(
            participant_id=str(row.get("participant_id", participant_id or "unknown")),
            timestamp_utc=ts,
            sbp_mmhg=float(row["sbp_mmhg"]),
            dbp_mmhg=float(row["dbp_mmhg"]),
            notes=str(row.get("notes", "")) if "notes" in df.columns else "",
        ))
    return readings


def align_cuff_to_windows(
    recording: LEAPRecording,
    cuff_readings: list[CuffReading],
    window_seconds: float = 300.0,
    max_lag_seconds: float = 900.0,
) -> list[AlignedWindow]:
    """
    Center a window around each cuff reading and pair them.

    max_lag_seconds: discard cuff readings with no PPG data within this range.
    Returns AlignedWindow list sorted by cuff timestamp.
    """
    results: list[AlignedWindow] = []
    half = pd.Timedelta(seconds=window_seconds / 2)
    fs = recording.fs
    ts = recording.timestamps_utc

    for cuff in sorted(cuff_readings, key=lambda c: c.timestamp_utc):
        t0 = cuff.timestamp_utc - half
        t1 = cuff.timestamp_utc + half

        mask = (ts >= t0) & (ts <= t1)
        if not mask.any():
            continue

        lag = (cuff.timestamp_utc - ts[mask].mean()).total_seconds()
        if abs(lag) > max_lag_seconds:
            continue

        results.append(AlignedWindow(
            cuff=cuff,
            ppg=recording.ppg_green[mask],
            accel=recording.accel[mask],
            window_start_utc=ts[mask][0],
            window_end_utc=ts[mask][-1],
            lag_seconds=lag,
        ))

    return results
