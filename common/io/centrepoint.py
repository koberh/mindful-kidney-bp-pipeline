"""CentrePoint V3 API PPG reader → LEAPRecording adapter.

The Mindful Kidney device data pulled from the ActiGraph CentrePoint V3 API
is NOT in the LEAP ActiLife CSV export format that `common.io.read_ppg_csv`
expects. The differences:

  * gzip-compressed despite the ``.csv`` extension (magic ``1f 8b 08``)
  * a plain named-column header row:
        Ambient,Green,Channel,StudyId,SubjectId,MonitorSerial,Sensor,Timestamp
  * Timestamp is **Unix epoch milliseconds** (true UTC), not a
    ``%m/%d/%Y %H:%M:%S.%f`` local-time string
  * two interleaved channels (Channel 0 / Channel 1) share each timestamp
  * NO accelerometer channel is exported (only PPG + temperature)

Layout on disk (confirmed 2026-05-27):
    data/{subjectId}/ppg-green-100-hz/{YYYY-MM-DD}.csv   100 Hz green PPG
    data/{subjectId}/ppg-green/{YYYY-MM-DD}.csv           25 Hz green PPG
    data/{subjectId}/temperature/{YYYY-MM-DD}.csv         device temp (~1/60 Hz)

Timestamps are genuine UTC epoch-ms: each daily file spans one UTC calendar
day (e.g. the 2026-05-21 file runs 05:40–23:45 UTC), so files are split on
UTC midnight. ``tz_offset_hours`` here is only forwarded as metadata for
downstream local-time (sleep/wake) logic — it is NOT applied to the
timestamps, which are already true UTC.

Accelerometer: CentrePoint did not export accel for this study, so we
synthesise a constant 1 g (gravity on +Z) dummy array. Per the Phase 3
model design (OPEN_QUESTIONS Q9) the 21 activity features derived from accel
are sliced off before the LSTM, so a dummy accel does NOT affect BP
predictions — it only neutralises the accel-driven sleep/activity panels
(which then fall back to the time-of-day prior).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.io import LEAPMetadata, LEAPRecording


# ── PPG reader ───────────────────────────────────────────────────────────────

def read_centrepoint_ppg(path: Path) -> pd.DataFrame:
    """Load one gzipped CentrePoint PPG CSV into a wide-format DataFrame.

    Returns a DataFrame indexed by tz-naive UTC timestamp with columns
    ``Green_0, Green_1, Ambient_0, Ambient_1``. Green is NEGATED so systolic
    peaks point up (matching the LEAP reader convention; raw green ADC is a
    large negative value in normal operation).
    """
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=["Ambient", "Green", "Channel", "Timestamp"],
        dtype={"Ambient": "int32", "Green": "int64", "Channel": "int8",
               "Timestamp": "int64"},
    )

    # Epoch milliseconds → tz-naive UTC datetime
    df["timestamp_utc"] = pd.to_datetime(df["Timestamp"], unit="ms")

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

    # ch0 and ch1 are interleaved 1:1 at each timestamp; align by row order.
    n = min(len(ch0), len(ch1))
    wide = pd.concat([ch0.iloc[:n], ch1.iloc[:n]], axis=1)
    wide = wide.dropna(subset=["timestamp_utc"])
    wide = wide.sort_values("timestamp_utc").reset_index(drop=True)

    # Negate green: raw is large-negative, systolic = least negative
    wide["Green_0"] = -wide["Green_0"]
    wide["Green_1"] = -wide["Green_1"]

    return wide.set_index("timestamp_utc")


# ── Full recording loader ─────────────────────────────────────────────────────

def load_centrepoint_recording(
    ppg_paths: list[Path] | Path,
    participant_id: str,
    tz_offset_hours: float = -7.0,
    ppg_fs: int = 100,
    serial_number: str = "unknown",
) -> LEAPRecording:
    """Load one or more daily CentrePoint PPG files into a LEAPRecording.

    Args:
        ppg_paths:        single path or list of daily ppg CSVs (will be sorted
                          and concatenated in chronological order)
        participant_id:   label for this recording
        tz_offset_hours:  local-to-UTC offset for downstream sleep/wake logic
                          ONLY (timestamps are already true UTC). San Diego in
                          May = PDT = -7.0.
        ppg_fs:           nominal PPG sample rate (100 or 25)
        serial_number:    device serial for metadata

    A constant 1 g (+Z) dummy accelerometer is synthesised — accel does not
    affect BP predictions (see module docstring).
    """
    if isinstance(ppg_paths, (str, Path)):
        ppg_paths = [Path(ppg_paths)]
    ppg_paths = sorted(Path(p) for p in ppg_paths)

    frames = [read_centrepoint_ppg(p) for p in ppg_paths]
    wide = pd.concat(frames, axis=0)
    wide = wide[~wide.index.duplicated(keep="first")].sort_index()

    ppg_times = pd.DatetimeIndex(wide.index)
    n = len(ppg_times)

    # Dummy accel: gravity on +Z → magnitude 1 g → ENMO ~0 (still).
    accel = np.zeros((n, 3), dtype=np.float64)
    accel[:, 2] = 1.0

    meta = LEAPMetadata(
        participant_id=participant_id,
        serial_number=serial_number,
        firmware="centrepoint-v3-api",
        accel_fs=ppg_fs,           # dummy accel shares the PPG time axis
        ppg_fs=ppg_fs,
        start_utc=ppg_times[0],
        tz_offset_hours=tz_offset_hours,
    )

    return LEAPRecording(
        metadata=meta,
        ppg_ch0=wide["Green_0"].to_numpy(dtype=np.float64),
        ppg_ch1=wide["Green_1"].to_numpy(dtype=np.float64),
        ambient_ch0=wide["Ambient_0"].to_numpy(dtype=np.float64),
        accel=accel,
        timestamps_utc=ppg_times,
    )
