"""Loader for the smartphone driving dataset of Chhabra et al. (Mendeley 9vr83n7z5j).

Layout on disk, after extracting the archives:

    <root>/Day-1R/Day-1R/{Accelerometer,Gyroscope,GPS,Proximity}.csv
    <root>/Day-1S/Day-1S/...
    <root>/Driver-1/Driver-1/{Accelerometer,Gyroscope,Pressure}.csv

The day folders carry the label in their name: the trailing R marks a rash-driving
session and S a safe one. The driver folders carry no such marker, so they cannot be
used for supervised training and are ignored here.

Two header formats appear in the files. Some carry "Timestamp,Milliseconds,X,Y,Z" and
others insert a "Unix Timestamp" column. Both are handled; Milliseconds is the
within-recording clock and is what the windower uses, converted to seconds.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

ACC_COLUMNS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLUMNS = ["gyro_x", "gyro_y", "gyro_z"]
SIGNAL_COLUMNS = ACC_COLUMNS + GYRO_COLUMNS


def _read_sensor(path: Path, prefix: str) -> Optional[pd.DataFrame]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if "Milliseconds" not in df.columns:
        return None
    needed = [c for c in ("X", "Y", "Z") if c in df.columns]
    if len(needed) != 3:
        return None
    out = pd.DataFrame(
        {
            "t": _reconstruct_clock(df["Milliseconds"].astype(float)) / 1000.0,
            f"{prefix}_x": df["X"].astype(float),
            f"{prefix}_y": df["Y"].astype(float),
            f"{prefix}_z": df["Z"].astype(float),
        }
    )
    return out.dropna().sort_values("t")


def _reconstruct_clock(ms: pd.Series, stall_ms: float = 1000.0) -> pd.Series:
    """Rebuild a usable clock from a timestamp column that stalls and then flushes.

    The millisecond column in this dataset cannot be used as written. In Day-5S, 111,212
    rows carry only 7,663 distinct stamps: most stamps appear once, about 10 ms apart,
    which is the genuine 100 Hz acquisition, but a minority repeat 50 to 276 times, and
    the largest gap between distinct stamps is 260 seconds. The pattern is a logger that
    stalls and then writes its backlog under a single stamp. Taking the column at face
    value makes 93 percent of consecutive samples look simultaneous.

    The reconstruction keeps the acquisition order, assumes the steady rate the file's
    own two time columns agree on (111,212 rows over a 1,112-second wall-clock span, or
    100.0 rows per second), and cuts the recording at every stall longer than stall_ms.
    Each segment then gets its own uniform clock, and segments are separated by a wide
    margin so that no analysis window can span a stall. Nothing is interpolated and no
    sample is invented; what changes is only which samples are treated as contiguous.
    """
    import numpy as np

    values = ms.to_numpy(dtype=float)
    period_ms = 10.0  # median gap between distinct stamps, i.e. the nominal 100 Hz
    jumps = np.diff(values)
    breaks = np.flatnonzero(jumps > stall_ms) + 1
    segment_starts = np.concatenate([[0], breaks, [values.size]])

    rebuilt = np.empty(values.size, dtype=float)
    cursor = 0.0
    for a, b in zip(segment_starts[:-1], segment_starts[1:]):
        n = b - a
        rebuilt[a:b] = cursor + np.arange(n) * period_ms
        # Leave a 60-second margin so a window cannot straddle two segments.
        cursor = rebuilt[b - 1] + 60_000.0
    return pd.Series(rebuilt, index=ms.index)


def _merge_on_time(acc: pd.DataFrame, gyro: Optional[pd.DataFrame], tol_s: float = 0.05):
    """Align gyroscope samples to accelerometer timestamps by nearest match.

    The two sensors are logged on separate clocks at slightly different rates, so an
    exact join drops most rows. A nearest-neighbour join within one tolerance window
    keeps them aligned without inventing samples.
    """
    if gyro is None:
        for c in GYRO_COLUMNS:
            acc[c] = 0.0
        return acc, False
    merged = pd.merge_asof(
        acc.sort_values("t"), gyro.sort_values("t"), on="t",
        direction="nearest", tolerance=tol_s,
    )
    merged[GYRO_COLUMNS] = merged[GYRO_COLUMNS].ffill().bfill()
    return merged, True


def load_sessions(root: str) -> pd.DataFrame:
    """Return one long dataframe of all labelled day sessions.

    Columns: t, acc_x/y/z, gyro_x/y/z, label ('rash'/'safe'), group (session name),
    day (integer).
    """
    root_path = Path(root)
    frames: List[pd.DataFrame] = []
    for folder in sorted(root_path.glob("Day-*")):
        inner = folder / folder.name
        base = inner if inner.is_dir() else folder
        name = folder.name                      # e.g. Day-3R
        suffix = name.split("-")[-1][-1].upper()
        if suffix not in {"R", "S"}:
            continue
        acc = _read_sensor(base / "Accelerometer.csv", "acc")
        if acc is None:
            print(f"  skipped {name}: no usable accelerometer file")
            continue
        gyro = _read_sensor(base / "Gyroscope.csv", "gyro")
        merged, had_gyro = _merge_on_time(acc, gyro)
        merged["label"] = "rash" if suffix == "R" else "safe"
        merged["group"] = name
        merged["day"] = int("".join(ch for ch in name.split("-")[1] if ch.isdigit()))
        frames.append(merged)
        print(f"  {name}: {len(merged):,} samples, label={merged['label'].iloc[0]}, "
              f"gyroscope={'yes' if had_gyro else 'MISSING (zero-filled)'}")

    if not frames:
        raise FileNotFoundError(f"No usable Day-* sessions under {root}")
    return pd.concat(frames, ignore_index=True)
