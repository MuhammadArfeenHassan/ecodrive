"""Segment per-sample inertial recordings into fixed-length overlapping windows.

The windower is deliberately independent of any particular dataset: it takes a long
dataframe of samples with a timestamp, a set of signal columns, a label column and a
grouping column (driver or trip), and returns one row per window. Nothing downstream
needs to know where the samples came from.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    """Configuration for one windowing pass.

    length_s: window length in seconds.
    overlap: fraction of a window shared with the next one (0.5 = 50% overlap).
    min_fill: a window is discarded unless it holds at least this fraction of the
        samples the nominal rate would predict, which drops windows that straddle a
        gap in the recording.
    label_purity: a window is discarded unless at least this fraction of its samples
        carry the window's majority label, so windows spanning a label change are
        not fed to the model with an arbitrary label.
    """

    length_s: float = 2.0
    overlap: float = 0.5
    min_fill: float = 0.8
    label_purity: float = 0.9


def estimate_rate_hz(timestamps: pd.Series) -> float:
    """Median sampling rate, in Hz, from the median inter-sample gap."""
    gaps = np.diff(np.sort(timestamps.to_numpy(dtype=float)))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float("nan")
    return float(1.0 / np.median(gaps))


def make_windows(
    df: pd.DataFrame,
    signal_columns: Sequence[str],
    time_column: str,
    label_column: str,
    group_column: str,
    spec: WindowSpec,
) -> List[dict]:
    """Split each group's samples into windows.

    Returns a list of dicts, each holding the raw signal matrix for one window plus the
    window's label and group. Feature extraction happens separately so that the same
    windows can feed both the feature-based model and any raw-signal model later.
    """
    step_s = spec.length_s * (1.0 - spec.overlap)
    if step_s <= 0:
        raise ValueError("overlap must be < 1.0")

    windows: List[dict] = []
    for group, chunk in df.groupby(group_column, sort=True):
        chunk = chunk.sort_values(time_column)
        t = chunk[time_column].to_numpy(dtype=float)
        if t.size < 2:
            continue
        rate = estimate_rate_hz(chunk[time_column])
        expected = max(1, int(round(spec.length_s * rate)))
        signals = chunk[list(signal_columns)].to_numpy(dtype=float)
        labels = chunk[label_column].to_numpy()

        start = t[0]
        end_of_data = t[-1]
        while start + spec.length_s <= end_of_data + 1e-9:
            stop = start + spec.length_s
            sel = (t >= start) & (t < stop)
            n = int(sel.sum())
            if n >= spec.min_fill * expected:
                window_labels = labels[sel]
                values, counts = np.unique(window_labels, return_counts=True)
                majority = values[counts.argmax()]
                purity = counts.max() / n
                if purity >= spec.label_purity:
                    windows.append(
                        {
                            "group": group,
                            "label": majority,
                            "t_start": start,
                            "n_samples": n,
                            "signals": signals[sel],
                        }
                    )
            start += step_s
    return windows


def windows_summary(windows: List[dict]) -> pd.DataFrame:
    """One row per window with its group and label, for counting and sanity checks."""
    return pd.DataFrame(
        [{"group": w["group"], "label": w["label"], "n_samples": w["n_samples"]} for w in windows]
    )
