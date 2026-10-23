"""Per-window feature extraction for inertial driving signals.

Each window becomes a fixed-length feature vector: time-domain statistics that describe
how hard and how abruptly the vehicle moved, and frequency-band energies that describe
how that motion was distributed over time. Both are computed per axis, plus magnitude
channels that are invariant to how the phone was oriented in the cab.
"""

from typing import List, Sequence

import numpy as np
import pandas as pd


def _band_energies(signal: np.ndarray, rate_hz: float, bands: Sequence[tuple]) -> List[float]:
    """Fraction of spectral energy in each frequency band."""
    if signal.size < 4 or not np.isfinite(rate_hz) or rate_hz <= 0:
        return [0.0] * len(bands)
    centered = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / rate_hz)
    total = spectrum.sum()
    if total <= 0:
        return [0.0] * len(bands)
    return [float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in bands]


def window_features(
    signals: np.ndarray,
    column_names: Sequence[str],
    rate_hz: float,
    bands: Sequence[tuple] = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 15.0)),
) -> dict:
    """Features for one window.

    signals: (n_samples, n_channels) array in the order of column_names.
    """
    feats: dict = {}
    dt = 1.0 / rate_hz if (np.isfinite(rate_hz) and rate_hz > 0) else np.nan

    for j, name in enumerate(column_names):
        x = signals[:, j]
        feats[f"{name}_mean"] = float(np.mean(x))
        feats[f"{name}_std"] = float(np.std(x))
        feats[f"{name}_min"] = float(np.min(x))
        feats[f"{name}_max"] = float(np.max(x))
        feats[f"{name}_range"] = float(np.max(x) - np.min(x))
        feats[f"{name}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"{name}_p10"] = float(np.percentile(x, 10))
        feats[f"{name}_p90"] = float(np.percentile(x, 90))
        feats[f"{name}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))

        # Jerk: the rate of change of the signal. Aggressive driving shows up as
        # abruptness rather than magnitude, and a hard but smooth stop is not the
        # same event as a snatched one.
        if x.size > 1 and np.isfinite(dt):
            jerk = np.diff(x) / dt
            feats[f"{name}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
            feats[f"{name}_jerk_max"] = float(np.max(np.abs(jerk)))
        else:
            feats[f"{name}_jerk_rms"] = 0.0
            feats[f"{name}_jerk_max"] = 0.0

        for (lo, hi), energy in zip(bands, _band_energies(x, rate_hz, bands)):
            feats[f"{name}_band_{lo}_{hi}"] = energy

    # Orientation-free magnitude channels. A phone resting in a cup holder and one in a
    # cradle produce different per-axis readings for the same manoeuvre; the vector
    # magnitude does not depend on how the device was mounted.
    n_axes = min(3, signals.shape[1])
    if n_axes == 3:
        mag = np.linalg.norm(signals[:, :3], axis=1)
        feats["mag_mean"] = float(np.mean(mag))
        feats["mag_std"] = float(np.std(mag))
        feats["mag_max"] = float(np.max(mag))
        feats["mag_p90"] = float(np.percentile(mag, 90))
        if mag.size > 1 and np.isfinite(dt):
            feats["mag_jerk_rms"] = float(np.sqrt(np.mean((np.diff(mag) / dt) ** 2)))
        else:
            feats["mag_jerk_rms"] = 0.0

    return feats


def features_table(windows: List[dict], column_names: Sequence[str], rate_hz: float) -> pd.DataFrame:
    """Feature matrix for a list of windows, with label and group columns attached."""
    rows = []
    for w in windows:
        row = window_features(w["signals"], column_names, rate_hz)
        row["label"] = w["label"]
        row["group"] = w["group"]
        row["t_start"] = w["t_start"]
        rows.append(row)
    return pd.DataFrame(rows)
