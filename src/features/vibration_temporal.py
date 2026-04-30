"""Vibration encoder-input features: per-channel time-aligned multichannel series.

Produces a 3-D tensor with shape ``(n_vib, 3, T_vib)`` where the inner
3 channels are ``[amplitude, hilbert_envelope, rolling_kurtosis]``.

Distinct from `vibration_envelope.py`, which produces a fixed-length feature
vector per window for classical baselines. Here we keep the time series so the
V1 1-D CNN can convolve along it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis as _kurtosis


def compute_vibration_input_stack(
    accel_data: np.ndarray,
    *,
    kurtosis_window: int = 17,
) -> np.ndarray:
    """Build the V1 vibration encoder input.

    Args:
        accel_data: ``(n_vib, T_vib)`` raw amplitude (already at the per-dataset
            target sample rate produced by `WavVibrationAdapter`).
        kurtosis_window: Sliding-window length in samples for rolling kurtosis.
            Must be odd and ≥ 5; defaults to 17 to give a meaningful tail at
            both 4 Hz and 16 Hz CSV cadences.

    Returns:
        ``(n_vib, 3, T_vib)`` float32 array. Channels:
          0 — raw amplitude (zero-mean per channel),
          1 — Hilbert envelope (analytic-signal magnitude),
          2 — rolling kurtosis (centred sliding window, edges zero-padded).
    """
    if accel_data.ndim != 2:
        raise ValueError("accel_data must be 2-D (n_vib, T_vib)")
    if kurtosis_window < 5 or kurtosis_window % 2 == 0:
        raise ValueError("kurtosis_window must be odd and >= 5")

    n_vib, T = int(accel_data.shape[0]), int(accel_data.shape[1])
    if T < kurtosis_window:
        # Tiny segment: fall back to a smaller centred window.
        kurtosis_window = max(5, T // 2 * 2 + 1)

    out = np.zeros((n_vib, 3, T), dtype=np.float32)
    half = kurtosis_window // 2

    for i in range(n_vib):
        x = accel_data[i].astype(np.float64)
        x = x - float(np.mean(x))
        out[i, 0] = x.astype(np.float32)

        # Hilbert envelope is well-defined for any 1-D real signal of length ≥ 2.
        if T >= 2:
            envelope = np.abs(hilbert(x)).astype(np.float32)
        else:
            envelope = np.abs(x).astype(np.float32)
        out[i, 1] = envelope

        # Rolling kurtosis with centred window.
        rk = np.zeros(T, dtype=np.float64)
        for t in range(half, T - half):
            window = x[t - half : t + half + 1]
            std = float(np.std(window))
            if std < 1e-12:
                rk[t] = 0.0
            else:
                rk[t] = float(_kurtosis(window, fisher=True, bias=True))
        out[i, 2] = rk.astype(np.float32)

    return out


__all__ = ["compute_vibration_input_stack"]
