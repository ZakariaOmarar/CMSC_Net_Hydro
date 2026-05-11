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
    standardize: bool = True,
    standardization_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the V1 vibration encoder input.

    Args:
        accel_data: ``(n_vib, T_vib)`` raw amplitude (already at the per-dataset
            target sample rate produced by `WavVibrationAdapter`).
        kurtosis_window: Sliding-window length in samples for rolling kurtosis.
            Must be odd and ≥ 5; defaults to 17 to give a meaningful tail at
            both 4 Hz and 16 Hz CSV cadences.
        standardize: When True (default), the amplitude and Hilbert-envelope
            channels are z-score-normalised.  See `standardization_stats`
            for the granularity options.  Rolling kurtosis is dimensionless
            and never re-standardised.  Set False to reproduce the
            pre-2026-05 behaviour for legacy comparisons.
        standardization_stats: When `standardize=True` and this is None, the
            standardisation is computed **per-recording-per-channel**
            (legacy default): each channel of each recording is z-scored to
            zero-mean unit-variance.  This bridges cross-dataset scale (D1
            peak amps ~ 1 000 ADU vs D4 raw ~ 18 000 ADU) but **strips the
            intra-recording amplitude information** (Pump amplitude > Standstill
            amplitude *within* a recording is normalised away).  Pass a
            `(mean, std)` tuple of shape `(n_vib_channels,)` to apply a
            **per-dataset** z-score instead — preserves intra-recording
            amplitude differences while still bridging cross-dataset scale.
            Per-dataset stats should be precomputed by the caller from the
            full healthy training pool of one dataset.

    Returns:
        ``(n_vib, 3, T_vib)`` float32 array. Channels:
          0 — amplitude (zero-mean; unit-variance when standardize=True),
          1 — Hilbert envelope (zero-mean unit-variance when standardize=True),
          2 — rolling kurtosis (centred sliding window, edges zero-padded;
              dimensionless, never re-standardised).
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
    eps = 1e-8

    # Per-dataset stats path: if the caller supplied (mean, std) of shape
    # (n_vib,), we standardise the amplitude using THOSE statistics so
    # intra-recording amplitude differences (Pump > Standstill) survive.
    use_dataset_stats = (
        standardize and standardization_stats is not None
    )
    ds_mean = ds_std = None
    if use_dataset_stats:
        ds_mean = np.asarray(standardization_stats[0], dtype=np.float64)
        ds_std = np.asarray(standardization_stats[1], dtype=np.float64)
        if ds_mean.shape != (n_vib,) or ds_std.shape != (n_vib,):
            raise ValueError(
                f"standardization_stats must have shape ({n_vib},); "
                f"got mean={ds_mean.shape}, std={ds_std.shape}"
            )

    for i in range(n_vib):
        x = accel_data[i].astype(np.float64)

        if use_dataset_stats:
            # Per-dataset standardisation: subtract dataset mean, divide by
            # dataset std.  Intra-recording amplitude scale is preserved.
            amp = (x - float(ds_mean[i])) / max(float(ds_std[i]), eps)
            # Hilbert envelope on the original (un-standardised) waveform,
            # then z-scored against the dataset envelope distribution
            # (approximated by |x − ds_mean| / ds_std for stability).
            if T >= 2:
                envelope_raw = np.abs(hilbert(x - float(ds_mean[i])))
            else:
                envelope_raw = np.abs(x - float(ds_mean[i]))
            # Standardise envelope to zero-mean unit-variance using its own
            # per-recording stats — envelope is non-negative so per-dataset
            # stats are not directly applicable.
            env_mean = float(np.mean(envelope_raw))
            env_std = float(np.std(envelope_raw))
            envelope = envelope_raw - env_mean
            if env_std > eps:
                envelope = envelope / env_std
            out[i, 0] = amp.astype(np.float32)
            out[i, 1] = envelope.astype(np.float32)
            # Update `x` for the kurtosis computation (zero-mean).
            x = x - float(ds_mean[i])
        else:
            # Per-recording standardisation (legacy path): zero-mean and
            # unit-variance per channel per recording.
            x = x - float(np.mean(x))
            amp = x.copy()
            if standardize:
                std_x = float(np.std(amp))
                if std_x > eps:
                    amp = amp / std_x
            out[i, 0] = amp.astype(np.float32)

            if T >= 2:
                envelope = np.abs(hilbert(x))
            else:
                envelope = np.abs(x)
            if standardize:
                env_mean = float(np.mean(envelope))
                env_std = float(np.std(envelope))
                envelope = envelope - env_mean
                if env_std > eps:
                    envelope = envelope / env_std
            out[i, 1] = envelope.astype(np.float32)

        # Rolling kurtosis with centred window — dimensionless, kept as-is.
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
