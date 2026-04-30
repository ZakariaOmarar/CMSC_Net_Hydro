"""Layer 5 — Per-mode robust baseline and anomaly scoring.

For each operating mode, maintains a rolling 24-hour within-mode baseline
and computes a log-space anomaly score for each timestep.

Scoring method
--------------
Reconstruction errors are log1p-transformed (they are approximately lognormal).
Alert thresholds are set by rolling percentile of the baseline:
  95th percentile → "watch" (5% of normal blocks)
  99th percentile → "alert" (1% of normal blocks)

Percentile thresholds give a guaranteed false-alarm rate regardless of the
error distribution shape, unlike fixed Gaussian sigma thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.modeling.mode.p4_smoother.topology import STATE_NAMES

# 1.4826 converts MAD to equivalent Gaussian σ (used for z-score display only)
_MAD_SCALE = 1.4826

WATCH_SIGMA = 4.0
ALERT_SIGMA = 6.0

# Rolling baseline window: 24 hours × 3600 s/h = 86,400 mode-seconds
BASELINE_WINDOW_S = 86_400

# Percentile thresholds for alert levels
WATCH_PERCENTILE = 95.0   # 5% of baseline blocks → watch
ALERT_PERCENTILE = 99.0   # 1% of baseline blocks → alert


def compute_anomaly_scores(
    reconstruction_errors: np.ndarray,
    smoothed_labels: np.ndarray,
    *,
    baseline_window_s: int = BASELINE_WINDOW_S,
    watch_sigma: float = WATCH_SIGMA,
    alert_sigma: float = ALERT_SIGMA,
    min_baseline_samples: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-second anomaly z-scores from AE reconstruction errors.

    For each timestep t in mode k, computes the robust z-score against the
    trailing 24-hour within-mode baseline.

    Parameters
    ----------
    reconstruction_errors : (T,) float32 — MSE errors from per-mode AEs
    smoothed_labels : (T,) int32 — state indices from Layer 4
    baseline_window_s : rolling window length in seconds (default 24 h)
    min_baseline_samples : minimum samples required to compute a score

    Returns
    -------
    z_scores : (T,) float32 — robust z-score per timestep (NaN where baseline insufficient)
    alert_level : (T,) int8 — 0=normal, 1=watch, 2=alert
    """
    T = len(reconstruction_errors)
    z_scores = np.full(T, np.nan, dtype=np.float32)
    alert_level = np.zeros(T, dtype=np.int8)

    for state_id in range(len(STATE_NAMES)):
        in_mode = smoothed_labels == state_id
        mode_timesteps = np.where(in_mode)[0]

        if len(mode_timesteps) < min_baseline_samples:
            continue

        # For each within-mode timestep, compute baseline from trailing window
        for i, t in enumerate(mode_timesteps):
            # Collect within-mode timesteps in [t - baseline_window_s, t - 1]
            t0 = max(0, t - baseline_window_s)
            window_mask = in_mode.copy()
            window_mask[:t0] = False
            window_mask[t:] = False
            window_errors = reconstruction_errors[window_mask]
            window_errors = window_errors[np.isfinite(window_errors)]

            if len(window_errors) < min_baseline_samples:
                continue

            baseline_median = float(np.median(window_errors))
            mad = float(np.median(np.abs(window_errors - baseline_median)))
            sigma = _MAD_SCALE * mad

            if sigma < 1e-12:
                continue

            err = float(reconstruction_errors[t])
            if not np.isfinite(err):
                continue

            z = (err - baseline_median) / sigma
            z_scores[t] = float(z)

            if z >= alert_sigma:
                alert_level[t] = 2
            elif z >= watch_sigma:
                alert_level[t] = 1

    return z_scores, alert_level


def compute_anomaly_scores_fast(
    reconstruction_errors: np.ndarray,
    smoothed_labels: np.ndarray,
    *,
    baseline_window_s: int = BASELINE_WINDOW_S,
    watch_sigma: float = WATCH_SIGMA,
    alert_sigma: float = ALERT_SIGMA,
    min_baseline_samples: int = 300,
    watch_percentile: float = WATCH_PERCENTILE,
    alert_percentile: float = ALERT_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling percentile-based anomaly scoring in log1p-error space.

    Reconstruction errors are log1p-transformed before scoring because their
    distribution is approximately lognormal (heavy right tail). Alert thresholds
    use rolling percentiles rather than Gaussian sigma, which guarantees the
    target false-alarm rate regardless of distribution shape:

      watch_percentile=95  →  5% of baseline blocks trigger watch
      alert_percentile=99  →  1% of baseline blocks trigger alert

    The z_scores output still uses median+MAD for diagnostic display.
    Baseline is recomputed every 60 mode-timesteps for efficiency.

    Returns
    -------
    z_scores : (T,) float32 — log-space robust z-score (diagnostic only)
    alert_level : (T,) int8 — 0=normal, 1=watch, 2=alert
    """
    T = len(reconstruction_errors)
    z_scores = np.full(T, np.nan, dtype=np.float32)
    alert_level = np.zeros(T, dtype=np.int8)

    for state_id in range(len(STATE_NAMES)):
        in_mode = smoothed_labels == state_id

        mode_t = np.where(in_mode)[0]
        if len(mode_t) < min_baseline_samples:
            continue

        # Log1p-transform: reconstruction errors are approximately lognormal
        raw_errors = reconstruction_errors[mode_t]
        mode_errors = np.log1p(np.maximum(raw_errors, 0.0).astype(np.float64))
        n = len(mode_t)

        # Recompute baseline every 60 mode-timesteps (1-minute granularity)
        update_interval = 60
        current_median = np.nan
        current_sigma = np.nan
        watch_thr = np.nan
        alert_thr = np.nan

        for i in range(n):
            if i % update_interval == 0:
                start = max(0, i - baseline_window_s)
                window = mode_errors[start:i]
                finite = window[np.isfinite(window)]
                if len(finite) >= min_baseline_samples:
                    current_median = float(np.percentile(finite, 50))
                    mad = float(np.median(np.abs(finite - current_median)))
                    current_sigma = _MAD_SCALE * mad
                    watch_thr = float(np.percentile(finite, watch_percentile))
                    alert_thr = float(np.percentile(finite, alert_percentile))
                else:
                    current_median = current_sigma = watch_thr = alert_thr = np.nan

            if np.isnan(current_median) or not np.isfinite(mode_errors[i]):
                continue

            # z-score for diagnostic display (median+MAD in log space)
            if current_sigma > 1e-12:
                z = (mode_errors[i] - current_median) / current_sigma
            else:
                z = 0.0
            z_scores[mode_t[i]] = float(z)

            # Alert level based on percentile thresholds (distribution-agnostic)
            if not np.isnan(alert_thr) and mode_errors[i] >= alert_thr:
                alert_level[mode_t[i]] = 2
            elif not np.isnan(watch_thr) and mode_errors[i] >= watch_thr:
                alert_level[mode_t[i]] = 1

    return z_scores, alert_level


def save_anomaly_scores(
    z_scores: np.ndarray,
    alert_level: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "anomaly_z_scores.npy", z_scores)
    np.save(output_dir / "anomaly_alert_level.npy", alert_level)
