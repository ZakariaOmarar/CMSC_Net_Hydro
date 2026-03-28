"""Per-channel time-domain feature extraction for acoustic and vibration signals."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from ..data import DataSegment
from .feature_frame import FeatureFrame


class TimeDomainExtractor:
    """Per-channel time-domain statistics for one analysis window.

    Computes RMS, peak, crest factor, Fisher kurtosis, skewness, and zero-crossing
    rate for each requested channel group. These six scalars capture energy level,
    impulsiveness, and oscillation frequency — all of which shift distinctly between
    Turbine, Pump, and Standstill modes and are sensitive to early-stage faults.

    Args:
        channels: Which sensors to process. One of ``"all"``, ``"mic"``, or ``"accel"``.
    """

    def __init__(self, channels: str = "all") -> None:
        if channels not in {"all", "mic", "accel"}:
            raise ValueError("channels must be one of: all, mic, accel")
        self._channels = channels

    def extract(self, segment: DataSegment) -> FeatureFrame:
        features: dict[str, float | np.ndarray] = {}

        if self._channels in {"all", "mic"}:
            self._extract_channels(
                features,
                segment.mic_data,
                segment.mic_channel_names,
                segment.mic_sample_rate,
            )

        if self._channels in {"all", "accel"}:
            self._extract_channels(
                features,
                segment.accel_data,
                segment.accel_channel_names,
                segment.accel_sample_rate,
            )

        return FeatureFrame(
            features=features,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            segment_metadata=dict(segment.metadata),
        )

    @staticmethod
    def _extract_channels(
        out: dict[str, float | np.ndarray],
        data: np.ndarray,
        names: tuple[str, ...],
        sample_rate: int,
    ) -> None:
        for i, name in enumerate(names):
            x = data[i]
            n = x.shape[0]

            rms = float(np.sqrt(np.mean(x**2)))
            peak = float(np.max(np.abs(x)))
            crest = peak / rms if rms > 0 else 0.0

            k = kurtosis(x, fisher=True, bias=True)
            s = skew(x, bias=True)
            k = 0.0 if np.isnan(k) else float(k)
            s = 0.0 if np.isnan(s) else float(s)

            signs = np.sign(x)
            signs[signs == 0] = 1
            zcr = (
                float(np.sum(np.diff(signs) != 0)) / (n / sample_rate) if n > 0 else 0.0
            )

            out[f"{name}_rms"] = rms
            out[f"{name}_peak"] = peak
            out[f"{name}_crest_factor"] = crest
            out[f"{name}_kurtosis"] = k
            out[f"{name}_skewness"] = s
            out[f"{name}_zero_crossing_rate"] = zcr
