"""Normalization methods for multimodal segments."""

from __future__ import annotations

import numpy as np

from ..data import DataSegment
from ..exceptions import PreprocessingError


class Normalizer:
    """Per-channel normalization for mic and vibration streams."""

    def __init__(self, method: str = "zscore") -> None:
        valid = {"zscore", "minmax", "peak"}
        if method not in valid:
            raise PreprocessingError(f"Unknown normalization method {method!r}")
        self._method = method

    def process(self, segment: DataSegment) -> DataSegment:
        mic_out = self._normalize(segment.mic_data)
        accel_out = self._normalize(segment.accel_data)

        return DataSegment(
            mic_data=mic_out,
            accel_data=accel_out,
            mic_sample_rate=segment.mic_sample_rate,
            accel_sample_rate=segment.accel_sample_rate,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            channel_names=segment.channel_names,
            metadata={**segment.metadata, "normalization": self._method},
        )

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self._method == "zscore":
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            std = np.where(std == 0.0, 1.0, std)
            return (data - mean) / std

        if self._method == "minmax":
            lo = data.min(axis=1, keepdims=True)
            hi = data.max(axis=1, keepdims=True)
            span = np.where((hi - lo) == 0.0, 1.0, hi - lo)
            return 2.0 * (data - lo) / span - 1.0

        peak = np.max(np.abs(data), axis=1, keepdims=True)
        peak = np.where(peak == 0.0, 1.0, peak)
        return data / peak
