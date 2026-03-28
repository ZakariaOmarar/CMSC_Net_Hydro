"""Vibration dominant-frequency features from adapter metadata.

The ingestion adapter stores per-sensor dominant vibration frequency traces under
segment.metadata["vibration_frequencies"]. This extractor summarizes those traces
per analysis window.
"""

from __future__ import annotations

import numpy as np

from ..data import DataSegment
from .feature_frame import FeatureFrame


class VibrationFrequencyExtractor:
    """Summarize dominant vibration frequency traces into an 8-element statistics vector.

    The ingestion adapter stores a per-sensor dominant-frequency array under
    ``segment.metadata["vibration_frequencies"]`` with shape
    ``(n_accel_channels, total_vib_samples)``. This extractor identifies the
    samples that fall within the current window using the ``window_index`` and
    ``overlap`` metadata keys, then computes ``[mean, std, min, max, p10, p50,
    p90, slope]`` for each accelerometer channel.

    The dominant vibration frequency is a coarse but reliable operating-point
    indicator: it clusters tightly around narrow bands in Turbine and Pump modes
    while spreading widely during Standstill and fault events.
    Falls back to a zero vector when frequency metadata is absent.

    Args:
        metadata_key: Key under which the ingestion adapter stores the
            frequency trace array in ``segment.metadata``.
    """

    def __init__(self, metadata_key: str = "vibration_frequencies") -> None:
        self._metadata_key = metadata_key

    def extract(self, segment: DataSegment) -> FeatureFrame:
        features: dict[str, float | np.ndarray] = {}

        freq_data = segment.metadata.get(self._metadata_key)
        freq_arr = self._resolve_window_frequency_slice(segment, freq_data)

        if freq_arr is None:
            # Graceful fallback keeps pipeline robust when frequency metadata is absent.
            for name in segment.accel_channel_names:
                features[f"{name}_vib_freq_stats"] = np.zeros((8,), dtype=np.float64)
        else:
            for i, name in enumerate(segment.accel_channel_names):
                if i >= freq_arr.shape[0]:
                    vec = np.zeros((8,), dtype=np.float64)
                else:
                    vec = self._summarize(freq_arr[i])
                features[f"{name}_vib_freq_stats"] = vec

        return FeatureFrame(
            features=features,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            segment_metadata=dict(segment.metadata),
        )

    def _resolve_window_frequency_slice(
        self,
        segment: DataSegment,
        freq_data: object,
    ) -> np.ndarray | None:
        if freq_data is None:
            return None

        arr = np.asarray(freq_data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return None

        window_index = int(segment.metadata.get("window_index", 0))
        window_s = float(segment.metadata.get("window_s", segment.duration_s))
        overlap = float(segment.metadata.get("overlap", 0.0))
        fs = float(segment.accel_sample_rate)

        n_win = max(1, int(round(window_s * fs)))
        hop = max(1, int(round(n_win * (1.0 - overlap))))
        start = max(0, window_index * hop)
        end = min(arr.shape[1], start + n_win)
        if start >= arr.shape[1] or end <= start:
            return None

        return arr[:, start:end]

    @staticmethod
    def _summarize(x: np.ndarray) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64)
        if vec.size == 0:
            return np.zeros((8,), dtype=np.float64)

        mean = float(np.mean(vec))
        std = float(np.std(vec))
        min_v = float(np.min(vec))
        max_v = float(np.max(vec))
        p10 = float(np.percentile(vec, 10.0))
        p50 = float(np.percentile(vec, 50.0))
        p90 = float(np.percentile(vec, 90.0))
        if vec.size >= 2:
            slope = float(np.polyfit(np.arange(vec.size, dtype=np.float64), vec, 1)[0])
        else:
            slope = 0.0

        return np.asarray(
            [mean, std, min_v, max_v, p10, p50, p90, slope], dtype=np.float64
        )
