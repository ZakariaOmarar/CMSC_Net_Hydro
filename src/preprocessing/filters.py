"""Filtering utilities for thesis multimodal data."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..data import DataSegment
from ..exceptions import FilterError


class DCRemover:
    """Remove per-channel DC offset."""

    def process(self, segment: DataSegment) -> DataSegment:
        mic_out = segment.mic_data - segment.mic_data.mean(axis=1, keepdims=True)
        accel_out = segment.accel_data - segment.accel_data.mean(axis=1, keepdims=True)

        return DataSegment(
            mic_data=mic_out,
            accel_data=accel_out,
            mic_sample_rate=segment.mic_sample_rate,
            accel_sample_rate=segment.accel_sample_rate,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            channel_names=segment.channel_names,
            metadata={**segment.metadata, "dc_removed": True},
        )


class BandpassFilter:
    """Apply modality-specific Butterworth bandpass filters.

    For low-rate vibration streams, accel filtering is skipped if cutoff ranges
    exceed Nyquist constraints.
    """

    def __init__(
        self,
        mic_low: float = 20.0,
        mic_high: float = 7_000.0,
        accel_low: float = 0.1,
        accel_high: float = 1.5,
        order: int = 4,
    ) -> None:
        self._mic_low = mic_low
        self._mic_high = mic_high
        self._accel_low = accel_low
        self._accel_high = accel_high
        self._order = order

    def process(self, segment: DataSegment) -> DataSegment:
        mic_sos = _design_bandpass(
            self._mic_low,
            self._mic_high,
            segment.mic_sample_rate,
            self._order,
        )
        mic_out = _apply_sos(segment.mic_data, mic_sos)

        accel_note = "filtered"
        try:
            accel_sos = _design_bandpass(
                self._accel_low,
                self._accel_high,
                segment.accel_sample_rate,
                self._order,
            )
            accel_out = _apply_sos(segment.accel_data, accel_sos)
        except FilterError:
            accel_out = segment.accel_data
            accel_note = "skipped"

        return DataSegment(
            mic_data=mic_out,
            accel_data=accel_out,
            mic_sample_rate=segment.mic_sample_rate,
            accel_sample_rate=segment.accel_sample_rate,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            channel_names=segment.channel_names,
            metadata={
                **segment.metadata,
                "bandpass_mic": f"{self._mic_low}-{self._mic_high}Hz",
                "bandpass_accel": f"{self._accel_low}-{self._accel_high}Hz",
                "bandpass_accel_status": accel_note,
            },
        )


def _design_bandpass(low: float, high: float, sr: int, order: int) -> np.ndarray:
    nyq = sr / 2.0
    if low <= 0:
        raise FilterError(f"Low cutoff must be > 0, got {low}")
    if high >= nyq:
        raise FilterError(f"High cutoff {high} >= Nyquist {nyq}")
    if low >= high:
        raise FilterError(f"Low cutoff {low} must be lower than high cutoff {high}")

    try:
        return butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    except Exception as exc:
        raise FilterError(f"Bandpass design failed: {exc}") from exc


def _apply_sos(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    out = np.empty_like(data)
    for ch in range(data.shape[0]):
        try:
            out[ch] = sosfiltfilt(sos, data[ch])
        except ValueError as exc:
            # Very short windows can violate filtfilt pad-length requirements.
            if "must be greater than padlen" in str(exc):
                out[ch] = data[ch]
            else:
                raise FilterError(f"Filter application failed: {exc}") from exc
    return out
