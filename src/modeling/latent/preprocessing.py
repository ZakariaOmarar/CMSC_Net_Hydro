"""Signal preprocessing applied to each DataSegment before feature extraction.

Every raw recording segment passes through four steps in order:
1. SensitivityCalibrator — corrects per-channel voltage sensitivity to physical units.
2. DCRemover — subtracts the running mean to eliminate sensor DC bias.
3. BandpassFilter — removes infrasound and ultrasound outside the diagnostically
   relevant band.
4. Normalizer (z-score) — equalizes channel-wise energy so no single sensor dominates
   the feature space.

ModelVariant gates which feature extractors are activated downstream:
  ``"mic_only"``         — acoustic channels only, for ablation experiments.
  ``"mic_vibration"``    — full multimodal setup (recommended).
"""

from __future__ import annotations

from typing import Literal

from ...data import DataSegment
from ...exceptions import ModelInferenceError
from ...preprocessing import (
    BandpassFilter,
    DCRemover,
    Normalizer,
    SensitivityCalibrator,
    WindowedSegmenter,
)

ModelVariant = Literal["mic_only", "mic_vibration"]
VALID_MODEL_VARIANTS: tuple[ModelVariant, ...] = ("mic_only", "mic_vibration")


def validate_variant(mode: str) -> ModelVariant:
    if mode not in VALID_MODEL_VARIANTS:
        raise ModelInferenceError(
            f"Unknown detect mode {mode!r}; expected one of {VALID_MODEL_VARIANTS}"
        )
    return mode


def preprocess_segment(segment: DataSegment) -> DataSegment:
    calibrator = SensitivityCalibrator()
    dc_remover = DCRemover()
    bandpass = BandpassFilter()
    normalizer = Normalizer(method="zscore")

    segment = calibrator.process(segment)
    segment = dc_remover.process(segment)
    segment = bandpass.process(segment)
    segment = normalizer.process(segment)
    return segment


def build_segmenter(window_s: float, overlap: float) -> WindowedSegmenter:
    return WindowedSegmenter(window_s=window_s, overlap=overlap)


def select_single_mic(segment: DataSegment, mic_channel_index: int) -> DataSegment:
    if mic_channel_index < 0 or mic_channel_index >= segment.n_mic_channels:
        raise ModelInferenceError(
            f"mic_channel_index {mic_channel_index} out of range for "
            f"{segment.n_mic_channels} channels"
        )

    mic_data = segment.mic_data[mic_channel_index : mic_channel_index + 1].copy()
    channel_names = (
        segment.mic_channel_names[mic_channel_index],
        *segment.accel_channel_names,
    )

    return DataSegment(
        mic_data=mic_data,
        accel_data=segment.accel_data.copy(),
        mic_sample_rate=segment.mic_sample_rate,
        accel_sample_rate=segment.accel_sample_rate,
        start_time=segment.start_time,
        duration_s=segment.duration_s,
        channel_names=channel_names,
        metadata={**segment.metadata, "mic_channel_index": mic_channel_index},
    )
