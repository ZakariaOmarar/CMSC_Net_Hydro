from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.data import DataSegment
from src.features import VibrationEnvelopeExtractor


def _build_segment(*, metadata: dict | None = None) -> DataSegment:
    mic_sr = 16_000
    accel_sr = 4
    duration_s = 5.0

    t_mic = np.arange(int(mic_sr * duration_s), dtype=np.float64) / mic_sr
    t_accel = np.arange(int(accel_sr * duration_s), dtype=np.float64) / accel_sr

    mic = np.stack(
        [0.1 * np.sin(2 * np.pi * (200.0 + i) * t_mic) for i in range(4)],
        axis=0,
    )
    accel = np.stack(
        [
            10.0
            + 0.2 * np.sin(2 * np.pi * (0.3 + 0.05 * i) * t_accel)
            + 0.01 * i * t_accel
            for i in range(4)
        ],
        axis=0,
    )

    return DataSegment.from_arrays(
        mic_data=mic,
        accel_data=accel,
        start_time=datetime(2026, 3, 26, tzinfo=timezone.utc),
        mic_sr=mic_sr,
        accel_sr=accel_sr,
        metadata=metadata or {},
    )


def test_vibration_envelope_feature_vector_has_18_per_sensor() -> None:
    segment = _build_segment()
    extractor = VibrationEnvelopeExtractor()

    frame = extractor.extract(segment)

    keys = sorted(k for k in frame.features if k.endswith("_vibration_features"))
    assert len(keys) == segment.n_accel_channels

    for key in keys:
        vec = np.asarray(frame.features[key], dtype=np.float64)
        assert vec.shape == (18,)
        assert np.all(np.isfinite(vec))
        assert np.allclose(vec[-4:], 0.0)


def test_vibration_envelope_transition_features_follow_explicit_flag() -> None:
    segment = _build_segment(metadata={"is_transition_window": True})
    extractor = VibrationEnvelopeExtractor()

    frame = extractor.extract(segment)
    vec = np.asarray(frame.features["accel_0_vibration_features"], dtype=np.float64)

    transition = vec[-4:]
    assert not np.allclose(transition, 0.0)


def test_vibration_envelope_transition_features_follow_mask_and_window_index() -> None:
    segment = _build_segment(
        metadata={
            "transition_mask": [False, True, False],
            "window_index": 1,
        }
    )
    extractor = VibrationEnvelopeExtractor()

    frame = extractor.extract(segment)
    vec = np.asarray(frame.features["accel_1_vibration_features"], dtype=np.float64)

    transition = vec[-4:]
    assert not np.allclose(transition, 0.0)
