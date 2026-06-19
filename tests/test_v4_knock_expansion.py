"""Tests for per-knock V4 expansion + array-footprint outlier classification.

These run without the recorded datasets: the builder test synthesises a tiny
two-knock recording so the leakage / multi-sample behaviour is exercised in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import torch

from src.data import DataSegment
from src.ingestion.test_dataset_loader import TestDatasetSegment
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.localization import (
    GridSpec,
    KnockEventConfig,
    array_sensor_xyz,
    assert_no_position_leak,
    classify_position,
    classify_positions,
    precompute_v4_knock_event_samples,
)


# --------------------------------------------------------------------------- #
# array_geometry
# --------------------------------------------------------------------------- #
def _unit_cube_sensors() -> np.ndarray:
    return np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float64,
    )


def test_classify_position_inside_and_outside() -> None:
    sensors = _unit_cube_sensors()
    inside = classify_position([0.5, 0.5, 0.5], sensors, margin_m=0.0)
    assert inside.inside and inside.signed_distance_m < 0
    assert inside.method == "convex_hull"

    outside = classify_position([2.0, 0.5, 0.5], sensors, margin_m=0.0)
    assert not outside.inside
    assert outside.signed_distance_m > 0.9  # ~1 m beyond the +x face


def test_classify_position_margin_admits_near_boundary() -> None:
    sensors = _unit_cube_sensors()
    p = [1.03, 0.5, 0.5]  # 3 cm outside the +x face
    assert not classify_position(p, sensors, margin_m=0.0).inside
    assert classify_position(p, sensors, margin_m=0.05).inside


def test_classify_position_coplanar_falls_back_to_bbox() -> None:
    # All sensors at z=0 → degenerate 3-D hull → bounding-box fallback.
    coplanar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    v = classify_position([0.5, 0.5, 0.2], coplanar, margin_m=0.0)
    assert v.method == "bounding_box"
    # z=0.2 is outside the zero-thickness box, so it reads as just-outside.
    assert v.signed_distance_m > 0


def test_array_sensor_xyz_unions_modalities() -> None:
    mic = np.zeros((4, 3))
    vib = np.ones((3, 3))
    assert array_sensor_xyz(mic, vib).shape == (7, 3)


def test_classify_positions_aggregates_inside_if_any() -> None:
    sensors = _unit_cube_sensors()
    far = sensors + np.array([10.0, 0, 0])  # a second, shifted array
    pos = (0.5, 0.5, 0.5)
    records = [(pos, sensors[:4], sensors[4:]), (pos, far[:4], far[4:])]
    out = classify_positions(records, margin_m=0.0)
    key = (0.5, 0.5, 0.5)
    assert out[key]["inside"] is True  # inside for the first array
    assert out[key]["n_recordings"] == 2


# --------------------------------------------------------------------------- #
# leak guard
# --------------------------------------------------------------------------- #
def _fake_sample(pos):
    from src.modeling.localization import V4Sample

    return V4Sample(
        srp_volume=np.zeros((4, 4, 2), np.float32),
        tdoa_tokens=np.zeros((0, 8), np.float32),
        context=np.zeros(8, np.float32),
        x_for_v3=np.zeros(8, np.float32),
        target_xyz=np.asarray(pos, np.float32),
        scada=None, mode_label=None, recording_id="r", source_dir="d", dataset_id="x",
    )


def test_assert_no_position_leak_raises_on_overlap() -> None:
    tr = [_fake_sample((0.1, 0.0, 0.0))]
    va = [_fake_sample((0.1, 0.0, 0.0))]
    try:
        assert_no_position_leak(tr, va)
        raised = False
    except AssertionError:
        raised = True
    assert raised
    # disjoint positions must pass (no exception raised)
    assert_no_position_leak([_fake_sample((0.1, 0, 0))], [_fake_sample((0.2, 0, 0))])


# --------------------------------------------------------------------------- #
# per-knock builder on a synthetic two-knock recording
# --------------------------------------------------------------------------- #
def _two_knock_segment() -> TestDatasetSegment:
    rng = np.random.default_rng(0)
    mic_sr, accel_sr = 16000, 376
    dur = 2.0
    n_mic = int(dur * mic_sr)
    n_acc = int(dur * accel_sr)
    n_mic_ch, n_acc_ch = 4, 4

    mic = (0.01 * rng.standard_normal((n_mic_ch, n_mic))).astype(np.float32)
    acc = (0.01 * rng.standard_normal((n_acc_ch, n_acc))).astype(np.float32)
    for t in (0.5, 1.2):  # two distinct knocks
        m0 = int(t * mic_sr)
        mic[:, m0 : m0 + int(0.03 * mic_sr)] += 8.0 * rng.standard_normal((n_mic_ch, int(0.03 * mic_sr))).astype(np.float32)
        a0 = int(t * accel_sr)
        acc[:, a0 : a0 + 3] += 8.0 * rng.standard_normal((n_acc_ch, 3)).astype(np.float32)

    seg = DataSegment.from_arrays(
        mic_data=mic, accel_data=acc,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        mic_sr=mic_sr, accel_sr=accel_sr, metadata={},
    )
    mic_xyz = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0.2, 0.2, 0]], np.float64)
    vib_xyz = np.array([[0.05, 0.05, 0], [0.15, 0.05, 0], [0.05, 0.15, 0], [0.15, 0.15, 0]], np.float64)
    return TestDatasetSegment(
        segment=seg, mic_positions=mic_xyz, vib_positions=vib_xyz,
        mic_ids=("A", "B", "C", "D"), vib_ids=("a", "b", "c", "d"),
        mode_label=None, op_condition=None, spatial_label=(0.1, 0.1, 0.0),
        dataset_id="d5", recording_id="synthetic_knock", source_dir="syn",
        is_anomaly=True,
    )


def _smoke_v2() -> tuple[V2FusionEncoder, V2SSLConfig]:
    torch.manual_seed(0)
    cfg = V2SSLConfig(
        window_seconds=0.5, window_stride_seconds=0.25,
        feature_dim=32, embed_dim=32, n_heads=2, proj_dim=16,
        epochs=1, batch_size=8, val_ratio=0.5,
        n_mels=32, n_fft=256, hop_length=128, use_cwt=False,
        gain_jitter_db=0.0, channel_dropout_p=0.0,
        spec_augment_freq_mask=0, spec_augment_time_mask=0, seed=0,
    )
    return V2FusionEncoder(feature_dim=32, embed_dim=32, n_heads=2), cfg


def test_knock_builder_emits_multiple_samples_per_recording() -> None:
    encoder, v2_cfg = _smoke_v2()
    grid = GridSpec(lo=(-0.1, -0.1, -0.05), hi=(0.3, 0.3, 0.2), n=(8, 8, 4))
    seg = _two_knock_segment()
    samples = precompute_v4_knock_event_samples(
        encoder, [seg], v2_cfg=v2_cfg, grid=grid,
        cfg=KnockEventConfig(crop_seconds=0.12, noise_floor_mult=3.0),
        device="cpu",
    )
    # Two knocks → at least two transient-centred samples.
    assert len(samples) >= 2
    for s in samples:
        assert s.srp_volume.shape == (8, 8, 4)
        assert np.all(np.isfinite(s.srp_volume))
        assert s.tdoa_tokens.shape[1] == 8
        assert tuple(np.round(s.target_xyz, 3)) == (0.1, 0.1, 0.0)
        assert np.isfinite(s.context).all()
    # Event centres should bracket the two injected knocks (~0.5 s and ~1.2 s).
    centres = sorted(s.window_start_s for s in samples)
    assert any(abs(c - 0.5) < 0.2 for c in centres)
    assert any(abs(c - 1.2) < 0.2 for c in centres)
