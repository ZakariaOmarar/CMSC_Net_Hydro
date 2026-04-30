"""Smoke tests for the V1 encoder-input feature extractors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.audio_spectral import (
    compute_encoder_input_stack,
    compute_log_mel_spectrogram,
)
from src.features.vibration_temporal import compute_vibration_input_stack
from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(name: str) -> DatasetSpec:
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{name}.yaml")
    return DatasetSpec(
        id=spec.id,
        root=REPO_ROOT / spec.root,
        n_mics=spec.n_mics,
        n_vibrations=spec.n_vibrations,
        accel_target_sr=spec.accel_target_sr,
        position_source=(
            REPO_ROOT / spec.position_source
            if spec.position_source not in ("default", "rowii")
            else spec.position_source
        ),
        label_scheme=spec.label_scheme,
        extra=spec.extra,
    )


def _short_segment(name: str, max_seconds: float = 4.0):
    """Load one segment and crop to the first ``max_seconds`` to keep feature
    extraction tests fast."""
    loader = TestDatasetLoader(_spec(name))
    seg = loader.list_segments()[0]
    n_mic_keep = int(round(max_seconds * seg.segment.mic_sample_rate))
    n_vib_keep = max(8, int(round(max_seconds * seg.segment.accel_sample_rate)))
    mic = seg.segment.mic_data[:, :n_mic_keep]
    vib = seg.segment.accel_data[:, :n_vib_keep]
    return seg, mic, vib


def test_log_mel_single_channel_shape() -> None:
    rng = np.random.default_rng(0)
    fs = 16_000
    x = rng.standard_normal(fs * 2)  # 2 s of noise
    mel = compute_log_mel_spectrogram(x, fs, n_mels=64, n_fft=1024, hop_length=256)
    assert mel.ndim == 2
    assert mel.shape[0] == 64
    assert mel.shape[1] >= 1
    assert np.all(np.isfinite(mel))
    assert np.all(mel >= 0.0)  # log1p of non-negative power is non-negative


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_encoder_input_stack_shape(name: str) -> None:
    seg, mic, _ = _short_segment(name)
    stack = compute_encoder_input_stack(
        mic,
        fs=seg.segment.mic_sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        cwt_n_scales=64,
    )
    assert stack.ndim == 4
    assert stack.shape[0] == seg.segment.n_mic_channels
    assert stack.shape[1] == 2  # (log-mel, CWT)
    assert stack.shape[2] == 64  # frequency axis
    assert stack.shape[3] >= 1
    assert np.all(np.isfinite(stack))
    # log-mel and CWT live on the same (F, T) grid
    assert stack[0, 0].shape == stack[0, 1].shape


@pytest.mark.parametrize("name", ["d1", "d2", "d3"])
def test_vibration_input_stack_shape(name: str) -> None:
    seg, _, vib = _short_segment(name)
    stack = compute_vibration_input_stack(vib, kurtosis_window=5)
    assert stack.ndim == 3
    assert stack.shape[0] == seg.segment.n_accel_channels
    assert stack.shape[1] == 3  # (amplitude, envelope, rolling kurtosis)
    assert stack.shape[2] == vib.shape[1]
    assert np.all(np.isfinite(stack))
    # Envelope is non-negative.
    assert np.all(stack[:, 1, :] >= 0.0)
