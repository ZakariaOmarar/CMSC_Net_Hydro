from __future__ import annotations

import numpy as np

from src.features import (
    build_cwt_mfcc_encoder_input,
    compute_cwt_scalogram_stack,
    compute_mfcc_stack,
    compute_stft_stack,
)


def _build_test_mics(
    n_mics: int = 3, fs: int = 16_000, duration_s: float = 1.0
) -> np.ndarray:
    t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
    base = 0.2 * np.sin(2 * np.pi * 240.0 * t) + 0.05 * np.sin(2 * np.pi * 1_200.0 * t)
    return np.stack([np.roll(base, i * 11) for i in range(n_mics)], axis=0)


def test_cwt_scalogram_stack_shape() -> None:
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=0.5)

    cwt = compute_cwt_scalogram_stack(mic, fs, n_scales=32)

    assert cwt.shape == (mic.shape[0], 32, mic.shape[1])
    assert np.all(np.isfinite(cwt))
    assert np.all(cwt >= 0.0)


def test_mfcc_stack_with_deltas_shape() -> None:
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=1.0)

    mfcc = compute_mfcc_stack(
        mic,
        fs,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )

    assert mfcc.shape[0] == mic.shape[0]
    assert mfcc.shape[1] == 60
    assert mfcc.shape[2] > 0
    assert np.all(np.isfinite(mfcc))


def test_build_cwt_mfcc_encoder_input_aligns_and_concatenates() -> None:
    fs = 16_000
    mic = _build_test_mics(fs=fs, duration_s=1.0)

    cwt = compute_cwt_scalogram_stack(mic, fs, n_scales=32)
    mfcc = compute_mfcc_stack(
        mic,
        fs,
        n_mfcc=20,
        n_fft=1024,
        hop_length=256,
    )

    fused = build_cwt_mfcc_encoder_input(cwt, mfcc)

    assert fused.shape[0] == mic.shape[0] * 2
    assert fused.shape[1] == cwt.shape[1]
    assert fused.shape[2] == min(cwt.shape[2], mfcc.shape[2])
    assert np.all(np.isfinite(fused))


def test_stft_stack_baseline_shape() -> None:
    mic = _build_test_mics(duration_s=1.0)

    stft = compute_stft_stack(mic, n_fft=1024, hop_length=256)

    assert stft.shape[0] == mic.shape[0]
    assert stft.shape[1] == 513
    assert stft.shape[2] > 0
    assert np.all(np.isfinite(stft))
