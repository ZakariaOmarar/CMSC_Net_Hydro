"""Acoustic time-frequency representations for encoder-oriented workflows.

CWT scalograms are the primary input planned for the cross-attention transformer
being developed in the thesis — they preserve both temporal and frequency structure
that plain STFT spectrograms smear around low-frequency turbine tones. MFCCs
provide a compact perceptual alternative. The STFT utility is kept for classical
baselines and direct comparison with published benchmarks.
"""

from __future__ import annotations

import numpy as np


def compute_cwt_scalogram(
    signal: np.ndarray,
    fs: int,
    *,
    wavelet: str = "cmor1.5-1.0",
    n_scales: int = 64,
    min_freq_hz: float = 20.0,
) -> np.ndarray:
    """Compute a log-compressed CWT scalogram for a single acoustic channel.

    Returns a 2D array with shape ``(n_scales, n_samples)``.
    """
    import pywt

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_scales <= 1:
        raise ValueError("n_scales must be > 1")
    if min_freq_hz <= 0:
        raise ValueError("min_freq_hz must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size == 0:
        raise ValueError("signal cannot be empty")

    nyquist = fs / 2.0
    if min_freq_hz >= nyquist:
        raise ValueError("min_freq_hz must be below Nyquist")

    freqs = np.geomspace(min_freq_hz, nyquist, n_scales)
    scales = pywt.frequency2scale(wavelet, freqs / fs)
    coefficients, _ = pywt.cwt(
        x,
        scales,
        wavelet,
        sampling_period=1.0 / fs,
    )

    scalogram = np.log1p(np.abs(coefficients))
    return scalogram.astype(np.float64)


def compute_cwt_scalogram_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    wavelet: str = "cmor1.5-1.0",
    n_scales: int = 64,
    min_freq_hz: float = 20.0,
) -> np.ndarray:
    """Compute CWT scalograms for all mic channels.

    Returns a 3D array with shape ``(n_mics, n_scales, n_samples)``.
    """
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_cwt_scalogram(
            data[i],
            fs,
            wavelet=wavelet,
            n_scales=n_scales,
            min_freq_hz=min_freq_hz,
        )
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def compute_mfcc_with_deltas(
    signal: np.ndarray,
    fs: int,
    *,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute MFCC, delta, and delta-delta for one channel.

    Returns a 2D array with shape ``(3 * n_mfcc, n_frames)``.
    """
    import librosa

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be > 0")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size < n_fft:
        raise ValueError("signal length must be >= n_fft")

    mfcc = librosa.feature.mfcc(
        y=x.astype(np.float32),
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([mfcc, delta, delta2], axis=0).astype(np.float64)


def compute_mfcc_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute stacked MFCC(+delta/+delta-delta) across mic channels.

    Returns a 3D array with shape ``(n_mics, 3 * n_mfcc, n_frames)``.
    """
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_mfcc_with_deltas(
            data[i],
            fs,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def compute_log_stft_spectrogram(
    signal: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log(1 + |STFT|) for one channel.

    This is provided as a classical baseline representation.
    """
    import librosa

    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size < n_fft:
        raise ValueError("signal length must be >= n_fft")

    stft = librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    return np.log1p(np.abs(stft)).astype(np.float64)


def compute_stft_stack(
    mic_data: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute stacked log-STFT representations for all mic channels."""
    data = _as_2d_channels(mic_data, name="mic_data")
    out = [
        compute_log_stft_spectrogram(data[i], n_fft=n_fft, hop_length=hop_length)
        for i in range(data.shape[0])
    ]
    return np.stack(out, axis=0)


def build_cwt_mfcc_encoder_input(
    cwt_stack: np.ndarray,
    mfcc_stack: np.ndarray,
    *,
    target_freq_bins: int | None = None,
    target_time_steps: int | None = None,
) -> np.ndarray:
    """Fuse CWT and MFCC stacks into one encoder tensor.

    The output shape is ``(2 * n_mics, freq_bins, time_steps)`` where the first
    ``n_mics`` channels are CWT and the next ``n_mics`` channels are MFCC.
    """
    cwt = _as_3d_repr(cwt_stack, name="cwt_stack")
    mfcc = _as_3d_repr(mfcc_stack, name="mfcc_stack")

    if cwt.shape[0] != mfcc.shape[0]:
        raise ValueError("cwt_stack and mfcc_stack must have the same n_mics")

    time_steps = (
        int(target_time_steps)
        if target_time_steps is not None
        else min(cwt.shape[2], mfcc.shape[2])
    )
    if time_steps <= 0:
        raise ValueError("target_time_steps must be > 0")

    freq_bins = (
        int(target_freq_bins) if target_freq_bins is not None else int(cwt.shape[1])
    )
    if freq_bins <= 0:
        raise ValueError("target_freq_bins must be > 0")

    cwt_aligned = _resample_axis(cwt, axis=2, new_size=time_steps)
    mfcc_aligned = _resample_axis(mfcc, axis=2, new_size=time_steps)
    cwt_aligned = _resample_axis(cwt_aligned, axis=1, new_size=freq_bins)
    mfcc_aligned = _resample_axis(mfcc_aligned, axis=1, new_size=freq_bins)

    return np.concatenate([cwt_aligned, mfcc_aligned], axis=0).astype(np.float64)


def _as_2d_channels(data: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (channels, samples)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_3d_repr(data: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (channels, bins, steps)")
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _resample_axis(data: np.ndarray, *, axis: int, new_size: int) -> np.ndarray:
    if new_size <= 0:
        raise ValueError("new_size must be > 0")
    if data.shape[axis] == new_size:
        return data.astype(np.float64)

    moved = np.moveaxis(data, axis, -1)
    old_size = moved.shape[-1]
    flat = moved.reshape(-1, old_size)

    src = np.linspace(0.0, 1.0, num=old_size)
    dst = np.linspace(0.0, 1.0, num=new_size)

    out = np.empty((flat.shape[0], new_size), dtype=np.float64)
    for i in range(flat.shape[0]):
        out[i] = np.interp(dst, src, flat[i])

    out = out.reshape(*moved.shape[:-1], new_size)
    return np.moveaxis(out, -1, axis)
