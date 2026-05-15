"""Acoustic encoder-input features: log-mel + CWT scalograms stacked per mic.

Produces a 4-D tensor with shape ``(n_mics, 2, F, T_frames)`` where channel 0 is
log-mel and channel 1 is the CWT scalogram resampled to the same time/frequency
grid. This is the direct input format for the V1 acoustic 2-D CNN encoder.

Distinct from `acoustic_representations.py`, which produces individual features
for downstream feature-frame baselines.
"""

from __future__ import annotations

import numpy as np

from .acoustic_representations import compute_cwt_scalogram


def compute_log_mel_spectrogram(
    signal: np.ndarray,
    fs: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Log-mel spectrogram for one channel.

    Returns a 2-D array of shape ``(n_mels, n_frames)``, in log1p of the mel
    power. Uses librosa under the hood.
    """
    import librosa

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0")

    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if x.size == 0:
        raise ValueError("signal cannot be empty")

    nyquist = fs / 2.0
    if fmax is None:
        fmax = nyquist
    if fmax > nyquist:
        fmax = nyquist

    mel_power = librosa.feature.melspectrogram(
        y=x,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return np.log1p(mel_power).astype(np.float64)


def compute_encoder_input_stack(
    mic_data: np.ndarray,
    fs: int,
    *,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    cwt_n_scales: int = 64,
    cwt_min_freq_hz: float = 20.0,
    standardize: bool = False,
) -> np.ndarray:
    """Build the V1 acoustic encoder input.

    Args:
        mic_data: Shape ``(n_mics, n_samples)``.
        fs: Microphone sample rate.
        n_mels / n_fft / hop_length: log-mel parameters.
        cwt_n_scales / cwt_min_freq_hz: CWT scalogram parameters.
        standardize: When True, each of the two representations (log-mel =
            channel 0, CWT = channel 1) is per-recording z-scored across all
            mics, frequencies and frames so both feed the ``Conv2d`` on a
            unit-variance scale.  **Default False** — this is the F4 audit
            experiment (2026-05-14); it was found not to be load-bearing
            (the V1 acoustic encoder trains fine without it once BatchNorm
            is used) and is left off as the known-good behaviour.  The knob
            is retained because the channel-scale-mismatch concern is real
            and may matter once the SSL objective is revisited.

    Returns:
        ``(n_mics, 2, F, T_frames)`` float32 array. Channel 0 is log-mel,
        channel 1 is CWT (resampled along time and frequency to match log-mel
        with simple linear interpolation, so the CNN sees both representations
        on a shared grid).  With ``standardize=True`` each channel has zero
        mean and unit variance over the (mic, frequency, time) axes.
    """
    if mic_data.ndim != 2:
        raise ValueError("mic_data must be 2-D (n_mics, n_samples)")
    n_mics = int(mic_data.shape[0])

    log_mels: list[np.ndarray] = []
    cwts: list[np.ndarray] = []
    for i in range(n_mics):
        log_mels.append(
            compute_log_mel_spectrogram(
                mic_data[i],
                fs,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )
        cwts.append(
            compute_cwt_scalogram(
                mic_data[i],
                fs,
                n_scales=cwt_n_scales,
                min_freq_hz=cwt_min_freq_hz,
            )
        )

    target_F = log_mels[0].shape[0]
    target_T = log_mels[0].shape[1]

    aligned: list[np.ndarray] = []
    for mel, cwt in zip(log_mels, cwts):
        cwt_aligned = _bilinear_resize(cwt, (target_F, target_T))
        aligned.append(np.stack([mel, cwt_aligned], axis=0))

    stack = np.stack(aligned, axis=0).astype(np.float32)

    if standardize:
        # Per-channel (log-mel vs CWT) per-recording z-score.  Statistics are
        # taken across mics + frequency + time — preserving relative dynamics
        # WITHIN each channel while equalising the BETWEEN-channel scale that
        # would otherwise let the first ``Conv2d`` learn to ignore the
        # smaller-magnitude channel.
        eps = 1e-8
        for ch in (0, 1):
            mean = float(stack[:, ch, :, :].mean())
            std = float(stack[:, ch, :, :].std())
            stack[:, ch, :, :] = (stack[:, ch, :, :] - mean) / max(std, eps)

    return stack


def _bilinear_resize(arr: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to ``new_shape`` via separable linear interpolation."""
    target_F, target_T = new_shape
    cur_F, cur_T = arr.shape
    if cur_F == target_F and cur_T == target_T:
        return arr.astype(np.float64)

    # Interpolate along time first.
    t_src = np.linspace(0.0, 1.0, cur_T)
    t_dst = np.linspace(0.0, 1.0, target_T)
    along_T = np.empty((cur_F, target_T), dtype=np.float64)
    for r in range(cur_F):
        along_T[r] = np.interp(t_dst, t_src, arr[r])

    if cur_F == target_F:
        return along_T

    f_src = np.linspace(0.0, 1.0, cur_F)
    f_dst = np.linspace(0.0, 1.0, target_F)
    out = np.empty((target_F, target_T), dtype=np.float64)
    for c in range(target_T):
        out[:, c] = np.interp(f_dst, f_src, along_T[:, c])
    return out


__all__ = ["compute_log_mel_spectrogram", "compute_encoder_input_stack"]
