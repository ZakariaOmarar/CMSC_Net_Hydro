"""Frequency-domain feature extraction via windowed FFT.

The Francis pump-turbine at ROW II is characterized by a rotational fundamental at
approximately 5.867 Hz (375 rpm / 60, adjusted for loading) and a vane-pass
frequency at approximately 117.3 Hz. These and their harmonics are the primary
tonal lines that both distinguish operating modes and indicate incipient faults.
"""

from __future__ import annotations

import numpy as np

from ..data import DataSegment
from .feature_frame import FeatureFrame

_DEFAULT_BANDS = [
    (0.0, 50.0),
    (50.0, 200.0),
    (200.0, 1_000.0),
    (1_000.0, 5_000.0),
]
_BAND_LABELS = ["0_50", "50_200", "200_1k", "1k_5k"]


class FrequencyDomainExtractor:
    """Spectral shape descriptors and turbine-specific line amplitudes per channel.

    Applies a Hanning window and computes the rfft magnitude spectrum (padded to
    at least n_fft or one full cycle of f0 to resolve the fundamental). Emits:

    - Spectral shape: centroid, 85%-energy rolloff, bandwidth, spectral flatness.
    - Machine lines: fundamental (f0) and vane-pass (vpf) amplitudes, plus 2×, 3×,
      4× harmonic ratios relative to f0.
    - Band energies in four fixed bands from DC to 5 kHz.

    The harmonic structure changes markedly across Turbine, Pump, and Standstill
    modes and is the most discriminative frequency-domain signal for both mode
    detection and anomaly scoring.

    Args:
        fundamental_freq_hz: Rotational fundamental frequency, default 5.867 Hz.
        vane_pass_freq_hz: Blade-pass frequency, default 117.3 Hz.
        n_fft: Minimum FFT length; automatically extended to resolve f0 if needed.
        channels: Channel group to process: ``"all"``, ``"mic"``, or ``"accel"``.
    """

    def __init__(
        self,
        fundamental_freq_hz: float = 5.867,
        vane_pass_freq_hz: float = 117.3,
        n_fft: int = 4096,
        channels: str = "all",
    ) -> None:
        if channels not in {"all", "mic", "accel"}:
            raise ValueError("channels must be one of: all, mic, accel")

        self._f0 = fundamental_freq_hz
        self._vpf = vane_pass_freq_hz
        self._n_fft = n_fft
        self._channels = channels

    def extract(self, segment: DataSegment) -> FeatureFrame:
        features: dict[str, float | np.ndarray] = {}

        if self._channels in {"all", "mic"}:
            self._extract(
                features,
                segment.mic_data,
                segment.mic_channel_names,
                segment.mic_sample_rate,
            )

        if self._channels in {"all", "accel"}:
            self._extract(
                features,
                segment.accel_data,
                segment.accel_channel_names,
                segment.accel_sample_rate,
            )

        return FeatureFrame(
            features=features,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            segment_metadata=dict(segment.metadata),
        )

    def _extract(
        self,
        out: dict[str, float | np.ndarray],
        data: np.ndarray,
        names: tuple[str, ...],
        sample_rate: int,
    ) -> None:
        for i, name in enumerate(names):
            x = data[i]
            n = len(x)
            fft_len = max(self._n_fft, n, int(np.ceil(sample_rate / self._f0)))

            window = np.hanning(n)
            xw = x * window
            spec = np.fft.rfft(xw, n=fft_len)
            freqs = np.fft.rfftfreq(fft_len, d=1.0 / sample_rate)
            mag = (
                np.abs(spec) * (2.0 / np.sum(window))
                if np.sum(window) > 0
                else np.abs(spec)
            )
            power = mag**2

            total_power = float(np.sum(power))
            if total_power == 0.0:
                out[f"{name}_spectral_centroid"] = 0.0
                out[f"{name}_spectral_rolloff"] = 0.0
                out[f"{name}_spectral_bandwidth"] = 0.0
                out[f"{name}_spectral_flatness"] = 0.0
                out[f"{name}_fundamental_amplitude"] = 0.0
                out[f"{name}_vpf_amplitude"] = 0.0
                for h in (2, 3, 4):
                    out[f"{name}_harmonic_ratio_{h}x"] = 0.0
                for label in _BAND_LABELS:
                    out[f"{name}_band_energy_{label}"] = 0.0
                continue

            centroid = float(np.sum(freqs * power) / total_power)
            cumulative = np.cumsum(power)
            ridx = int(np.searchsorted(cumulative, 0.85 * total_power))
            ridx = min(ridx, len(freqs) - 1)
            rolloff = float(freqs[ridx])
            bandwidth = float(
                np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)
            )

            ppos = power[power > 0]
            flatness = (
                float(np.exp(np.mean(np.log(ppos))) / np.mean(ppos))
                if len(ppos)
                else 0.0
            )

            f0_amp = float(mag[_nearest_bin(freqs, self._f0)])
            vpf_amp = float(mag[_nearest_bin(freqs, self._vpf)])

            out[f"{name}_spectral_centroid"] = centroid
            out[f"{name}_spectral_rolloff"] = rolloff
            out[f"{name}_spectral_bandwidth"] = bandwidth
            out[f"{name}_spectral_flatness"] = flatness
            out[f"{name}_fundamental_amplitude"] = f0_amp
            out[f"{name}_vpf_amplitude"] = vpf_amp

            for h in (2, 3, 4):
                h_amp = float(mag[_nearest_bin(freqs, h * self._f0)])
                out[f"{name}_harmonic_ratio_{h}x"] = (
                    h_amp / f0_amp if f0_amp > 0 else 0.0
                )

            for label, (lo, hi) in zip(_BAND_LABELS, _DEFAULT_BANDS):
                mask = (freqs >= lo) & (freqs < hi)
                out[f"{name}_band_energy_{label}"] = float(np.sum(power[mask]))


def _nearest_bin(freqs: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(freqs - target)))
