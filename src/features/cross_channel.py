"""Cross-channel and cross-modal relational features for source localization.

Faults in a reverberant turbine housing radiate from a specific physical location.
Cross-sensor features capture the spatial imprint of that radiation: correlated
amplitude patterns, coherent phase relationships at the machine's characteristic
frequencies, and the relative acoustic energy balance between generator-level and
turbine-level sensor rings. These features are the primary input for the TDOA-based
source localization component of the thesis.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import coherence as sig_coherence

from ..data import DataSegment
from .feature_frame import FeatureFrame


class CrossChannelExtractor:
    """Inter-sensor relational features for mic and vibration channels.

    Emits:
    - Pearson correlation matrices (upper triangle) for mic channels and accel channels.
    - Generator-vs-turbine level energy ratio (spatial acoustic balance).
    - Spectral coherence at f0, 2×f0, 3×f0, and vpf for all mic-channel pairs.
    - Compound cross-modal scalars: acoustic_dominance, structural_indicator,
      flow_fault_indicator, subsurface_indicator.
    - Normalized per-channel energy distribution over all 13 sensors.

    Args:
        fundamental_freq_hz: Machine rotational fundamental, default 5.867 Hz.
        vane_pass_freq_hz: Blade-pass frequency, default 117.3 Hz.
    """

    def __init__(
        self, fundamental_freq_hz: float = 5.867, vane_pass_freq_hz: float = 117.3
    ) -> None:
        self._f0 = fundamental_freq_hz
        self._vpf = vane_pass_freq_hz

    def extract(self, segment: DataSegment) -> FeatureFrame:
        mic = segment.mic_data
        accel = segment.accel_data

        features: dict[str, float | np.ndarray] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mic_corr = np.nan_to_num(np.corrcoef(mic), nan=0.0)
            accel_corr = np.nan_to_num(np.corrcoef(accel), nan=0.0)

        features["mic_correlation_matrix"] = mic_corr[
            np.triu_indices(mic.shape[0], k=1)
        ].astype(np.float64)
        features["accel_correlation_matrix"] = accel_corr[
            np.triu_indices(accel.shape[0], k=1)
        ].astype(np.float64)

        # Generator level mics are channels 0-3, turbine level mics are channels 4-8.
        if mic.shape[0] >= 9:
            gen_energy = float(np.mean(np.sum(mic[0:4] ** 2, axis=1)))
            turb_energy = float(np.mean(np.sum(mic[4:9] ** 2, axis=1)))
        else:
            half = max(1, mic.shape[0] // 2)
            gen_energy = float(np.mean(np.sum(mic[:half] ** 2, axis=1)))
            turb_energy = (
                float(np.mean(np.sum(mic[half:] ** 2, axis=1)))
                if mic.shape[0] > half
                else 1e-12
            )

        features["generator_vs_turbine_energy"] = (
            gen_energy / turb_energy if turb_energy > 0 else 1.0
        )

        key_freqs = [self._f0, 2 * self._f0, 3 * self._f0, self._vpf]
        mic_coh_values: list[float] = []

        nperseg = min(
            mic.shape[1], max(256, int(np.ceil(segment.mic_sample_rate / self._f0)))
        )
        for i in range(mic.shape[0]):
            for j in range(i + 1, mic.shape[0]):
                freqs, coh = sig_coherence(
                    mic[i], mic[j], fs=segment.mic_sample_rate, nperseg=nperseg
                )
                coh = np.nan_to_num(coh, nan=0.0)
                for f in key_freqs:
                    idx = int(np.argmin(np.abs(freqs - f)))
                    mic_coh_values.append(float(coh[idx]))

        features["mic_coherence_mean"] = (
            float(np.mean(mic_coh_values)) if mic_coh_values else 0.0
        )
        features["mic_coherence_per_pair"] = np.asarray(
            mic_coh_values, dtype=np.float64
        )

        # Cross-modal summary indicators.
        mic_rms = float(np.sqrt(np.mean(mic**2)))
        accel_rms = float(np.sqrt(np.mean(accel**2)))
        total = mic_rms + accel_rms
        acoustic_dominance = mic_rms / total if total > 0 else 0.5

        mean_coh = float(np.mean(mic_coh_values)) if mic_coh_values else 0.0
        balance = 1.0 - abs(2.0 * acoustic_dominance - 1.0)

        features["acoustic_dominance"] = acoustic_dominance
        features["structural_indicator"] = balance * mean_coh
        features["flow_fault_indicator"] = acoustic_dominance * (1.0 - mean_coh)
        features["subsurface_indicator"] = (1.0 - acoustic_dominance) * (1.0 - mean_coh)

        all_energy = np.concatenate([np.sum(mic**2, axis=1), np.sum(accel**2, axis=1)])
        denom = float(np.sum(all_energy))
        features["channel_energy_distribution"] = (
            (all_energy / denom).astype(np.float64)
            if denom > 0
            else np.zeros_like(all_energy, dtype=np.float64)
        )

        return FeatureFrame(
            features=features,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            segment_metadata=dict(segment.metadata),
        )
