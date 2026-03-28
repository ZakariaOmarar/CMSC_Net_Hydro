"""Low-rate vibration envelope features for pre-computed amplitude streams."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from ..data import DataSegment
from .feature_frame import FeatureFrame


class VibrationEnvelopeExtractor:
    """Rich per-sensor feature vector combining time, spectral, and transition statistics.

    For each accelerometer channel the extractor produces a concatenated vector:
    - 11 time-domain statistics (RMS through linear slope).
    - 3 spectral statistics (total energy, centroid, sub-synchronous energy ratio).
    - 4 transition statistics (gradient magnitude, monotonicity flags, rolling
      variance delta) — only populated for windows flagged as mode transitions,
      otherwise zero-padded.

    The transition part helps downstream models learn the distinct mechanical
    transient that occurs when the machine reverses between Pump and Turbine
    directions. The transition flag is read from segment.metadata or inferred
    from a boolean transition_mask array indexed by window_index.

    Args:
        sub_sync_cutoff_hz: Frequency boundary for sub-synchronous energy ratio.
        transition_mask_key: Metadata key for the boolean per-window mask array.
        transition_flag_key: Metadata key for a direct scalar transition flag.
    """

    def __init__(
        self,
        *,
        sub_sync_cutoff_hz: float = 1.0,
        transition_mask_key: str = "transition_mask",
        transition_flag_key: str = "is_transition_window",
    ) -> None:
        if sub_sync_cutoff_hz <= 0.0:
            raise ValueError("sub_sync_cutoff_hz must be > 0")
        self._sub_sync_cutoff_hz = float(sub_sync_cutoff_hz)
        self._transition_mask_key = transition_mask_key
        self._transition_flag_key = transition_flag_key

    def extract(self, segment: DataSegment) -> FeatureFrame:
        features: dict[str, float | np.ndarray] = {}
        is_transition = self._resolve_transition_flag(segment.metadata)

        for i, name in enumerate(segment.accel_channel_names):
            x = segment.accel_data[i].astype(np.float64)
            time_part = _vibration_time_features(x)
            spectral_part = _vibration_spectral_features(
                x,
                fs_vib_effective=float(segment.accel_sample_rate),
                sub_sync_cutoff_hz=self._sub_sync_cutoff_hz,
            )

            if is_transition:
                t_vib = np.arange(x.shape[0], dtype=np.float64) / float(
                    segment.accel_sample_rate
                )
                transition_part = _vibration_transition_features(x, t_vib)
            else:
                transition_part = np.zeros(4, dtype=np.float64)

            features[f"{name}_vibration_features"] = np.concatenate(
                [time_part, spectral_part, transition_part],
                axis=0,
            ).astype(np.float64)

        return FeatureFrame(
            features=features,
            start_time=segment.start_time,
            duration_s=segment.duration_s,
            segment_metadata=dict(segment.metadata),
        )

    def _resolve_transition_flag(self, metadata: dict) -> bool:
        explicit = metadata.get(self._transition_flag_key)
        if explicit is not None:
            return bool(explicit)

        mask = metadata.get(self._transition_mask_key)
        window_index = metadata.get("window_index")
        if mask is None or window_index is None:
            return False

        try:
            idx = int(window_index)
            if idx < 0:
                return False
            return bool(mask[idx])
        except Exception:
            return False


def _vibration_time_features(amplitude_window: np.ndarray) -> np.ndarray:
    eps = 1e-8
    x = np.asarray(amplitude_window, dtype=np.float64)

    rms = float(np.sqrt(np.mean(x**2)))
    mean_abs = float(np.mean(np.abs(x)))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.max(x) - np.min(x))
    crest_factor = float(peak / (rms + eps))

    k = float(kurtosis(x, fisher=True, bias=True))
    s = float(skew(x, bias=True))
    if np.isnan(k):
        k = 0.0
    if np.isnan(s):
        s = 0.0

    std = float(np.std(x))

    if x.shape[0] < 2:
        mean_roc = 0.0
        max_roc = 0.0
        slope = 0.0
    else:
        roc_abs = np.abs(np.diff(x))
        mean_roc = float(np.mean(roc_abs))
        max_roc = float(np.max(roc_abs))
        try:
            slope = float(np.polyfit(np.arange(x.shape[0], dtype=np.float64), x, 1)[0])
        except Exception:
            slope = 0.0

    return np.array(
        [
            rms,
            mean_abs,
            peak,
            peak_to_peak,
            crest_factor,
            k,
            s,
            std,
            mean_roc,
            max_roc,
            slope,
        ],
        dtype=np.float64,
    )


def _vibration_spectral_features(
    amplitude_window: np.ndarray,
    *,
    fs_vib_effective: float,
    sub_sync_cutoff_hz: float,
) -> np.ndarray:
    eps = 1e-8
    x = np.asarray(amplitude_window, dtype=np.float64)

    if x.shape[0] == 0 or fs_vib_effective <= 0.0:
        return np.zeros(3, dtype=np.float64)

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs_vib_effective)

    total_energy = float(np.sum(spectrum**2))
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + eps))
    sub_sync_energy = float(
        np.sum(spectrum[freqs < sub_sync_cutoff_hz] ** 2) / (total_energy + eps)
    )

    return np.array([total_energy, centroid, sub_sync_energy], dtype=np.float64)


def _vibration_transition_features(
    amplitude_window: np.ndarray,
    t_vib: np.ndarray,
) -> np.ndarray:
    x = np.asarray(amplitude_window, dtype=np.float64)
    t = np.asarray(t_vib, dtype=np.float64)

    if x.shape[0] < 2 or t.shape[0] != x.shape[0]:
        return np.zeros(4, dtype=np.float64)

    try:
        roc = np.gradient(x, t)
        max_abs_roc = float(np.max(np.abs(roc)))
    except Exception:
        max_abs_roc = 0.0

    delta = np.diff(x)
    is_up = float(int(np.all(delta > 0.0)))
    is_down = float(int(np.all(delta < 0.0)))

    mid = x.shape[0] // 2
    if mid == 0 or mid == x.shape[0]:
        rolling_var = 0.0
    else:
        rolling_var = float(np.var(x[:mid]) - np.var(x[mid:]))

    return np.array([max_abs_roc, is_up, is_down, rolling_var], dtype=np.float64)
