"""Layer 5 — TU sub-mode discovery based on load level.

Clusters within-TU operation into load bands using the Allg_M1 active power
signal (or the LightGBM logits as a proxy when Allg_M1 is absent).

Sub-modes
---------
TU-deep-part-load   < 20% rated power   (rope vortex regime)
TU-part-load        20–60% rated power
TU-near-BEP         60–90% rated power  (best efficiency point region)
TU-full-load        > 90% rated power

Boundaries are derived empirically from the P_Ist distribution within confirmed
TU operation; the percentiles above are starting points to be refined from data.
"""

from __future__ import annotations

import numpy as np

from src.modeling.mode.p1_physics.thresholds import extract_signal
from src.modeling.mode.p4_smoother.topology import STATE_TO_IDX

TU_IDX = STATE_TO_IDX["TU"]

# Default load band boundaries as fractions of rated power
# Adjust from the empirical P_Ist distribution within TU operation
DEFAULT_BANDS = [0.0, 0.20, 0.60, 0.90, 1.01]
SUB_MODE_NAMES = [
    "TU-deep-part-load",
    "TU-part-load",
    "TU-near-BEP",
    "TU-full-load",
]


def discover_tu_sub_modes(
    smoothed_labels: np.ndarray,
    allg: np.ndarray,
    channel_names: list[str],
    *,
    band_fractions: list[float] = DEFAULT_BANDS,
) -> np.ndarray:
    """Assign each TU timestep to a load-band sub-mode.

    Parameters
    ----------
    smoothed_labels : (T,) int32 — Layer 4 state sequence
    allg : (T, N_ch) float32 — Allg_M1 data
    channel_names : channel names

    Returns
    -------
    sub_mode_labels : (T,) int8
      -1 = not TU
       0 = TU-deep-part-load
       1 = TU-part-load
       2 = TU-near-BEP
       3 = TU-full-load
    """
    T = len(smoothed_labels)
    sub = np.full(T, -1, dtype=np.int8)

    tu_mask = smoothed_labels == TU_IDX
    if not np.any(tu_mask):
        return sub

    power = extract_signal(allg, channel_names, "power")
    if power is None:
        return sub

    # Derive rated power from 95th percentile of TU active power
    tu_power = power[tu_mask]
    rated_power = float(np.percentile(tu_power[np.isfinite(tu_power)], 95))
    if rated_power < 1.0:
        return sub

    p_norm = power / rated_power  # normalised to [0, 1] approximately

    for i, (lo, hi) in enumerate(zip(band_fractions[:-1], band_fractions[1:])):
        band_mask = tu_mask & (p_norm >= lo) & (p_norm < hi)
        sub[band_mask] = np.int8(i)

    return sub


def sub_mode_name(sub_id: int) -> str:
    if sub_id < 0 or sub_id >= len(SUB_MODE_NAMES):
        return "non-TU"
    return SUB_MODE_NAMES[sub_id]
