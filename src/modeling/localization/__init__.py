# pyright: reportMissingImports=false
"""Localization sub-package.

`classical` ships the classical signal-processing primitives (GCC-PHAT,
SRP-PHAT).  `v4_features`, `v4_loc_head`, and `v4_trainer` add the V4
anomaly-gated learning-based localization head (the live pipeline).
`localization_head` holds the earlier neural S2/S3/dual-SRP localizers, kept as
Chapter 6 baselines and built on the `classical` primitives; it is not imported
by the live pipeline.
"""

from .v4_features import (
    C_AIR_MS,
    C_PLASTIC_3DP_MS,
    V4_CANDIDATE_GRID,
    GridSpec,
    compute_accel_tdoa_tokens,
    compute_burst_aware_srp_phat_volume,
    compute_srp_phat_volume,
    find_burst_window,
)
from .v4_loc_head import (
    FiLMResidualHead,
    HeatmapCross3D,
    TDOASetEncoder,
    V4LocalizationHead,
    soft_argmax_3d,
)
from .v4_temporal import (
    AlertBurst,
    detect_alert_bursts,
    evaluate_burst_localization,
    smooth_predictions_over_bursts,
)
from .v4_trainer import (
    V4Config,
    V4Result,
    V4Sample,
    precompute_v4_samples,
    split_samples_by_dataset,
    split_samples_by_position,
    train_v4_localization,
)

__all__ = [
    "C_AIR_MS",
    "C_PLASTIC_3DP_MS",
    "V4_CANDIDATE_GRID",
    "AlertBurst",
    "FiLMResidualHead",
    "GridSpec",
    "HeatmapCross3D",
    "TDOASetEncoder",
    "V4Config",
    "V4LocalizationHead",
    "V4Result",
    "V4Sample",
    "compute_accel_tdoa_tokens",
    "compute_burst_aware_srp_phat_volume",
    "compute_srp_phat_volume",
    "detect_alert_bursts",
    "evaluate_burst_localization",
    "find_burst_window",
    "precompute_v4_samples",
    "smooth_predictions_over_bursts",
    "soft_argmax_3d",
    "split_samples_by_dataset",
    "split_samples_by_position",
    "train_v4_localization",
]
