"""Standalone V0 baselines for the test_dataset pipeline.

These baselines read raw audio from the `TestDatasetLoader` and compute
features on the fly, so each V0 number compares apples-to-apples against the
subsequent iterations.

Two baselines, one per research question:
  - `lstm_ae`     — V0 anomaly baseline on log-mel windows (RQ2 reference).
  - `mode_lgbm`   — V0 supervised mode classifier on hand-engineered features
                    (RQ1 upper-bound reference; the only place mode labels are
                    legitimately used as a training target).

The V0 SRP-PHAT localization baseline (RQ3 reference) wraps the classical
primitives in `src/modeling/localization/classical.py` and lives there.
"""

from .lstm_ae import (
    LSTMAutoencoderV0,
    V0Config,
    extract_log_mel_windows,
    extract_vibration_temporal_windows,
    score_recordings,
    train_v0_lstm_ae,
)
from .mode_lgbm import (
    ModeTrainResult,
    V0ModeConfig,
    extract_mode_features,
    predict_modes,
    train_v0_mode_lgbm,
)
from .srp_phat_baseline import (
    SRPConfig,
    evaluate_srp_phat,
    predict_srp_phat,
    summarise,
)

__all__ = [
    "LSTMAutoencoderV0",
    "ModeTrainResult",
    "SRPConfig",
    "V0Config",
    "V0ModeConfig",
    "evaluate_srp_phat",
    "extract_log_mel_windows",
    "extract_mode_features",
    "extract_vibration_temporal_windows",
    "predict_modes",
    "predict_srp_phat",
    "score_recordings",
    "summarise",
    "train_v0_lstm_ae",
    "train_v0_mode_lgbm",
]
