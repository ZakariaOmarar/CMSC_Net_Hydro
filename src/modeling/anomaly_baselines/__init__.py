"""Standalone V0 baselines for the test_dataset pipeline.

Distinct from `src.modeling.baselines`, which operates on pre-computed (z, c)
latents from the existing CNF pipeline.  V0 baselines here read raw audio from
the `TestDatasetLoader` and compute features on the fly, so each V0 number
compares apples-to-apples against subsequent iterations.

Two baselines per the plan's "one per RQ" rule:
  - `lstm_ae`     — V0 anomaly baseline on log-mel windows (RQ2 reference).
  - `mode_lgbm`   — V0 supervised mode classifier on hand-engineered features
                    (RQ1 upper-bound reference; the only place mode labels are
                    legitimately used as a training target).

The V0 SRP-PHAT localization baseline (RQ3 reference) wraps the existing
`src/modeling/localization/localization_head.py` and lives there.
"""

from .lstm_ae import (
    LSTMAutoencoderV0,
    V0Config,
    extract_log_mel_windows,
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

__all__ = [
    "LSTMAutoencoderV0",
    "ModeTrainResult",
    "V0Config",
    "V0ModeConfig",
    "extract_log_mel_windows",
    "extract_mode_features",
    "predict_modes",
    "score_recordings",
    "train_v0_lstm_ae",
    "train_v0_mode_lgbm",
]
