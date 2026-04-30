"""Layer 3 — LightGBM 12-class mode/transition classifier.

Classes
-------
0  ST
1  TU
2  PU
3  PH
4  ST→PU
5  ST→TU
6  ST→PH
7  PH→TU
8  PH→PU
9  PU→ST
10 TU→ST
11 PU→PH  (or TU→PH — merged into one class if data is scarce)

Training set
------------
- Steady states: timesteps where oracle_confidence == HIGH
- Transitions: all signature-matched transition segments from Layer 2
- Class imbalance handled via class_weight and downsampling of steady states

Holdout
-------
Leave-one-day-out cross-validation across the 7-day campaign.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    warnings.warn("lightgbm not installed. Install via: pip install lightgbm")

from src.modeling.mode.p1_physics.oracle import CONFIDENCE_CODE, LABEL_CODE, CODE_LABEL
from src.modeling.mode.p2_transitions.signatures import VALID_TRANSITIONS

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

STEADY_CLASS_NAMES = ["ST", "TU", "PU", "PH"]
TRANSITION_CLASS_NAMES = [
    "ST→PU", "ST→TU", "ST→PH",
    "PH→TU", "PH→PU",
    "PU→ST", "TU→ST",
    "PU→PH", "TU→PH",
]
ALL_CLASS_NAMES = STEADY_CLASS_NAMES + TRANSITION_CLASS_NAMES
N_CLASSES = len(ALL_CLASS_NAMES)  # 4 + 9 = 13 (or fewer if some never appear)

CLASS_TO_IDX = {name: i for i, name in enumerate(ALL_CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Training set assembly
# ---------------------------------------------------------------------------


def build_training_set(
    X_feat: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_confidence: np.ndarray,
    transition_segments: list,  # list[TransitionSegment] from p2_transitions
    timestamps_ns: np.ndarray | None = None,
    *,
    max_steady_ratio: int = 5,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build (X_train, y_train[, day_index]) from oracle + transition segments.

    Parameters
    ----------
    X_feat:
        (T, D) float32 feature matrix.
    oracle_labels:
        (T,) int8 from OracleResult.
    oracle_confidence:
        (T,) int8 from OracleResult.
    transition_segments:
        List of TransitionSegment from Layer 2.
    timestamps_ns:
        (T,) int64 — used to build day index for LOO-CV. Optional.
    max_steady_ratio:
        Maximum ratio of steady-state samples to the smallest transition class.
        Prevents steady states overwhelming rare transitions.

    Returns
    -------
    X_train, y_train, day_index (or None if timestamps not provided)
    """
    high_code = CONFIDENCE_CODE["HIGH"]

    # --- Steady-state samples (oracle-HIGH only) --------------------------
    steady_idx_list = []
    steady_y_list = []
    for label_str in STEADY_CLASS_NAMES:
        code = LABEL_CODE[label_str]
        mask = (oracle_labels == code) & (oracle_confidence == high_code)
        idx = np.where(mask)[0]
        steady_idx_list.append(idx)
        steady_y_list.append(np.full(len(idx), CLASS_TO_IDX[label_str], dtype=np.int32))

    # --- Transition samples (signature-matched) ---------------------------
    trans_idx_list = []
    trans_y_list = []
    for seg in transition_segments:
        cls_name = seg.transition_type
        if cls_name not in CLASS_TO_IDX:
            continue  # MICRO_EVENT, INVALID_TOPOLOGY, UNKNOWN — skip
        if not seg.signature_match:
            continue  # unverified — skip
        idx = np.arange(seg.t_start_s, min(seg.t_end_s + 1, len(X_feat)))
        trans_idx_list.append(idx)
        trans_y_list.append(np.full(len(idx), CLASS_TO_IDX[cls_name], dtype=np.int32))

    # --- Downsample steady states -----------------------------------------
    all_trans_idx = np.concatenate(trans_idx_list) if trans_idx_list else np.array([], dtype=int)
    n_trans_total = max(len(all_trans_idx), 1)
    # Find smallest transition class size
    trans_class_counts = {}
    for arr, y in zip(trans_idx_list, trans_y_list):
        cls_id = int(y[0]) if len(y) else -1
        trans_class_counts[cls_id] = trans_class_counts.get(cls_id, 0) + len(arr)
    min_trans = min(trans_class_counts.values()) if trans_class_counts else 1
    max_per_steady = max(min_trans * max_steady_ratio, 1000)

    rng = np.random.default_rng(random_seed)
    steady_idx_ds = []
    steady_y_ds = []
    for idx, y in zip(steady_idx_list, steady_y_list):
        if len(idx) > max_per_steady:
            chosen = rng.choice(len(idx), max_per_steady, replace=False)
            idx = idx[chosen]
            y = y[chosen]
        steady_idx_ds.append(idx)
        steady_y_ds.append(y)

    # --- Combine ----------------------------------------------------------
    all_idx = np.concatenate(steady_idx_ds + trans_idx_list + ([] if not trans_idx_list else []))
    all_y   = np.concatenate(steady_y_ds + trans_y_list + ([] if not trans_y_list else []))

    if len(all_idx) == 0:
        raise ValueError("No training samples assembled. Check oracle output and transition segments.")

    X_train = X_feat[all_idx]
    y_train = all_y

    # Day index for LOO-CV
    day_idx = None
    if timestamps_ns is not None:
        seconds = timestamps_ns[all_idx] // 1_000_000_000
        days_epoch = seconds // 86_400
        day_idx = (days_epoch - days_epoch.min()).astype(np.int32)

    return X_train, y_train, day_idx


# ---------------------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------------------


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_leaves: int = 63,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1,
    verbose: int = -1,
) -> "lgb.LGBMClassifier":
    """Train a LightGBM multi-class classifier."""
    if not _HAS_LGB:
        raise ImportError("lightgbm is required for Layer 3. Install: pip install lightgbm")

    # Class weights inversely proportional to class frequency
    classes, counts = np.unique(y_train, return_counts=True)
    freq = counts.astype(np.float64)
    weight_map = {int(c): float(1.0 / f) for c, f in zip(classes, freq)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train])

    clf = lgb.LGBMClassifier(
        num_leaves=n_leaves,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf


def loo_cv(
    X_feat: np.ndarray,
    y: np.ndarray,
    day_idx: np.ndarray,
    *,
    lgbm_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Leave-one-day-out cross-validation.

    Returns a dict with per-day accuracy and an overall confusion matrix.
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm required")

    lgbm_kwargs = lgbm_kwargs or {}
    unique_days = np.unique(day_idx)
    results = []
    all_preds = np.empty_like(y)

    for hold_day in unique_days:
        train_mask = day_idx != hold_day
        test_mask  = day_idx == hold_day

        if train_mask.sum() < 100 or test_mask.sum() < 10:
            continue

        clf = train_lgbm(X_feat[train_mask], y[train_mask], **lgbm_kwargs)
        preds = clf.predict(X_feat[test_mask])
        all_preds[test_mask] = preds

        acc = float(np.mean(preds == y[test_mask]))
        results.append({"day": int(hold_day), "n_test": int(test_mask.sum()), "accuracy": acc})

    # Overall confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, all_preds, labels=list(range(N_CLASSES)))

    return {
        "per_day": results,
        "overall_accuracy": float(np.mean(all_preds == y)),
        "confusion_matrix": cm.tolist(),
        "class_names": ALL_CLASS_NAMES,
    }


def save_lgbm(clf: "lgb.LGBMClassifier", path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(path))


def load_lgbm(path: str | Path) -> "lgb.LGBMClassifier":
    if not _HAS_LGB:
        raise ImportError("lightgbm required")
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(path))
    clf = lgb.LGBMClassifier()
    clf._Booster = booster
    return clf


def predict_proba(clf: "lgb.LGBMClassifier", X_feat: np.ndarray) -> np.ndarray:
    """Return (T, N_CLASSES) posterior probabilities."""
    return clf.predict_proba(X_feat).astype(np.float32)
