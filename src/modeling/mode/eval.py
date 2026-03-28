"""Evaluation helpers for mode classification training and cross-validation.

Provides:
- evaluate — full-pass loss, accuracy, predictions, and class probabilities.
- macro_f1_score — unweighted mean F1 across classes, the primary ranking metric.
- resolve_class_weights — inverse-frequency class weights for imbalanced data.
- build_cv_splits — StratifiedKFold or GroupKFold split construction.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "mode.eval requires PyTorch. Install with: pip install torch"
    ) from exc

try:
    from sklearn.model_selection import GroupKFold, StratifiedKFold
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "mode.eval requires scikit-learn. Install with: pip install scikit-learn"
    ) from exc

from .data import batch_iterator


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for x_t, y_t in batch_iterator(
            x,
            y,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            seed=0,
        ):
            logits = model(x_t)
            loss = F.cross_entropy(logits, y_t)
            losses.append(float(loss.detach().cpu().item()))
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            preds.append(pred.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    loss_mean = float(np.mean(losses)) if losses else float("nan")
    acc = float(np.mean(y_pred == y)) if y.size else float("nan")
    return loss_mean, acc, y_pred, y_prob


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    if y_true.size == 0:
        return float("nan")

    per_class_f1: list[float] = []
    for cls in range(n_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append((2.0 * precision * recall) / (precision + recall))

    return float(np.mean(per_class_f1)) if per_class_f1 else float("nan")


def resolve_class_weights(y_train: np.ndarray, *, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    if np.any(counts <= 0.0):
        raise ValueError("Class weights require every class to appear in training set")

    total = float(np.sum(counts))
    weights = total / (float(n_classes) * counts)
    return weights.astype(np.float32)


def build_cv_splits(
    y: np.ndarray,
    recording_ids: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, str | None]:
    if n_splits < 2:
        raise ValueError("cv_folds must be >= 2")

    groups = recording_ids.astype(str)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        folds = [
            (train_idx.astype(np.int64), val_idx.astype(np.int64))
            for train_idx, val_idx in splitter.split(y, y, groups=groups)
        ]

        n_classes = int(len(np.unique(y)))
        valid_group_folds = True
        for train_idx, _ in folds:
            if len(np.unique(y[train_idx])) < n_classes:
                valid_group_folds = False
                break

        if valid_group_folds:
            return folds, "group_kfold", None

    class_counts = np.bincount(y)
    max_folds_by_class = int(np.min(class_counts)) if class_counts.size else 0
    fallback_splits = min(int(n_splits), int(max_folds_by_class))
    if fallback_splits < 2:
        raise ValueError(
            "Not enough samples per class to build CV folds. "
            "Need at least 2 samples per class."
        )

    splitter = StratifiedKFold(
        n_splits=int(fallback_splits),
        shuffle=True,
        random_state=int(seed),
    )
    folds = [
        (train_idx.astype(np.int64), val_idx.astype(np.int64))
        for train_idx, val_idx in splitter.split(y, y)
    ]
    warning = (
        "GroupKFold could not provide class-complete training folds; falling back "
        "to window-level StratifiedKFold, which can overestimate performance."
    )
    return folds, "window_stratified", warning


__all__ = [
    "build_cv_splits",
    "evaluate",
    "macro_f1_score",
    "resolve_class_weights",
]
