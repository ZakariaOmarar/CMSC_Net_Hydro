"""Shared metrics helpers for baseline anomaly train/infer flows."""

from __future__ import annotations

import numpy as np


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = y_true.astype(bool)
    y_p = y_pred.astype(bool)

    tp = int(np.sum(y_t & y_p))
    if tp == 0:
        # Both precision and recall collapse to 0; F1 is 0 by convention.
        return 0.0

    # With tp > 0: (tp + fp) >= 1 and (tp + fn) >= 1, so no ZeroDivisionError.
    fp = int(np.sum(~y_t & y_p))
    fn = int(np.sum(y_t & ~y_p))

    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)

    return 2.0 * precision * recall / (precision + recall)


def compute_binary_metrics(
    *,
    scores: np.ndarray,
    labels_anomaly: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = labels_anomaly.astype(bool)
    y_pred = scores > float(threshold)

    accuracy    = float(np.mean(y_true == y_pred))
    f1_anomaly  = binary_f1(y_true, y_pred)
    f1_normal   = binary_f1(~y_true, ~y_pred)
    macro_f1    = (f1_anomaly + f1_normal) / 2.0

    return {
        "accuracy":   accuracy,
        "macro_f1":   macro_f1,
        "f1_anomaly": f1_anomaly,
        "f1_normal":  f1_normal,
    }


__all__ = ["binary_f1", "compute_binary_metrics"]