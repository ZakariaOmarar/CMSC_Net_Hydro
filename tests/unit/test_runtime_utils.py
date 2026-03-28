from __future__ import annotations

import numpy as np

from src.modeling.core.runtime_utils import (
    apply_standardizer,
    fit_standardizer,
    majority_smooth_labels,
    run_lengths,
)


def test_majority_smooth_labels_and_run_lengths() -> None:
    labels = np.asarray(["Pump", "Turbine", "Turbine", "Pump", "Pump"], dtype=str)

    smoothed = majority_smooth_labels(labels, window=3)
    lengths = run_lengths(smoothed)

    assert smoothed.tolist() == ["Pump", "Pump", "Turbine", "Turbine", "Pump"]
    assert lengths.tolist() == [1, 2, 1, 2, 1]


def test_standardizer_helpers_floor_small_std() -> None:
    x = np.asarray(
        [
            [1.0, 5.0, 10.0],
            [2.0, 5.0, 20.0],
            [3.0, 5.0, 30.0],
        ],
        dtype=np.float32,
    )

    mean, std = fit_standardizer(x)
    x_norm = apply_standardizer(x, mean=mean, std=std)

    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert float(std[1]) == 1.0
    assert np.all(np.isfinite(x_norm))
    assert x_norm.shape == x.shape
