from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.core.runtime_utils import compute_window_step_s
from src.modeling.flow.infer import (
    FlowInferenceResult,
    apply_mode_aware_fault_logic,
    build_anomaly_events,
)


def test_anomaly_events_include_window_timestamps() -> None:
    result = FlowInferenceResult(
        scores=np.asarray([0.1, 2.1, 0.2, 3.2], dtype=np.float32),
        flags=np.asarray([False, True, False, True], dtype=bool),
        thresholds=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    step_s = compute_window_step_s(window_s=5.0, overlap=0.5)
    events = build_anomaly_events(
        scores=result.scores,
        flags=result.flags,
        thresholds=result.thresholds,
        window_s=5.0,
        overlap=0.5,
    )

    assert step_s == 2.5
    assert len(events) == 2
    assert events[0]["window_index"] == 1
    assert events[0]["timestamp_s"] == 2.5
    assert events[1]["window_index"] == 3
    assert events[1]["timestamp_s"] == 7.5


def test_mode_aware_gating_requires_stable_mode_and_healthy_deviation() -> None:
    scores = np.asarray([1.0, 2.1, 8.0, 9.0, 10.0], dtype=np.float32)
    base_flags = np.asarray([False, True, True, True, True], dtype=bool)
    base_thresholds = np.asarray([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    mode_labels = np.asarray(["Pump", "Pump", "Pump", "Pump", "Pump"], dtype=str)
    mode_probs = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.80, 0.10, 0.10],
            [0.92, 0.05, 0.03],
            [0.88, 0.07, 0.05],
        ],
        dtype=np.float32,
    )
    healthy_stats = {"Pump": (2.0, 1.0)}

    (
        mode_flags,
        mode_z,
        mode_ref_thresholds,
        mode_conf,
        mode_labels_smoothed,
        mode_runs,
    ) = apply_mode_aware_fault_logic(
        base_scores=scores,
        base_flags=base_flags,
        threshold_array=base_thresholds,
        mode_labels=mode_labels,
        mode_probabilities=mode_probs,
        healthy_mode_stats=healthy_stats,
        mode_consistency_window=3,
        mode_stable_min_windows=2,
        mode_z_threshold=3.0,
    )

    assert mode_flags.tolist() == [False, False, True, True, True]
    assert np.allclose(mode_ref_thresholds, np.asarray([5.0, 5.0, 5.0, 5.0, 5.0]))
    assert mode_labels_smoothed.tolist() == ["Pump", "Pump", "Pump", "Pump", "Pump"]
    assert mode_runs.tolist() == [1, 2, 3, 4, 5]
    assert np.allclose(mode_conf, np.asarray([0.9, 0.85, 0.8, 0.92, 0.88]))
    assert mode_z[2] > 3.0


def test_mode_aware_gating_skips_modes_without_healthy_reference() -> None:
    scores = np.asarray([7.0, 7.5, 8.0, 9.0], dtype=np.float32)
    base_flags = np.asarray([True, True, True, True], dtype=bool)
    base_thresholds = np.asarray([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    mode_labels = np.asarray(["Pump", "Pump", "Unknown", "Unknown"], dtype=str)
    mode_probs = np.asarray(
        [
            [0.80, 0.20],
            [0.78, 0.22],
            [0.51, 0.49],
            [0.52, 0.48],
        ],
        dtype=np.float32,
    )
    healthy_stats = {"Pump": (2.0, 1.0)}

    mode_flags, mode_z, mode_ref_thresholds, _, _, _ = apply_mode_aware_fault_logic(
        base_scores=scores,
        base_flags=base_flags,
        threshold_array=base_thresholds,
        mode_labels=mode_labels,
        mode_probabilities=mode_probs,
        healthy_mode_stats=healthy_stats,
        mode_consistency_window=1,
        mode_stable_min_windows=2,
        mode_z_threshold=3.0,
    )

    assert mode_flags.tolist() == [False, True, False, False]
    assert mode_z.tolist() == [5.0, 5.5, 0.0, 0.0]
    assert mode_ref_thresholds.tolist() == [5.0, 5.0, 2.0, 2.0]
