from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.flow.detection_head import (
    ConditionalRealNVP,
    ContextSmoother,
    FlowConfig,
    apply_transition_policy,
    calibrate_threshold,
    filter_healthy_recording_ids,
    is_healthy_recording_id,
    train_flow_epoch,
)


def test_realnvp_forward_inverse_consistency() -> None:
    cfg = FlowConfig(feature_dim=8, d_ctx=4, n_layers=6, hidden_dim=32)
    flow = ConditionalRealNVP(cfg)

    z = torch.randn(5, cfg.d_model)
    c = torch.randn(5, cfg.d_ctx)

    u, log_det = flow(z, c)
    z_rec = flow.inverse(u, c)

    assert u.shape == z.shape
    assert log_det.shape == (5,)
    assert torch.allclose(z, z_rec, atol=1e-5, rtol=1e-5)


def test_anomaly_score_matches_negative_log_likelihood() -> None:
    cfg = FlowConfig(feature_dim=8, d_ctx=4, n_layers=4, hidden_dim=32)
    flow = ConditionalRealNVP(cfg)

    z = torch.randn(7, cfg.d_model)
    c = torch.randn(7, cfg.d_ctx)

    ll = flow.log_likelihood(z, c)
    score = flow.anomaly_score(z, c)

    assert ll.shape == (7,)
    assert score.shape == (7,)
    assert torch.allclose(score, -ll)


def test_context_smoother_exponential_weighting() -> None:
    smoother = ContextSmoother(k=3, decay=0.5)
    smoother.update(torch.tensor([1.0, 1.0]))
    smoother.update(torch.tensor([2.0, 2.0]))
    smoother.update(torch.tensor([3.0, 3.0]))

    smoothed = smoother.smooth()
    expected_scalar = (0.25 * 1.0 + 0.5 * 2.0 + 1.0 * 3.0) / (0.25 + 0.5 + 1.0)

    assert smoothed.shape == (2,)
    assert torch.allclose(smoothed, torch.full((2,), expected_scalar))


def test_transition_policies() -> None:
    scores = torch.tensor([0.5, 1.2, 2.1, 0.8])
    transitions = torch.tensor([False, True, True, False])

    flags_none, thr_none = apply_transition_policy(
        scores,
        threshold=1.0,
        is_transition_window=transitions,
        policy="context_only",
    )
    assert thr_none.shape == scores.shape
    assert flags_none.tolist() == [False, True, True, False]

    flags_expand, thr_expand = apply_transition_policy(
        scores,
        threshold=1.0,
        is_transition_window=transitions,
        policy="expand_threshold",
        transition_factor=1.5,
    )
    assert np.isclose(float(thr_expand[1].item()), 1.5)
    assert flags_expand.tolist() == [False, False, True, False]

    flags_suppress, _ = apply_transition_policy(
        scores,
        threshold=1.0,
        is_transition_window=transitions,
        policy="suppress",
    )
    assert flags_suppress.tolist() == [False, False, False, False]


def test_threshold_calibration_and_healthy_filtering() -> None:
    threshold = calibrate_threshold([0.1, 0.2, 0.3, 0.4, 10.0], percentile=80)
    assert threshold > 0.3

    recording_ids = ["Pump", "Turbine", "RandomFault", "randomfault_experiment"]
    healthy = filter_healthy_recording_ids(recording_ids)

    assert is_healthy_recording_id("Pump")
    assert not is_healthy_recording_id("RandomFault")
    assert healthy == ["Pump", "Turbine"]


def test_train_flow_epoch_runs_and_returns_finite_loss() -> None:
    cfg = FlowConfig(feature_dim=8, d_ctx=4, n_layers=4, hidden_dim=32)
    flow = ConditionalRealNVP(cfg)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4, weight_decay=1e-5)

    batches = [
        (torch.randn(16, cfg.d_model), torch.randn(16, cfg.d_ctx)),
        (torch.randn(16, cfg.d_model), torch.randn(16, cfg.d_ctx)),
    ]

    loss = train_flow_epoch(flow, batches=batches, optimizer=optimizer, grad_clip=1.0)

    assert np.isfinite(loss)
    assert loss > 0.0
