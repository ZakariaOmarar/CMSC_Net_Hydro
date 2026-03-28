from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.flow.train_core import build_flat_features, cluster_score_stats


def test_build_flat_features_shape_and_dtype() -> None:
    z = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    c = np.asarray([[10.0], [20.0]], dtype=np.float32)

    x = build_flat_features(z, c)

    assert x.shape == (2, 3)
    assert x.dtype == np.float32
    assert np.allclose(x[0], np.asarray([1.0, 2.0, 10.0], dtype=np.float32))


def test_cluster_score_stats_returns_cluster_metrics() -> None:
    context = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.2, 5.1]],
        dtype=np.float32,
    )
    scores = np.asarray([1.0, 1.2, 3.8, 4.1], dtype=np.float32)

    stats = cluster_score_stats(
        context,
        scores,
        seed=42,
        max_clusters=2,
    )

    assert stats
    assert len(stats) <= 2
    for cluster_name, payload in stats.items():
        assert cluster_name.startswith("cluster_")
        for key in ("count", "mean", "std", "p95", "p99"):
            assert key in payload
