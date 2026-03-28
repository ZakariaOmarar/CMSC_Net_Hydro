from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.flow.artifacts import prepare_feature_matrix
from src.modeling.flow.data import LatentDataset
from src.modeling.flow.detection_head import LightweightContextEncoder


def _dataset(*, n: int = 8, z_dim: int = 188, c_dim: int = 7) -> LatentDataset:
    z = np.random.default_rng(0).normal(0.0, 1.0, size=(n, z_dim)).astype(np.float32)
    c = np.random.default_rng(1).normal(0.0, 1.0, size=(n, c_dim)).astype(np.float32)
    rid = np.asarray(["RandomFault"] * n, dtype=str)
    tr = np.zeros((n,), dtype=bool)
    return LatentDataset(z=z, c=c, recording_id=rid, is_transition_window=tr)


def test_prepare_feature_matrix_uses_selected_features_before_dim_match() -> None:
    ds = _dataset()
    encoder = LightweightContextEncoder(feature_dim=132, d_ctx=32)

    flow = cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(feature_dim=132),
            _feature_keep_indices=np.arange(132, dtype=np.int64),
        ),
    )

    x = prepare_feature_matrix(
        flow=flow,
        dataset=ds,
        context_encoder=encoder,
        mean=None,
        std=None,
    )

    assert x.shape == (ds.z.shape[0], 132)
    expected = np.concatenate([ds.z, ds.c], axis=1).astype(np.float32)[:, :132]
    assert np.allclose(x, expected)


def test_prepare_feature_matrix_raises_on_unresolvable_dims() -> None:
    ds = _dataset()
    encoder = LightweightContextEncoder(feature_dim=132, d_ctx=32)

    flow = cast(
        Any,
        SimpleNamespace(
            config=SimpleNamespace(feature_dim=132),
            _feature_keep_indices=np.arange(999, 999 + 10, dtype=np.int64),
        ),
    )

    with pytest.raises(ValueError, match="Cannot infer CNF input features"):
        prepare_feature_matrix(
            flow=flow,
            dataset=ds,
            context_encoder=encoder,
            mean=None,
            std=None,
        )
