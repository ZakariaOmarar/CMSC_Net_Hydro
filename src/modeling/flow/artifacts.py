"""Flow artifact loading and runtime feature preparation.

Handles the two responsibilities needed at inference time:
1. load_flow_artifact — deserializes a flow checkpoint and reconstructs the
   ConditionalRealNVP + LightweightContextEncoder from their saved state dicts.
2. prepare_feature_matrix — assembles the flat feature matrix (z concatenated with c,
   or a filtered subset) that the context encoder expects as input, handling dimension
   mismatches gracefully through candidate-feature probing.

resolve_flow_runtime is a thin helper that extracts the optional context encoder
and standardizer that may be attached to the flow model object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "flow.artifacts requires PyTorch. Install with: pip install torch"
    ) from exc

from ..core.artifact_contracts import validate_artifact_metadata
from .detection_head import ConditionalRealNVP
from .detection_head import FlowConfig
from .detection_head import LightweightContextEncoder
from ..models import build_conditional_flow
from .data import LatentDataset


@dataclass(frozen=True)
class FlowRuntimeArtifact:
    """Lightweight reference to a saved flow model and its calibrated threshold."""
    checkpoint_path: Path
    threshold: float


def resolve_flow_runtime(
    flow: ConditionalRealNVP,
) -> tuple[LightweightContextEncoder | None, np.ndarray | None, np.ndarray | None]:
    context_encoder = cast(
        LightweightContextEncoder | None,
        getattr(flow, "_context_encoder", None),
    )
    mean = cast(np.ndarray | None, getattr(flow, "_feature_mean", None))
    std = cast(np.ndarray | None, getattr(flow, "_feature_std", None))
    return context_encoder, mean, std


def prepare_feature_matrix(
    *,
    flow: ConditionalRealNVP,
    dataset: LatentDataset,
    context_encoder: LightweightContextEncoder,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    x_flat = np.concatenate([dataset.z, dataset.c], axis=1).astype(np.float32)
    x_z = np.asarray(dataset.z, dtype=np.float32)

    first = context_encoder._net[0]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("Invalid context encoder architecture")

    keep_idx = cast(np.ndarray | None, getattr(flow, "_feature_keep_indices", None))
    candidate_features: list[np.ndarray] = []

    if keep_idx is not None:
        idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
        if idx.size > 0:
            if np.all((idx >= 0) & (idx < x_flat.shape[1])):
                candidate_features.append(x_flat[:, idx])
            if np.all((idx >= 0) & (idx < x_z.shape[1])):
                candidate_features.append(x_z[:, idx])

    candidate_features.extend([x_flat, x_z])

    target_dim = int(first.in_features)
    x: np.ndarray | None = None
    for candidate in candidate_features:
        if candidate.shape[1] == target_dim:
            x = candidate
            break

    if x is None:
        keep_idx_len = (
            int(np.asarray(keep_idx, dtype=np.int64).reshape(-1).shape[0])
            if keep_idx is not None
            else 0
        )
        raise ValueError(
            f"Cannot infer CNF input features: expected dim {target_dim}, "
            f"got flat {x_flat.shape[1]} and z {x_z.shape[1]} "
            f"(feature_keep_indices={keep_idx_len})"
        )

    if mean is not None and std is not None:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
        std_arr = np.where(np.abs(std_arr) < 1e-6, 1.0, std_arr)
        if mean_arr.shape[0] == x.shape[1] and std_arr.shape[0] == x.shape[1]:
            x = ((x - mean_arr) / std_arr).astype(np.float32)

    if x.shape[1] != int(flow.config.feature_dim):
        raise ValueError(
            f"Feature dimension mismatch for CNF: expected {flow.config.feature_dim}, got {x.shape[1]}"
        )
    return x


def load_flow_artifact(
    artifact_path: Path,
    *,
    device: str = "cpu",
) -> tuple[ConditionalRealNVP, float]:
    try:
        blob = torch.load(Path(artifact_path), map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(Path(artifact_path), map_location=device)
    if not isinstance(blob, dict):
        raise ValueError("Invalid flow artifact payload")
    validate_artifact_metadata(blob=blob, expected_type="flow")

    cfg_raw = blob.get("flow_config")
    state_dict = blob.get("state_dict")
    threshold = blob.get("threshold")

    if not isinstance(cfg_raw, dict) or state_dict is None or threshold is None:
        raise ValueError(
            "Invalid artifact file: expected flow_config/state_dict/threshold"
        )

    def _to_int(name: str, value: Any) -> int:
        if value is None:
            raise ValueError(f"Invalid flow artifact: missing {name}")
        try:
            return int(value)
        except Exception as exc:
            raise ValueError(f"Invalid flow artifact: {name} must be an int") from exc

    # Support old artifacts that stored the dim under "d_model" instead of "feature_dim".
    feature_dim_raw = cfg_raw.get("feature_dim") or cfg_raw.get("d_model")
    feature_dim = _to_int("feature_dim", feature_dim_raw)
    hidden_dim = _to_int("hidden_dim", cfg_raw.get("hidden_dim"))

    cfg = FlowConfig(
        feature_dim=feature_dim,
        d_ctx=int(cfg_raw.get("d_ctx", 32)),
        n_layers=int(cfg_raw.get("n_layers", cfg_raw.get("n_coupling_layers", 8))),
        hidden_dim=hidden_dim,
    )
    flow = build_conditional_flow(cfg)
    flow.load_state_dict(state_dict)
    flow.eval()

    context_state = blob.get("context_encoder_state_dict")
    if isinstance(context_state, dict):
        context_encoder = LightweightContextEncoder(
            feature_dim=int(cfg.feature_dim),
            d_ctx=int(blob.get("context_dim", cfg.d_ctx)),
        )
        context_encoder.load_state_dict(context_state)
        context_encoder.eval()
        setattr(flow, "_context_encoder", context_encoder)

        mean = blob.get("scaler_mean")
        std = blob.get("scaler_std")
        if mean is not None and std is not None:
            setattr(flow, "_feature_mean", np.asarray(mean, dtype=np.float32))
            setattr(flow, "_feature_std", np.asarray(std, dtype=np.float32))

    keep_idx = blob.get("feature_keep_indices")
    if keep_idx is not None:
        setattr(flow, "_feature_keep_indices", np.asarray(keep_idx, dtype=np.int64))

    return flow, float(threshold)


__all__ = [
    "FlowRuntimeArtifact",
    "load_flow_artifact",
    "prepare_feature_matrix",
    "resolve_flow_runtime",
]
