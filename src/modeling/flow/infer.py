"""Inference runner for the conditional flow detection head.

Loads a trained flow model from disk, scores each window in a latent dataset,
and applies transition-aware thresholding to produce the final anomaly flags.
Context smoothing (exponential EMA over recent c vectors) stabilizes scores
during mode transitions, where the operational context shifts gradually.

The main public entry point is score_with_context_smoothing().
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "flow_infer requires PyTorch. Install with: pip install torch"
    ) from exc

from .detection_head import (
    ConditionalRealNVP,
    ContextSmoother,
    TransitionPolicy,
    apply_transition_policy,
)
from .artifacts import load_flow_artifact, prepare_feature_matrix, resolve_flow_runtime
from .eval import ModeArtifactBundle
from .eval import apply_mode_aware_fault_logic
from .eval import build_anomaly_events
from .eval import filter_healthy_dataset
from .eval import healthy_mode_score_stats as _healthy_mode_score_stats
from .eval import load_mode_artifact
from .eval import predict_modes
from .data import LatentDataset, load_latent_dataset


@dataclass(frozen=True)
class FlowInferenceResult:
    """Packed inference outputs for one latent dataset.

    Attributes:
        scores: Per-window anomaly scores (higher = more anomalous).
        flags: Boolean anomaly flags after threshold and transition policy.
        thresholds: Per-window effective threshold (may differ near transitions).
    """

    scores: np.ndarray
    flags: np.ndarray
    thresholds: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_with_context_only(
    flow: ConditionalRealNVP,
    dataset: LatentDataset,
    *,
    smoother_k: int,
    smoother_decay: float,
    device: str,
) -> np.ndarray:
    z = dataset.z
    c = dataset.c
    if z.ndim != 2 or c.ndim != 2 or z.shape[0] != c.shape[0]:
        raise ValueError("Expected dataset z/c shapes (n, d) with matching n")

    torch_device = torch.device(device)
    flow = flow.to(torch_device)
    smoother = ContextSmoother(k=smoother_k, decay=smoother_decay)

    context_encoder, mean, std = resolve_flow_runtime(flow)

    if context_encoder is None:
        scores: list[float] = []
        with torch.no_grad():
            for i in range(z.shape[0]):
                z_t = torch.from_numpy(z[i : i + 1]).to(
                    device=torch_device, dtype=torch.float32
                )
                c_t = torch.from_numpy(c[i : i + 1]).to(
                    device=torch_device, dtype=torch.float32
                )
                smoother.update(c_t.squeeze(0))
                c_smooth = smoother.smooth().unsqueeze(0)
                score_t = flow.anomaly_score(z_t, c_smooth)
                scores.append(float(score_t.detach().cpu().item()))
        return np.asarray(scores, dtype=np.float32)

    context_encoder = context_encoder.to(torch_device)
    context_encoder.eval()
    x = prepare_feature_matrix(
        flow=flow,
        dataset=dataset,
        context_encoder=context_encoder,
        mean=mean,
        std=std,
    )

    scores = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            x_t = torch.from_numpy(x[i : i + 1]).to(
                device=torch_device, dtype=torch.float32
            )
            c_t = context_encoder(x_t)
            smoother.update(c_t.squeeze(0))
            c_smooth = smoother.smooth().unsqueeze(0)
            score_t = flow.anomaly_score(x_t, c_smooth)
            scores.append(float(score_t.detach().cpu().item()))

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def healthy_mode_score_stats(
    flow: ConditionalRealNVP,
    healthy_dataset: LatentDataset,
    mode_bundle: ModeArtifactBundle,
    *,
    smoother_k: int,
    smoother_decay: float,
    device: str,
) -> dict[str, tuple[float, float]]:
    return _healthy_mode_score_stats(
        healthy_dataset,
        mode_bundle,
        score_fn=lambda ds: _score_with_context_only(
            flow,
            ds,
            smoother_k=smoother_k,
            smoother_decay=smoother_decay,
            device=device,
        ),
        device=device,
    )


def _scores_to_result(
    scores: list[float],
    transitions: np.ndarray,
    *,
    threshold: float,
    transition_policy: TransitionPolicy,
    transition_factor: float,
) -> FlowInferenceResult:
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    flags_t, thresholds_t = apply_transition_policy(
        score_tensor,
        threshold=float(threshold),
        is_transition_window=torch.from_numpy(transitions),
        policy=transition_policy,
        transition_factor=float(transition_factor),
    )
    return FlowInferenceResult(
        scores=score_tensor.numpy(),
        flags=flags_t.numpy(),
        thresholds=thresholds_t.numpy(),
    )


def score_with_context_smoothing(
    flow: ConditionalRealNVP,
    dataset: LatentDataset,
    *,
    threshold: float,
    smoother_k: int = 20,
    smoother_decay: float = 0.9,
    transition_policy: TransitionPolicy = "context_only",
    transition_factor: float = 1.5,
    device: str = "cpu",
) -> FlowInferenceResult:
    z = dataset.z
    c = dataset.c
    transitions = dataset.is_transition_window.astype(bool)

    if z.ndim != 2 or c.ndim != 2 or z.shape[0] != c.shape[0]:
        raise ValueError("Expected dataset z/c shapes (n, d) with matching n")

    torch_device = torch.device(device)
    flow = flow.to(torch_device)
    smoother = ContextSmoother(k=smoother_k, decay=smoother_decay)
    context_encoder, mean, std = resolve_flow_runtime(flow)

    scores: list[float] = []
    with torch.no_grad():
        if context_encoder is None:
            for i in range(z.shape[0]):
                z_t = torch.from_numpy(z[i : i + 1]).to(
                    device=torch_device, dtype=torch.float32
                )
                c_t = torch.from_numpy(c[i : i + 1]).to(
                    device=torch_device, dtype=torch.float32
                )
                smoother.update(c_t.squeeze(0))
                scores.append(
                    float(
                        flow.anomaly_score(z_t, smoother.smooth().unsqueeze(0)).item()
                    )
                )
        else:
            context_encoder = context_encoder.to(torch_device)
            context_encoder.eval()
            x = prepare_feature_matrix(
                flow=flow,
                dataset=dataset,
                context_encoder=context_encoder,
                mean=mean,
                std=std,
            )
            for i in range(x.shape[0]):
                x_t = torch.from_numpy(x[i : i + 1]).to(
                    device=torch_device, dtype=torch.float32
                )
                c_t = context_encoder(x_t)
                smoother.update(c_t.squeeze(0))
                scores.append(
                    float(
                        flow.anomaly_score(x_t, smoother.smooth().unsqueeze(0)).item()
                    )
                )

    return _scores_to_result(
        scores,
        transitions,
        threshold=threshold,
        transition_policy=transition_policy,
        transition_factor=transition_factor,
    )
