"""Conditional RealNVP detection head and supporting components.

The core anomaly model learns the conditional distribution p(z | c) of diagnostic
feature vectors z given the operational context c. At inference time, windows with
low p(z | c) — i.e. high anomaly score –log p(z | c) — are flagged as anomalous.

Conditioning on c is the key mechanism that handles domain shift: the same z vector
may be perfectly normal in Turbine mode but anomalous in Pump mode, so the flow
adapts its density estimate to the current operational regime rather than
assuming a single stationary distribution.

Components:
  FlowConfig               — frozen config dataclass.
  FiLM                     — Feature-wise Linear Modulation; injects context into
                              each MLP layer via per-feature scale and shift.
  CouplingMLP              — FiLM-conditioned MLP producing affine (s, t) pairs.
  AffineCouplingLayer      — RealNVP affine coupling layer with exact log-det.
  ConditionalRealNVP       — The full flow: stacked coupling layers with alternating
                              split pattern and exact conditional log-likelihood.
  LightweightContextEncoder— Small MLP that maps flat feature vectors to context c.
  ContextSmoother          — Exponential-decay smoothing over recent context vectors
                              to stabilize anomaly scores during mode transitions.
  apply_transition_policy  — Adjusts thresholds or suppresses flags for windows
                              near Pump↔Turbine transitions.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

from ..core.runtime_utils import is_healthy_recording_id as _is_healthy_recording_id

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "detection_head requires PyTorch. Install with: pip install torch"
    ) from exc


TransitionPolicy = Literal["context_only", "expand_threshold", "suppress"]


@dataclass(frozen=True)
class FlowConfig:
    """Configuration for a conditional RealNVP anomaly detection head.

    Args:
        feature_dim: Dimensionality of the flat feature vector z fed into the flow.
            Must be even for the chunk-based affine coupling split.
        d_ctx: Dimensionality of the context vector c produced by LightweightContextEncoder.
        n_layers: Number of affine coupling layers. More layers improve expressivity
            but increase training cost.
        hidden_dim: Hidden unit count in each CouplingMLP.
        dropout: Dropout probability applied after the FiLM layer in each CouplingMLP.
            Regularises the flow against overfitting on small datasets.
    """

    feature_dim: int = 64
    d_ctx: int = 32
    n_layers: int = 8
    hidden_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim must be > 0")
        if int(self.feature_dim) % 2 != 0:
            raise ValueError("feature_dim must be even for chunk-based coupling")
        if int(self.n_layers) < 1:
            raise ValueError("n_layers must be >= 1")
        if int(self.d_ctx) <= 0:
            raise ValueError("d_ctx must be > 0")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        object.__setattr__(self, "feature_dim", int(self.feature_dim))
        object.__setattr__(self, "n_layers", int(self.n_layers))
        object.__setattr__(self, "d_ctx", int(self.d_ctx))
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))
        object.__setattr__(self, "dropout", float(self.dropout))

    @property
    def d_model(self) -> int:
        """Alias for feature_dim — kept for backward compatibility."""
        return self.feature_dim

    @property
    def n_coupling_layers(self) -> int:
        """Alias for n_layers — kept for backward compatibility."""
        return self.n_layers


class FiLM(nn.Module):
    """Feature-wise linear modulation that injects context into an intermediate representation.

    Produces per-feature scale and shift parameters from the context vector c,
    then applies gamma * h + beta to the hidden state h. This is how the flow
    model adapts its behavior to the current operating mode without needing
    separate model instances per mode.
    """

    def __init__(self, d_ctx: int, feature_dim: int) -> None:
        super().__init__()
        self._generator = nn.Linear(d_ctx, 2 * feature_dim)
        nn.init.normal_(self._generator.weight, std=0.01)
        nn.init.zeros_(self._generator.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or c.ndim != 2 or h.shape[0] != c.shape[0]:
            raise ValueError("h and c must both be 2D with matching batch size")
        params = self._generator(c)
        gamma_prime, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + gamma_prime
        return gamma * h + beta


class CouplingMLP(nn.Module):
    """FiLM-conditioned MLP producing affine coupling parameters s and t."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        d_ctx: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._fc1 = nn.Linear(input_dim, hidden_dim)
        self._film = FiLM(d_ctx=d_ctx, feature_dim=hidden_dim)
        self._drop = nn.Dropout(p=float(dropout))
        self._fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._fc3 = nn.Linear(hidden_dim, output_dim * 2)

    def forward(
        self, z1: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self._fc1(z1))
        h = self._film(h, c)
        h = self._drop(h)
        h = F.relu(self._fc2(h))
        out = self._fc3(h)
        s, t = out.chunk(2, dim=-1)
        return s, t


class AffineCouplingLayer(nn.Module):
    """RealNVP affine coupling layer."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, d_ctx: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if feature_dim % 2 != 0:
            raise ValueError("feature_dim must be even for chunk-based coupling")

        self._half = feature_dim // 2
        self._mlp = CouplingMLP(
            input_dim=self._half,
            hidden_dim=hidden_dim,
            output_dim=self._half,
            d_ctx=d_ctx,
            dropout=float(dropout),
        )

    def forward(
        self, z: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1, z2 = z.chunk(2, dim=-1)
        s, t = self._mlp(z1, c)
        s = torch.tanh(s)

        z2_out = z2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([z1, z2_out], dim=-1), log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z1, z2_out = z.chunk(2, dim=-1)
        s, t = self._mlp(z1, c)
        s = torch.tanh(s)

        z2 = (z2_out - t) * torch.exp(-s)
        return torch.cat([z1, z2], dim=-1)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP with FiLM-conditioned coupling layers."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        self.config = config
        self._feature_dim = int(config.feature_dim)
        self._layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    feature_dim=config.feature_dim,
                    hidden_dim=config.hidden_dim,
                    d_ctx=config.d_ctx,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(
        self, z: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent z to base space u and accumulate exact log-det Jacobian."""
        if z.ndim != 2 or c.ndim != 2 or z.shape[0] != c.shape[0]:
            raise ValueError("z and c must both be 2D with matching batch size")
        if z.shape[1] != self._feature_dim or c.shape[1] != self.config.d_ctx:
            raise ValueError(
                f"Expected z shape (_, {self._feature_dim}) and c shape (_, {self.config.d_ctx})"
            )

        x = z
        log_det_total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for i, layer in enumerate(self._layers):
            x, log_det = layer(x, c)
            log_det_total = log_det_total + log_det
            # Alternate transformed half between layers.
            if i % 2 == 0:
                x = torch.flip(x, dims=[-1])

        return x, log_det_total

    def inverse(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inverse mapping from base space u back to latent z."""
        if u.ndim != 2 or c.ndim != 2 or u.shape[0] != c.shape[0]:
            raise ValueError("u and c must both be 2D with matching batch size")

        x = u
        for i in reversed(range(len(self._layers))):
            if i % 2 == 0:
                x = torch.flip(x, dims=[-1])
            x = self._layers[i].inverse(x, c)  # type: ignore

        return x

    def log_likelihood(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute exact conditional log likelihood log p(z | c)."""
        u, log_det_total = self.forward(z, c)
        log_p_base = -0.5 * (u**2).sum(dim=-1) - 0.5 * self._feature_dim * math.log(
            2.0 * math.pi
        )
        return log_p_base + log_det_total

    def anomaly_score(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Return anomaly score = -log p(z | c); higher means more anomalous."""
        return -self.log_likelihood(z, c)


class ContextSmoother:
    """Exponential-decay smoothing applied to context vectors over a sliding window.

    Mode transitions are not instantaneous — the sensor signatures shift gradually
    over several windows. Smoothing the context vector c with an exponential decay
    prevents abrupt spikes in the anomaly score that would otherwise occur as the
    encoder switches from one mode’s embedding to another.

    Args:
        k: Maximum number of recent context vectors to retain in the buffer.
        decay: Exponential decay weight; higher values weight recent vectors more.
    """

    def __init__(self, k: int = 20, decay: float = 0.9) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")

        self._queue: deque[torch.Tensor] = deque(maxlen=k)
        self._decay = float(decay)

    def update(self, c_t: torch.Tensor) -> None:
        if c_t.ndim not in (1, 2):
            raise ValueError("c_t must be 1D or 2D")
        self._queue.append(c_t.detach().clone())

    def smooth(self) -> torch.Tensor:
        if not self._queue:
            raise RuntimeError("ContextSmoother queue is empty")

        items = list(self._queue)
        n = len(items)
        device = items[-1].device
        dtype = items[-1].dtype

        weights = torch.tensor(
            [self._decay ** (n - 1 - i) for i in range(n)],
            dtype=dtype,
            device=device,
        )
        weights = weights / weights.sum()

        out = torch.zeros_like(items[-1])
        for w, c in zip(weights, items):
            out = out + w * c
        return out


class LightweightContextEncoder(nn.Module):
    """Small two-layer MLP that maps flat feature vectors to a compact context embedding.

    During training this encoder is first pre-trained with a contrastive NT-Xent loss
    to learn an operational-state representation without labels, then kept frozen
    while the flow is trained to model p(z | c). At inference it runs once per window
    to produce the c vector that gates the anomaly score.
    """

    def __init__(self, feature_dim: int, d_ctx: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if d_ctx <= 0:
            raise ValueError("d_ctx must be > 0")

        self._net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_ctx),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D input (batch, feature_dim), got shape {tuple(x.shape)}"
            )
        return self._net(x)


def nt_xent_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy for paired views."""
    if emb_a.ndim != 2 or emb_b.ndim != 2 or emb_a.shape != emb_b.shape:
        raise ValueError("emb_a and emb_b must be 2D tensors with identical shape")
    if emb_a.shape[0] < 2:
        raise ValueError("NT-Xent requires batch size >= 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    z1 = F.normalize(emb_a, dim=-1)
    z2 = F.normalize(emb_b, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    logits = torch.matmul(z, z.T) / float(temperature)

    n = emb_a.shape[0]
    eye = torch.eye(2 * n, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, float("-inf"))

    positives = torch.cat([torch.diag(logits, n), torch.diag(logits, -n)], dim=0)
    denom = torch.logsumexp(logits, dim=1)
    return -(positives - denom).mean()


def _nll_loss(
    flow: ConditionalRealNVP, z: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Flow training loss over healthy windows: negative mean log-likelihood."""
    return -flow.log_likelihood(z, c).mean()


def calibrate_threshold(scores: Iterable[float], percentile: float = 99.0) -> float:
    """Calibrate anomaly threshold on healthy validation scores."""
    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must be in (0, 100)")
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("scores cannot be empty")
    return float(np.percentile(arr, percentile))


def apply_transition_policy(
    scores: torch.Tensor,
    *,
    threshold: float,
    is_transition_window: torch.Tensor | None = None,
    policy: TransitionPolicy = "context_only",
    transition_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply transition handling policy and return anomaly flags and thresholds."""
    if scores.ndim != 1:
        raise ValueError("scores must have shape (batch,)")

    thresholds = torch.full_like(scores, float(threshold))

    if is_transition_window is None:
        return scores > thresholds, thresholds

    if (
        is_transition_window.ndim != 1
        or is_transition_window.shape[0] != scores.shape[0]
    ):
        raise ValueError("is_transition_window must have shape (batch,)")

    if policy == "context_only":
        return scores > thresholds, thresholds

    if policy == "expand_threshold":
        expanded = thresholds.clone()
        expanded[is_transition_window] = expanded[is_transition_window] * float(
            transition_factor
        )
        return scores > expanded, expanded

    if policy == "suppress":
        flags = scores > thresholds
        flags[is_transition_window] = False
        return flags, thresholds


def is_healthy_recording_id(recording_id: str) -> bool:
    """Healthy-data filter rule: exclude recordings containing RandomFault."""
    return _is_healthy_recording_id(recording_id)


def filter_healthy_recording_ids(recording_ids: Iterable[str]) -> list[str]:
    """Return only healthy recording IDs by repository naming convention."""
    return [rid for rid in recording_ids if is_healthy_recording_id(rid)]


def train_flow_epoch(
    flow: ConditionalRealNVP,
    *,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> float:
    """Train one epoch over (z, c) batches and return mean epoch loss."""
    if grad_clip <= 0.0:
        raise ValueError("grad_clip must be > 0")

    flow.train()
    losses: list[float] = []

    for z_batch, c_batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = _nll_loss(flow, z_batch, c_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise ValueError("batches must contain at least one batch")

    return float(np.mean(losses))
