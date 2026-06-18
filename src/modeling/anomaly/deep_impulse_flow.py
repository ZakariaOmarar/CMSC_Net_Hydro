"""Deep impulse-aware anomaly flow — learned raw front-end + anchored flow.

Why this exists: the SSL/CMA encoders optimise away the impulsive/spectral cues
a knock produces (Ridge bottleneck probe: embedding can't predict crest, R²≈0).
This model fixes that at the source:

  * a 1-D CNN reads the RAW waveform window directly (per modality) and is
    trained END-TO-END with a conditional normalizing flow on the HEALTHY
    negative-log-likelihood — a ONE-CLASS density objective, NOT contrastive /
    CMA, so transients are preserved instead of collapsed;
  * the validated hand-crafted impulse + spectral features are CONCATENATED to
    the learned embedding as a recall ANCHOR — they cannot be optimised away, so
    no knock slips through even if a new campaign looks different, and they also
    prevent the deep one-class objective from collapsing to a trivial solution;
  * a learned low-dim context head + the flow's context-conditional base
    normalise the healthy density per operating regime, so one global threshold
    transfers across campaigns.

Anomaly score = -log p([cnn_emb ⊕ anchor] | context).  Fit on healthy only;
sum-fuse the per-modality scores.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnf_head import ConditionalRealNVP


class RawCNN1D(nn.Module):
    """Compact 1-D CNN front-end on a fixed-length raw window -> embedding.

    Small-kernel strided convs keep it sensitive to sharp transients; modest
    depth/width suits the prototype-scale data (guards overfitting).
    """

    def __init__(self, in_len: int, emb_dim: int = 32, in_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 9, stride=4, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 9, stride=4, padding=4), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, 7, stride=4, padding=3), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, emb_dim, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.emb_dim = emb_dim

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.dim() == 2:
            w = w.unsqueeze(1)  # (B, L) -> (B, 1, L)
        return self.net(w).squeeze(-1)  # (B, emb_dim)


class DeepImpulseFlow(nn.Module):
    """Learned raw front-end + anchored conditional flow (one-class, per modality).

    `anchor` is the standardized hand-crafted impulse+spectral feature vector
    (computed outside, by `raw_impulse_detector.window_features`, then z-scored
    with healthy stats).  `context` is a learned low-dim regime descriptor; the
    flow's conditional base normalises by it.  Keeping the anchor as a fixed
    input (not produced by the CNN) is the anti-collapse / recall guarantee.
    """

    def __init__(self, in_len: int, n_anchor: int, *, emb_dim: int = 32,
                 ctx_dim: int = 8, flow_layers: int = 6, flow_hidden: int = 64) -> None:
        super().__init__()
        self.cnn = RawCNN1D(in_len, emb_dim)
        self.ctx_head = nn.Sequential(nn.Linear(emb_dim, ctx_dim), nn.Tanh())
        self.flow = ConditionalRealNVP(
            dim=emb_dim + n_anchor, c_dim=ctx_dim,
            n_layers=flow_layers, hidden_dim=flow_hidden, conditional_base=True,
        )
        self.in_len = in_len
        self.n_anchor = n_anchor

    def log_prob(self, raw: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        emb = self.cnn(raw)
        ctx = self.ctx_head(emb)
        x = torch.cat([emb, anchor], dim=1)
        return self.flow.log_prob(x, ctx)

    def anomaly_score(self, raw: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(raw, anchor)


__all__ = ["DeepImpulseFlow", "RawCNN1D"]
