"""V2 fusion: a single bidirectional cross-attention block.

Takes two per-modality token sequences from `PerModalityEncoder` (acoustic and
vibration), and produces fused versions where each modality has attended over
the other.  Implementation is two `MAB` cross-attention passes — one per
direction — sharing no weights.

Per the plan's smart-decisions table, this is **one block**.  Twins-Transformer
dual-branch and multi-scale spatiotemporal cross-attention are deferred.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.set_transformer import MAB


class BidirectionalCrossAttention(nn.Module):
    """One bidirectional cross-attention block.

    forward:
      acoustic_q  ──cross-attn over──▶ vibration_kv  →  fused_acoustic
      vibration_q ──cross-attn over──▶ acoustic_kv   →  fused_vibration

    Each direction is one `MAB` (pre-norm transformer block).  Output shapes
    match the inputs.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.a_from_v = MAB(dim, num_heads=num_heads, dropout=dropout)
        self.v_from_a = MAB(dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        acoustic_tokens: torch.Tensor,  # (B, N_a, D)
        vibration_tokens: torch.Tensor,  # (B, N_v, D)
        acoustic_key_padding_mask: torch.Tensor | None = None,
        vibration_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if acoustic_tokens.shape[-1] != vibration_tokens.shape[-1]:
            raise ValueError(
                "BidirectionalCrossAttention requires both modalities to share embed_dim; "
                f"got {acoustic_tokens.shape[-1]} vs {vibration_tokens.shape[-1]}"
            )
        fused_acoustic = self.a_from_v(
            acoustic_tokens, vibration_tokens, key_padding_mask=vibration_key_padding_mask
        )
        fused_vibration = self.v_from_a(
            vibration_tokens, acoustic_tokens, key_padding_mask=acoustic_key_padding_mask
        )
        return fused_acoustic, fused_vibration


__all__ = ["BidirectionalCrossAttention"]
