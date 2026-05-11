"""Per-modality encoder: plain CNN backbone + Set-Transformer pool.

Two backbones, both deliberately plain per the plan's smart-decisions table
(no CBAM, no BiLSTM, no Twins-Transformer):

  - `Acoustic2DCNN`: 2-D CNN on `(2, F, T)` log-mel + CWT input from
    `src/features/audio_spectral.py`.
  - `Vibration1DCNN`: 1-D CNN on `(3, T)` amplitude + Hilbert envelope +
    rolling-kurtosis input from `src/features/vibration_temporal.py`.

`PerModalityEncoder` wires a backbone to the channel-agnostic Set-Transformer
pool (`ChannelTokenEnricher` → MAB → PMA(num_seeds=1)).  It returns BOTH:

  - per-channel tokens, shape `(B, N, embed_dim)`, for V2's cross-attention
    fusion to consume after V1 weight transfer; and
  - a single PMA summary, shape `(B, embed_dim)`, for V1's contrastive loss
    and the cluster-purity sanity gate.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .set_transformer import MAB, PMA, ChannelTokenEnricher


class Acoustic2DCNN(nn.Module):
    """Plain 2-D CNN backbone applied per microphone.

    Input:  `(B, N_mic, 2, F, T)` — channel 0 is log-mel, channel 1 is CWT
    Output: `(B, N_mic, feature_dim)` — per-mic feature vector
    """

    def __init__(self, in_channels: int = 2, feature_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Acoustic2DCNN expects (B, N, C, F, T); got {tuple(x.shape)}")
        B, N, C, F, T = x.shape
        flat = x.reshape(B * N, C, F, T)
        h = self.cnn(flat).flatten(start_dim=1)  # (B*N, 128)
        h = self.proj(h)  # (B*N, feature_dim)
        return h.reshape(B, N, -1)


class Vibration1DCNN(nn.Module):
    """Plain 1-D CNN backbone applied per vibration channel.

    Input:  `(B, N_vib, 3, T)` — channels are amplitude / envelope / kurtosis
    Output: `(B, N_vib, feature_dim)` — per-vibration feature vector
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Vibration1DCNN expects (B, N, C, T); got {tuple(x.shape)}")
        B, N, C, T = x.shape
        flat = x.reshape(B * N, C, T)
        h = self.cnn(flat).flatten(start_dim=1)
        h = self.proj(h)
        return h.reshape(B, N, -1)


class PerModalityEncoder(nn.Module):
    """CNN backbone → ChannelTokenEnricher → MAB → PMA(1).

    Returns `(tokens, summary)` so V2 can consume the token sequence and V1
    can consume the summary.
    """

    def __init__(
        self,
        modality: Literal["acoustic", "vibration"],
        feature_dim: int = 128,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_modalities: int = 2,
        n_datasets: int = 5,
    ) -> None:
        super().__init__()
        self.modality = modality
        if modality == "acoustic":
            self.backbone: nn.Module = Acoustic2DCNN(in_channels=2, feature_dim=feature_dim)
            self.modality_idx = 0
        elif modality == "vibration":
            self.backbone = Vibration1DCNN(in_channels=3, feature_dim=feature_dim)
            self.modality_idx = 1
        else:
            raise ValueError(f"unknown modality {modality!r}")

        self.enricher = ChannelTokenEnricher(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_modalities=n_modalities,
            n_datasets=n_datasets,
        )
        self.self_attn = MAB(embed_dim, num_heads=n_heads)
        self.pma = PMA(embed_dim, num_seeds=1, num_heads=n_heads)
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,  # (B, N, ...) feature tensor for this modality
        xyz: torch.Tensor,  # (B, N, 3) sensor positions in metres
        dataset_idx: torch.Tensor,  # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)  # (B, N, feature_dim)
        tokens = self.enricher(feats, xyz, self.modality_idx, dataset_idx)  # (B, N, embed_dim)
        tokens = self.self_attn(tokens, tokens)  # one self-attention pass
        summary = self.pma(tokens).squeeze(1)  # (B, embed_dim)
        return tokens, summary


__all__ = ["Acoustic2DCNN", "Vibration1DCNN", "PerModalityEncoder"]
