"""V0 LSTM-AE anomaly baseline on log-mel windows (Khamaisi reference).

Self-contained: reads via `TestDatasetLoader`, computes log-mel via
`src/features/audio_spectral.py`, fits an LSTM-AE on healthy windows, scores
RandomFault windows by per-window reconstruction MSE.  No coupling to the
existing CNF latent cache.

Per-window scoring is converted to per-mode anomaly thresholds following the
same 95th/99th-percentile logic as `src/modeling/mode/p5_anomaly/per_mode_baseline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud

from ...features.audio_spectral import compute_log_mel_spectrogram
from ...ingestion.test_dataset_loader import (
    DatasetSpec,
    TestDatasetLoader,
    TestDatasetSegment,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V0Config:
    """Hyperparameters for the V0 LSTM-AE baseline."""

    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    window_seconds: float = 1.0  # one log-mel window
    window_overlap: float = 0.5  # 50 % overlap between consecutive windows
    hidden_dim: int = 128
    latent_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.2
    healthy_modes: tuple[str, ...] = ("Pump", "Standstill", "Turbine", "Healthy")
    anomaly_modes: tuple[str, ...] = ("RandomFault",)
    seed: int = 42
    device: str = "cpu"

    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction → sliding windows
# ---------------------------------------------------------------------------


def extract_log_mel_windows(
    segment: TestDatasetSegment,
    cfg: V0Config,
    *,
    pool: str = "mean",
) -> np.ndarray:
    """Compute log-mel for every mic, pool across mics, then slide windows.

    Returns ``(n_windows, n_frames_per_window, n_mels)`` float32.

    The pooling strategy is mean over mic channels, so the V0 baseline ingests a
    single mel-spectrogram per recording.  This is the simplest unconditional
    baseline; the channel-aware fusion arrives in V1+.
    """
    fs = int(segment.segment.mic_sample_rate)
    mels = []
    for ch in range(segment.segment.n_mic_channels):
        m = compute_log_mel_spectrogram(
            segment.segment.mic_data[ch],
            fs=fs,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )
        mels.append(m)
    arr = np.stack(mels, axis=0)  # (n_mics, n_mels, n_frames)
    if pool == "mean":
        pooled = arr.mean(axis=0)
    elif pool == "max":
        pooled = arr.max(axis=0)
    else:
        raise ValueError(f"unknown pool {pool!r}")
    # pooled: (n_mels, n_frames)

    frames_per_window = max(1, int(round(cfg.window_seconds * fs / cfg.hop_length)))
    step = max(1, int(round(frames_per_window * (1.0 - cfg.window_overlap))))
    n_frames = pooled.shape[1]
    if n_frames < frames_per_window:
        return np.zeros((0, frames_per_window, cfg.n_mels), dtype=np.float32)

    windows = []
    for start in range(0, n_frames - frames_per_window + 1, step):
        windows.append(pooled[:, start : start + frames_per_window].T)  # (T, F)
    if not windows:
        return np.zeros((0, frames_per_window, cfg.n_mels), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMAutoencoderV0(nn.Module):
    """Two-layer LSTM encoder + decoder, latent bottleneck.

    The encoder consumes a mel sequence ``(B, T, F)`` and produces a single
    latent ``(B, D_lat)`` from the last hidden state.  The decoder unrolls T
    steps from a repeated latent and reconstructs ``(B, T, F)``.

    Reconstruction error is the per-window mean squared error.
    """

    def __init__(self, cfg: V0Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.LSTM(
            input_size=cfg.n_mels,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.output = nn.Linear(cfg.hidden_dim, cfg.n_mels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)  # h_n: (n_layers, B, hidden)
        return self.to_latent(h_n[-1])  # (B, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        T = x.shape[1]
        seed = self.from_latent(z)  # (B, hidden)
        decoder_in = seed.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
        out, _ = self.decoder(decoder_in)
        return self.output(out)  # (B, T, F)

    @torch.no_grad()
    def reconstruction_score(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return ((x_hat - x) ** 2).mean(dim=(1, 2))  # (B,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    model: LSTMAutoencoderV0
    train_loss_history: list[float]
    val_loss_history: list[float]
    standardiser_mean: np.ndarray
    standardiser_std: np.ndarray
    healthy_train_recordings: list[str]
    healthy_val_recordings: list[str]


def _gather_healthy_windows(
    segments: Iterable[TestDatasetSegment], cfg: V0Config
) -> tuple[np.ndarray, list[str]]:
    """Collect (n_windows, T, F) plus the recording id of each window.

    Healthy = `is_anomaly=False` (covers D1/D2 mode folders **and** D3/D4
    speed-bucket recordings whose mode is unrecorded — both contribute
    valid healthy training material for the V0 reference model).
    The legacy `cfg.healthy_modes` filter is no longer used since it
    excluded D3/D4 (which now carry `mode_label = None`).
    """
    all_windows: list[np.ndarray] = []
    rec_ids_per_window: list[str] = []
    for s in segments:
        if s.is_anomaly:
            continue
        w = extract_log_mel_windows(s, cfg)
        if w.shape[0] == 0:
            continue
        all_windows.append(w)
        rec_ids_per_window.extend([s.recording_id] * w.shape[0])
    if not all_windows:
        return np.zeros((0, 0, cfg.n_mels), dtype=np.float32), []
    return np.concatenate(all_windows, axis=0), rec_ids_per_window


def _split_by_recording(
    rec_ids: list[str], val_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(rec_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_ids = unique[:n_val]
    train_ids = unique[n_val:]
    if not train_ids:
        # Tiny dataset (e.g. D1 with 4 recordings + val_ratio=0.5): keep ≥ 1 train.
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def train_v0_lstm_ae(
    loader: TestDatasetLoader, cfg: V0Config | None = None
) -> TrainResult:
    """Train the V0 LSTM-AE on healthy recordings of one dataset."""
    cfg = cfg or V0Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    segments = loader.list_segments()
    windows, rec_ids = _gather_healthy_windows(segments, cfg)
    if windows.shape[0] == 0:
        raise RuntimeError("no healthy windows found for V0 training")

    train_ids, val_ids = _split_by_recording(rec_ids, cfg.val_ratio, cfg.seed)
    train_mask = np.array([r in train_ids for r in rec_ids], dtype=bool)
    val_mask = np.array([r in val_ids for r in rec_ids], dtype=bool)

    train_x = windows[train_mask]
    val_x = windows[val_mask]

    # Standardise per-mel-bin on the training data, apply to val.
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1)) + 1e-6
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    device = torch.device(cfg.device)
    model = LSTMAutoencoderV0(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    train_loader = tud.DataLoader(
        tud.TensorDataset(torch.from_numpy(train_x)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = tud.DataLoader(
        tud.TensorDataset(torch.from_numpy(val_x)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(cfg.epochs):
        model.train()
        epoch_train = 0.0
        n_train = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optim.zero_grad()
            xb_hat = model(xb)
            loss = loss_fn(xb_hat, xb)
            loss.backward()
            optim.step()
            epoch_train += float(loss.item()) * xb.shape[0]
            n_train += xb.shape[0]
        train_history.append(epoch_train / max(1, n_train))

        model.eval()
        epoch_val = 0.0
        n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_hat = model(xb)
                loss = loss_fn(xb_hat, xb)
                epoch_val += float(loss.item()) * xb.shape[0]
                n_val += xb.shape[0]
        val_history.append(epoch_val / max(1, n_val))

    return TrainResult(
        model=model,
        train_loss_history=train_history,
        val_loss_history=val_history,
        standardiser_mean=mean.astype(np.float32),
        standardiser_std=std.astype(np.float32),
        healthy_train_recordings=sorted(train_ids),
        healthy_val_recordings=sorted(val_ids),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_recordings(
    model: LSTMAutoencoderV0,
    standardiser_mean: np.ndarray,
    standardiser_std: np.ndarray,
    segments: Iterable[TestDatasetSegment],
    cfg: V0Config,
) -> list[dict]:
    """Score every window of every segment.

    Returns a list of records, one per recording, with per-window scores plus
    the parsed mode/op-condition/spatial labels.
    """
    device = next(model.parameters()).device
    model.eval()
    records: list[dict] = []
    for s in segments:
        windows = extract_log_mel_windows(s, cfg)
        if windows.shape[0] == 0:
            continue
        norm = (windows - standardiser_mean) / standardiser_std
        x = torch.from_numpy(norm).to(device)
        with torch.no_grad():
            scores = model.reconstruction_score(x).cpu().numpy()
        records.append(
            {
                "dataset_id": s.dataset_id,
                "recording_id": s.recording_id,
                "mode_label": s.mode_label,
                "op_condition": s.op_condition,
                "spatial_label": s.spatial_label,
                "n_windows": int(scores.shape[0]),
                "scores": scores.astype(np.float32),
            }
        )
    return records


__all__ = [
    "V0Config",
    "LSTMAutoencoderV0",
    "TrainResult",
    "extract_log_mel_windows",
    "train_v0_lstm_ae",
    "score_recordings",
]
