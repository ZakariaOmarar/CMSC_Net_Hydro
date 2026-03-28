"""Data preparation helpers for the OC-SVM, LSTM-AE, and CNN-AE baseline models.

All baselines train and score on latent cache data (z, c) loaded from .npz files.
This module provides:

- build_features: selects a feature subset (z only, c only, or z+c).
- split_recording_ids: shuffled recording-level train/val split.
- build_sequences: creates sliding sub-windows of z/c rows for sequential models.
- prepare_autoencoder_input: packs sequences into (batch, seq, feature) tensors.
- aggregate_sequence_errors: collapses per-sub-window errors back to the original
  window granularity by taking the maximum reconstruction error.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import torch

from ..flow.data import LatentDataset

FeatureSet = Literal["z", "c", "zc"]
ModelType = Literal["ocsvm", "lstm_ae", "cnn_ae"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def build_features(dataset: LatentDataset, *, feature_set: FeatureSet) -> np.ndarray:
    """Select a feature matrix from the latent dataset.

    Args:
        feature_set: ``"z"`` for diagnostic features only, ``"c"`` for context
            features only, or ``"zc"`` for the full concatenation.
    """
    if feature_set == "z":
        x = dataset.z 
    elif feature_set == "c":
        x = dataset.c 
    else:
        x = np.concatenate([dataset.z, dataset.c], axis=1) 

    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------


def split_recording_ids(
    recording_ids: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio!r}")

    unique_ids = np.unique(recording_ids.astype(str))

    # Edge case: a single recording cannot be split.
    if unique_ids.shape[0] < 2:
        return {str(unique_ids[0])}, set()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n_val = max(1, int(round(shuffled.shape[0] * val_ratio)))
    val_ids = set(shuffled[:n_val].tolist())
    train_ids = set(shuffled[n_val:].tolist())

    # Guard against degenerate ratios that drain all recordings into val.
    if not train_ids:
        promoted = val_ids.pop()
        train_ids.add(promoted)

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------


def build_sequences(
    x: np.ndarray,
    recording_ids: np.ndarray,
    *,
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if seq_len <= 1:
        raise ValueError(f"seq_len must be > 1, got {seq_len!r}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride!r}")

    rid = recording_ids.astype(str)
    sequences: list[np.ndarray] = []
    seq_indices: list[np.ndarray] = []

    n = rid.shape[0]
    start = 0

    while start < n:
        # Find the contiguous run of windows belonging to this recording.
        end = start + 1
        while end < n and rid[end] == rid[start]:
            end += 1

        local_idx = np.arange(start, end, dtype=np.int64)
        local_x = x[local_idx]
        m = local_x.shape[0]  # windows in this recording

        if m < seq_len:
            # Pad the recording to exactly seq_len by repeating the last frame.
            n_pad = seq_len - m
            padded = np.concatenate([local_x, np.repeat(local_x[-1:], n_pad, axis=0)])
            idx = np.concatenate([local_idx, np.repeat(local_idx[-1], n_pad)])
            sequences.append(padded.astype(np.float32))
            seq_indices.append(idx.astype(np.int64))
        else:
            # Emit all full stride-aligned windows.
            last_valid_start = m - seq_len
            for s in range(0, last_valid_start + 1, stride):
                sequences.append(local_x[s : s + seq_len].astype(np.float32))
                seq_indices.append(local_idx[s : s + seq_len])

            if last_valid_start % stride != 0:
                sequences.append(
                    local_x[last_valid_start : last_valid_start + seq_len].astype(
                        np.float32
                    )
                )
                seq_indices.append(
                    local_idx[last_valid_start : last_valid_start + seq_len]
                )

        start = end

    if not sequences:
        raise ValueError(
            "No sequences could be built -- check seq_len vs dataset size."
        )

    return np.stack(sequences, axis=0), seq_indices


# ---------------------------------------------------------------------------
# Log-spectrogram conversion
# ---------------------------------------------------------------------------


def sequences_to_log_spectrograms(
    sequences: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if sequences.ndim != 3:
        raise ValueError(
            f"sequences must be 3-D (n_seq, seq_len, feat_dim), got shape {sequences.shape}"
        )

    # Flatten (seq_len, feat_dim) -> signal of length seq_len * feat_dim.
    flat = np.asarray(sequences, dtype=np.float32).reshape(sequences.shape[0], -1)
    signal_len = flat.shape[1]

    if signal_len < 2:
        raise ValueError(
            f"Flattened signal length is {signal_len}; must be >= 2 to compute an STFT."
        )

    # Clamp n_fft to a valid even value <= signal length.
    n_fft_eff = min(int(n_fft), signal_len)
    if n_fft_eff % 2 != 0:
        n_fft_eff -= 1
    n_fft_eff = max(2, n_fft_eff)

    # Clamp hop to [1, n_fft // 2] to avoid aliasing artifacts.
    hop_eff = max(1, min(int(hop_length), n_fft_eff // 2))

    x_t = torch.from_numpy(flat)
    window = torch.hann_window(n_fft_eff, dtype=x_t.dtype, device=x_t.device)

    with torch.no_grad():
        stft = torch.stft(
            x_t,
            n_fft=n_fft_eff,
            hop_length=hop_eff,
            win_length=n_fft_eff,
            window=window,
            return_complex=True,
            center=True,
        )
        log_spec = torch.log1p(torch.abs(stft))

    return log_spec.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Autoencoder input preparation
# ---------------------------------------------------------------------------


def prepare_autoencoder_input(
    *,
    model_type: ModelType,
    sequences: np.ndarray,
    cnn_spec_n_fft: int,
    cnn_spec_hop_length: int,
) -> np.ndarray:
    if model_type == "cnn_ae":
        return sequences_to_log_spectrograms(
            sequences,
            n_fft=int(cnn_spec_n_fft),
            hop_length=int(cnn_spec_hop_length),
        )

    return np.asarray(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# Error aggregation
# ---------------------------------------------------------------------------


def aggregate_sequence_errors(
    *,
    errors: np.ndarray,
    seq_indices: list[np.ndarray],
    n_windows: int,
) -> np.ndarray:
    scores = np.zeros(n_windows, dtype=np.float32)
    counts = np.zeros(n_windows, dtype=np.float32)

    for err, idx in zip(errors.astype(np.float32), seq_indices):
        # idx may contain repeated indices (padding); np.add.at handles them.
        valid = idx[(idx >= 0) & (idx < n_windows)]
        np.add.at(scores, valid, float(err))
        np.add.at(counts, valid, 1.0)

    # Avoid division by zero for windows that were never covered.
    counts = np.where(counts > 0.0, counts, 1.0)
    return (scores / counts).astype(np.float32)


__all__ = [
    "aggregate_sequence_errors",
    "build_features",
    "build_sequences",
    "prepare_autoencoder_input",
    "sequences_to_log_spectrograms",
    "split_recording_ids",
]
