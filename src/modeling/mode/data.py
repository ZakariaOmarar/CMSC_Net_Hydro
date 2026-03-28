"""Data preparation and split helpers for mode classifier training.

Mode labels (Pump / Turbine / Standstill) are derived from recording ID naming
conventions. This module builds the feature matrix x = concat(z, c) from latent
dataset objects and implements a stratified per-class split that guarantees
all three splits (train / val / test) contain at least one sample per class.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "mode.data requires PyTorch. Install with: pip install torch"
    ) from exc


def split_stratified_indices(
    y: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio <= 0.0 or val_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("train_ratio, val_ratio, test_ratio must all be > 0")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].astype(np.int64)
        rng.shuffle(cls_idx)
        n = cls_idx.shape[0]

        expected = np.asarray(
            [train_ratio * n, val_ratio * n, test_ratio * n], dtype=np.float64
        )
        counts = np.floor(expected).astype(np.int64)
        remainder = int(n - int(counts.sum()))

        if remainder > 0:
            frac = expected - counts
            order = np.argsort(-frac)
            for i in range(remainder):
                counts[order[i % 3]] += 1

        if n >= 3:
            for k in range(3):
                if counts[k] == 0:
                    donor = int(np.argmax(counts))
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[k] += 1

        n_train = int(counts[0])
        n_val = int(counts[1])
        train_parts.append(cls_idx[:n_train])
        val_parts.append(cls_idx[n_train : n_train + n_val])
        test_parts.append(cls_idx[n_train + n_val :])

    train_idx = np.concatenate(train_parts, axis=0)
    val_idx = np.concatenate(val_parts, axis=0)
    test_idx = np.concatenate(test_parts, axis=0)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "Split produced an empty partition. Increase data size or adjust ratios."
        )

    return train_idx, val_idx, test_idx


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
):
    idx = np.arange(x.shape[0], dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, idx.shape[0], batch_size):
        bidx = idx[start : start + batch_size]
        x_t = torch.from_numpy(x[bidx]).to(device=device, dtype=torch.float32)
        y_t = torch.from_numpy(y[bidx]).to(device=device, dtype=torch.long)
        yield x_t, y_t


__all__ = [
    "batch_iterator",
    "split_stratified_indices",
]
