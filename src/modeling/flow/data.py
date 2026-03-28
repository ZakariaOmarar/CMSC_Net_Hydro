"""Data contracts and loading helpers for flow-model training and inference.

latent cache files (.npz) are the shared data interface between the preprocessing
pipeline and all model families. This module defines the LatentDataset container
that holds the in-memory view of those files, and provides:

- load_latent_dataset: two-pass loader that validates shape consistency across
  files and streams data into pre-allocated arrays to avoid peak memory doubling.
- filter_healthy_latents: removes RandomFault recordings from the training set.
- split_healthy_train_val_test_indices: stratified split by recording ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.runtime_utils import is_healthy_recording_id


@dataclass(frozen=True)
class LatentDataset:
    """In-memory container for all latent windows loaded from .npz cache files.

    Attributes:
        z: Diagnostic feature vectors, shape (n_windows, d_z).
        c: Context feature vectors, shape (n_windows, d_c).
        recording_id: Source recording name for each window, used for healthy/fault
            filtering and train/val/test stratification by recording.
        is_transition_window: Boolean mask marking windows that fall near a
            Pump↔Turbine mode transition, which receive special anomaly-gating logic.
    """

    z: np.ndarray
    c: np.ndarray
    recording_id: np.ndarray
    is_transition_window: np.ndarray


def load_latent_dataset(latent_paths: Iterable[Path]) -> LatentDataset:
    paths = [Path(p) for p in latent_paths]
    if not paths:
        raise ValueError("No latent windows loaded")

    rows_per_file: list[int] = []
    z_dim: int | None = None
    c_dim: int | None = None

    # Pass 1: validate and determine total allocation shape.
    for p in paths:
        with np.load(p, allow_pickle=False) as blob:
            if "z" not in blob or "c" not in blob:
                raise ValueError(f"{p} must contain arrays 'z' and 'c'")

            z = np.asarray(blob["z"], dtype=np.float32)
            c = np.asarray(blob["c"], dtype=np.float32)
            if z.ndim != 2 or c.ndim != 2 or z.shape[0] != c.shape[0]:
                raise ValueError(
                    f"{p}: expected z and c shapes (n, d), same n; got {z.shape} and {c.shape}"
                )

            if z_dim is None:
                z_dim = int(z.shape[1])
                c_dim = int(c.shape[1])
            elif int(z.shape[1]) != int(z_dim) or int(c.shape[1]) != int(c_dim):
                raise ValueError(
                    f"{p}: latent feature dims mismatch; expected z={z_dim}, c={c_dim}, "
                    f"got z={z.shape[1]}, c={c.shape[1]}"
                )

            rows_per_file.append(int(z.shape[0]))

    total_rows = int(sum(rows_per_file))
    if total_rows <= 0:
        raise ValueError("Loaded dataset is empty")
    if z_dim is None or c_dim is None:
        raise ValueError("Unable to infer latent feature dimensions")

    z_all = np.empty((total_rows, int(z_dim)), dtype=np.float32)
    c_all = np.empty((total_rows, int(c_dim)), dtype=np.float32)
    # Use object dtype during fill to avoid fixed-width string truncation (e.g., "Pump" -> "P").
    rid_all = np.empty((total_rows,), dtype=object)
    tr_all = np.empty((total_rows,), dtype=bool)

    # Pass 2: stream each file into the preallocated output tensors.
    cursor = 0
    for p, n_rows in zip(paths, rows_per_file):
        with np.load(p, allow_pickle=False) as blob:
            z = np.asarray(blob["z"], dtype=np.float32)
            c = np.asarray(blob["c"], dtype=np.float32)

            if "recording_id" in blob:
                rid = np.asarray(blob["recording_id"]).astype(str)
            else:
                rid = np.asarray([p.stem] * int(n_rows), dtype=str)

            if "is_transition_window" in blob:
                tr = np.asarray(blob["is_transition_window"]).astype(bool)
            else:
                tr = np.zeros(int(n_rows), dtype=bool)

            if rid.shape[0] != int(n_rows) or tr.shape[0] != int(n_rows):
                raise ValueError(
                    f"{p}: recording_id/is_transition_window length must match n_windows"
                )

            end = cursor + int(n_rows)
            z_all[cursor:end] = z
            c_all[cursor:end] = c
            rid_all[cursor:end] = rid
            tr_all[cursor:end] = tr
            cursor = end

    return LatentDataset(
        z=z_all,
        c=c_all,
        recording_id=rid_all.astype(str),
        is_transition_window=tr_all,
    )


def filter_healthy_latents(dataset: LatentDataset) -> LatentDataset:
    mask = np.asarray(
        [is_healthy_recording_id(str(rid)) for rid in dataset.recording_id],
        dtype=bool,
    )
    if not np.any(mask):
        raise ValueError(
            "No healthy windows left after filtering RandomFault recordings"
        )

    return LatentDataset(
        z=dataset.z[mask],
        c=dataset.c[mask],
        recording_id=dataset.recording_id[mask],
        is_transition_window=dataset.is_transition_window[mask],
    )


def split_train_val_indices(
    recording_ids: np.ndarray,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    unique_ids = np.unique(recording_ids.astype(str))
    rng = np.random.default_rng(seed)

    if unique_ids.shape[0] >= 2:
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)
        n_val_ids = max(1, int(round(shuffled.shape[0] * val_ratio)))
        val_set = set(shuffled[:n_val_ids].tolist())
        val_mask = np.asarray([rid in val_set for rid in recording_ids], dtype=bool)
    else:
        idx = np.arange(recording_ids.shape[0])
        rng.shuffle(idx)
        n_val = max(1, int(round(recording_ids.shape[0] * val_ratio)))
        val_mask = np.zeros(recording_ids.shape[0], dtype=bool)
        val_mask[idx[:n_val]] = True

    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("Train/val split produced empty partition")

    return np.where(train_mask)[0], np.where(val_mask)[0]


def split_healthy_train_val_test_indices(
    recording_ids: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    train_val_idx, val_idx = split_train_val_indices(
        recording_ids,
        val_ratio=val_ratio,
        seed=seed,
    )

    if test_ratio <= 0.0:
        return train_val_idx, val_idx, np.zeros((0,), dtype=np.int64)

    adjusted_ratio = test_ratio / (1.0 - val_ratio)
    train_local, test_local = split_train_val_indices(
        recording_ids[train_val_idx],
        val_ratio=adjusted_ratio,
        seed=seed + 1337,
    )
    train_idx = train_val_idx[train_local]
    test_idx = train_val_idx[test_local]
    return train_idx, val_idx, test_idx


__all__ = [
    "LatentDataset",
    "filter_healthy_latents",
    "load_latent_dataset",
    "split_healthy_train_val_test_indices",
    "split_train_val_indices",
]
