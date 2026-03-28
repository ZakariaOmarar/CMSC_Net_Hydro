"""Cross-cutting runtime helpers used across train, inference, and eval modules.

Contains:

- enable_global_determinism — seeds Python/NumPy/Torch RNGs for reproducible runs.
- emit_event — prints structured JSON events for downstream log parsing.
- resolve_latent_paths — locates .npz latent cache files.
- compute_window_step_s — sliding-window arithmetic.
- is_healthy_recording_id / is_randomfault_recording / resolve_mode_label
  — convention-based recording classification from naming patterns.
- majority_smooth_labels / run_lengths — post-processing for mode predictions.
- fit_standardizer / apply_standardizer — per-feature z-score normalization.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def enable_global_determinism(seed: int) -> dict[str, object]:
    """Align Python/NumPy/Torch RNGs for reproducible runs."""
    random.seed(int(seed))
    np.random.seed(int(seed))

    torch_enabled = False
    cuda_enabled = False
    try:
        import torch

        torch_enabled = True
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            cuda_enabled = True
            torch.cuda.manual_seed_all(int(seed))

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    except Exception:
        torch_enabled = False
        cuda_enabled = False

    return {
        "seed": int(seed),
        "numpy": True,
        "python_random": True,
        "torch": bool(torch_enabled),
        "cuda": bool(cuda_enabled),
    }


def emit_event(event: str, *, quiet: bool, **payload: object) -> None:
    """Print a structured JSON event unless quiet output is requested."""
    if quiet:
        return
    body: dict[str, object] = {"event": event}
    body.update(payload)
    print(json.dumps(body))


def resolve_latent_paths(
    latent_root: Path,
    *,
    raise_if_missing: bool = True,
) -> list[Path]:
    """Resolve one latent .npz file or all .npz files under a directory."""
    root = Path(latent_root)
    if root.is_file():
        return [root]
    if root.is_dir():
        files = sorted(root.glob("*.npz"))
        if files:
            return files
    if raise_if_missing:
        raise FileNotFoundError(f"No latent .npz files found under {root}")
    return []


def compute_window_step_s(*, window_s: float, overlap: float) -> float:
    """Compute sliding-window step in seconds from size and overlap."""
    if window_s <= 0.0:
        raise ValueError("window_s must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    return float(window_s * (1.0 - overlap))


def is_randomfault_recording(recording_id: str) -> bool:
    """Return True when recording id indicates RandomFault data."""
    return "randomfault" in recording_id.strip().lower()


def is_healthy_recording_id(recording_id: str) -> bool:
    """Healthy-data filter rule: exclude recordings containing RandomFault."""
    return not is_randomfault_recording(recording_id)


def resolve_mode_label(recording_id: str) -> str:
    """Map recording id naming convention to canonical mode labels."""
    rid = recording_id.strip()
    lower = rid.lower()

    if is_randomfault_recording(rid):
        return "Turbine"
    if "turbine" in lower:
        return "Turbine"
    if "pump" in lower:
        return "Pump"
    if (
        "standstill" in lower
        or "standstil" in lower
        or "stand_still" in lower
        or "stand still" in lower
    ):
        return "Standstill"

    return rid if rid else "Unknown"


def majority_smooth_labels(labels: np.ndarray, *, window: int) -> np.ndarray:
    """Apply trailing majority-vote smoothing over string labels."""
    if window <= 1:
        return labels.astype(str)

    out: list[str] = []
    arr = labels.astype(str)
    for i in range(arr.shape[0]):
        start = max(0, i - window + 1)
        chunk = arr[start : i + 1]
        values, counts = np.unique(chunk, return_counts=True)
        out.append(str(values[int(np.argmax(counts))]))
    return np.asarray(out, dtype=str)


def run_lengths(labels: np.ndarray) -> np.ndarray:
    """Compute run length of current label at each position."""
    arr = labels.astype(str)
    out = np.zeros(arr.shape[0], dtype=np.int32)
    if arr.shape[0] == 0:
        return out

    out[0] = 1
    for i in range(1, arr.shape[0]):
        if arr[i] == arr[i - 1]:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 1
    return out


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature mean/std with numerical floor on std."""
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(
    x: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply precomputed per-feature standardization."""
    return ((x - mean) / std).astype(np.float32)
