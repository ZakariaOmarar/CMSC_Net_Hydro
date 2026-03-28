"""Mode-aware evaluation helpers for flow inference.

The operating mode (Turbine / Pump / Standstill) fundamentally changes the
acoustic and vibration signature of the machine. This module provides the
tools to exploit that structure:

- load_mode_artifact / predict_modes: run the trained mode classifier to label
  each inference window by its operating regime.
- filter_healthy_dataset: exclude RandomFault recordings from the baseline
  score distribution used in threshold calibration.
- apply_mode_aware_fault_logic: gate anomaly flags using mode predictions,
  suppressing false positives that arise during mode transitions.
- build_anomaly_events: group consecutive flagged windows into discrete events.
- healthy_mode_score_stats: compute per-mode (mean, std) score statistics
  for mode-stratified threshold calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..core.artifact_contracts import validate_artifact_metadata
from ..models import ModeCNN2DClassifier, ModeMLP
from ..core.runtime_utils import compute_window_step_s
from ..core.runtime_utils import is_healthy_recording_id
from ..core.runtime_utils import majority_smooth_labels
from ..core.runtime_utils import run_lengths
from .data import LatentDataset


@dataclass(frozen=True)
class ModeArtifactBundle:
    """Loaded mode classifier together with its class list and input standardizer.

    Produced by load_mode_artifact() and consumed by predict_modes() and
    healthy_mode_score_stats().
    """

    model: ModeMLP | ModeCNN2DClassifier
    classes: list[str]
    mean: np.ndarray
    std: np.ndarray


def load_mode_artifact(
    artifact_path: Path,
    *,
    device: str,
) -> ModeArtifactBundle:
    try:
        blob = torch.load(Path(artifact_path), map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(Path(artifact_path), map_location=device)
    if not isinstance(blob, dict):
        raise ValueError("Invalid mode artifact payload")
    validate_artifact_metadata(blob=blob, expected_type="mode")

    state_dict = blob.get("state_dict")
    input_dim = blob.get("input_dim")
    hidden_dim = blob.get("hidden_dim")
    classes = blob.get("classes")
    mean = blob.get("mean")
    std = blob.get("std")
    mode_architecture = str(blob.get("mode_architecture", "mlp")).strip().lower()

    if (
        state_dict is None
        or input_dim is None
        or hidden_dim is None
        or classes is None
        or mean is None
        or std is None
    ):
        raise ValueError(
            "Invalid mode artifact: expected state_dict/input_dim/hidden_dim/classes/mean/std"
        )

    if mode_architecture == "cnn2d":
        model = ModeCNN2DClassifier(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            n_classes=int(len(classes)),
        ).to(torch.device(device))
    else:
        model = ModeMLP(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            n_classes=int(len(classes)),
        ).to(torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    std_arr = np.where(np.abs(std_arr) < 1e-6, 1.0, std_arr)
    classes_arr = [str(c) for c in classes]

    return ModeArtifactBundle(
        model=model,
        classes=classes_arr,
        mean=mean_arr,
        std=std_arr,
    )


def predict_modes(
    bundle: ModeArtifactBundle,
    dataset: LatentDataset,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([dataset.z, dataset.c], axis=1).astype(np.float32)
    x_norm = (x - bundle.mean) / bundle.std

    torch_device = torch.device(device)
    with torch.no_grad():
        x_t = torch.from_numpy(x_norm).to(device=torch_device, dtype=torch.float32)
        logits = bundle.model(x_t)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)

    idx = np.argmax(probs, axis=1)
    labels = np.asarray([bundle.classes[int(i)] for i in idx], dtype=str)
    return labels, probs


def healthy_mode_score_stats(
    healthy_dataset: LatentDataset,
    mode_bundle: ModeArtifactBundle,
    *,
    score_fn: Callable[[LatentDataset], np.ndarray],
    device: str,
) -> dict[str, tuple[float, float]]:
    healthy_scores = score_fn(healthy_dataset)
    mode_labels, _ = predict_modes(mode_bundle, healthy_dataset, device=device)

    stats: dict[str, tuple[float, float]] = {}
    for mode in np.unique(mode_labels):
        sel = mode_labels == mode
        vals = healthy_scores[sel]
        if vals.size == 0:
            continue
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma < 1e-6:
            sigma = 1.0
        stats[str(mode)] = (mu, sigma)
    return stats


def filter_healthy_dataset(dataset: LatentDataset) -> LatentDataset:
    rid = dataset.recording_id.astype(str)
    mask = np.asarray([is_healthy_recording_id(r) for r in rid], dtype=bool)
    if not np.any(mask):
        raise ValueError("No healthy windows found in healthy-latent-root")
    return LatentDataset(
        z=dataset.z[mask],
        c=dataset.c[mask],
        recording_id=dataset.recording_id[mask],
        is_transition_window=dataset.is_transition_window[mask],
    )


def apply_mode_aware_fault_logic(
    *,
    base_scores: np.ndarray,
    base_flags: np.ndarray,
    threshold_array: np.ndarray,
    mode_labels: np.ndarray,
    mode_probabilities: np.ndarray,
    healthy_mode_stats: dict[str, tuple[float, float]],
    mode_consistency_window: int,
    mode_stable_min_windows: int,
    mode_z_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    smoothed_modes = majority_smooth_labels(
        mode_labels,
        window=max(1, int(mode_consistency_window)),
    )
    mode_runs = run_lengths(smoothed_modes)

    mode_z = np.zeros(base_scores.shape[0], dtype=np.float32)
    mode_ref_thresholds = np.copy(threshold_array).astype(np.float32)
    mode_flags = np.zeros(base_scores.shape[0], dtype=bool)

    for i in range(base_scores.shape[0]):
        mode = str(smoothed_modes[i])
        mu_sigma = healthy_mode_stats.get(mode)
        if mu_sigma is None:
            continue

        mu, sigma = mu_sigma
        z = float((base_scores[i] - mu) / sigma)
        mode_z[i] = float(z)
        mode_ref_thresholds[i] = float(mu + mode_z_threshold * sigma)

        stable = int(mode_runs[i]) >= int(mode_stable_min_windows)
        above_base = bool(base_flags[i])
        above_mode = float(base_scores[i]) > float(mode_ref_thresholds[i])
        mode_flags[i] = bool(stable and above_base and above_mode)

    max_prob = np.max(mode_probabilities, axis=1).astype(np.float32)
    return (
        mode_flags,
        mode_z,
        mode_ref_thresholds,
        max_prob,
        smoothed_modes,
        mode_runs.astype(np.int32),
    )


def build_anomaly_events(
    *,
    scores: np.ndarray,
    flags: np.ndarray,
    thresholds: np.ndarray,
    window_s: float,
    overlap: float,
) -> list[dict[str, float | int]]:
    step_s = compute_window_step_s(window_s=window_s, overlap=overlap)
    events: list[dict[str, float | int]] = []

    for idx, is_anomaly in enumerate(flags.astype(bool).tolist()):
        if not is_anomaly:
            continue
        events.append(
            {
                "window_index": int(idx),
                "timestamp_s": float(idx * step_s),
                "score": float(scores[idx]),
                "threshold": float(thresholds[idx]),
            }
        )

    return events


__all__ = [
    "ModeArtifactBundle",
    "apply_mode_aware_fault_logic",
    "build_anomaly_events",
    "filter_healthy_dataset",
    "healthy_mode_score_stats",
    "load_mode_artifact",
    "predict_modes",
]
