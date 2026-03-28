"""Baseline-model library surface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING
 
__all__ = [
    "BaselineInferenceResult",
    "BaselineTrainingArtifacts",
    "build_features",
    "compute_binary_metrics",
    "infer_baseline_model",
    "train_baseline_model",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "BaselineInferenceResult":  (".train", "BaselineInferenceResult"),
    "BaselineTrainingArtifacts": (".train", "BaselineTrainingArtifacts"),
    "build_features":           (".data",  "build_features"),
    "compute_binary_metrics":   (".metrics", "compute_binary_metrics"),
    "infer_baseline_model":     (".train", "infer_baseline_model"),
    "train_baseline_model":     (".train", "train_baseline_model"),
}
 
if TYPE_CHECKING:  # pragma: no cover
    from .data import build_features as build_features
    from .metrics import compute_binary_metrics as compute_binary_metrics
    from .train import BaselineInferenceResult as BaselineInferenceResult
    from .train import BaselineTrainingArtifacts as BaselineTrainingArtifacts
    from .train import infer_baseline_model as infer_baseline_model
    from .train import train_baseline_model as train_baseline_model
 
 
def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
 
    module_path, symbol = _EXPORTS[name]
    module = import_module(module_path, __name__)
    value = getattr(module, symbol)
 
    # Cache so the next access is a plain dict lookup.
    globals()[name] = value
    return value