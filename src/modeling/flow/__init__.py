"""Flow-model library surface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "FlowRuntimeArtifact",
    "FlowTrainingArtifacts",
    "LatentDataset",
    "filter_healthy_latents",
    "load_flow_artifact",
    "load_latent_dataset",
    "score_with_context_smoothing",
    "train_and_calibrate_flow",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "FlowRuntimeArtifact": (".artifacts", "FlowRuntimeArtifact"),
    "FlowTrainingArtifacts": (".train", "FlowTrainingArtifacts"),
    "LatentDataset": (".data", "LatentDataset"),
    "filter_healthy_latents": (".data", "filter_healthy_latents"),
    "load_flow_artifact": (".artifacts", "load_flow_artifact"),
    "load_latent_dataset": (".data", "load_latent_dataset"),
    "score_with_context_smoothing": (".infer", "score_with_context_smoothing"),
    "train_and_calibrate_flow": (".train", "train_and_calibrate_flow"),
}


if TYPE_CHECKING:
    from .artifacts import FlowRuntimeArtifact as FlowRuntimeArtifact
    from .artifacts import load_flow_artifact as load_flow_artifact
    from .data import LatentDataset as LatentDataset
    from .data import filter_healthy_latents as filter_healthy_latents
    from .data import load_latent_dataset as load_latent_dataset
    from .infer import score_with_context_smoothing as score_with_context_smoothing
    from .train import FlowTrainingArtifacts as FlowTrainingArtifacts
    from .train import train_and_calibrate_flow as train_and_calibrate_flow


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
