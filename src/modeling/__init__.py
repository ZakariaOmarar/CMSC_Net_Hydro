"""Modeling package public surface (library-first)."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BaselineInferenceResult",
    "BaselineTrainingArtifacts",
    "FlowTrainingArtifacts",
    "LatentBuildSummary",
    "LatentDataset",
    "ModeTrainingArtifacts",
    "build_latent_cache",
    "collect_model_reports",
    "filter_healthy_latents",
    "infer_baseline_model",
    "load_flow_artifact",
    "load_latent_dataset",
    "load_mode_artifact",
    "score_with_context_smoothing",
    "train_all_models_main",
    "train_and_calibrate_flow",
    "train_baseline_model",
    "train_mode_classifier",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "BaselineInferenceResult": (".baselines", "BaselineInferenceResult"),
    "BaselineTrainingArtifacts": (".baselines", "BaselineTrainingArtifacts"),
    "FlowTrainingArtifacts": (".flow", "FlowTrainingArtifacts"),
    "LatentBuildSummary": (".latent", "LatentBuildSummary"),
    "LatentDataset": (".flow", "LatentDataset"),
    "ModeTrainingArtifacts": (".mode", "ModeTrainingArtifacts"),
    "build_latent_cache": (".latent", "build_latent_cache"),
    "collect_model_reports": (".reporting", "collect_model_reports"),
    "filter_healthy_latents": (".flow", "filter_healthy_latents"),
    "infer_baseline_model": (".baselines", "infer_baseline_model"),
    "load_flow_artifact": (".flow", "load_flow_artifact"),
    "load_latent_dataset": (".flow", "load_latent_dataset"),
    "load_mode_artifact": (".mode", "load_mode_artifact"),
    "score_with_context_smoothing": (".flow", "score_with_context_smoothing"),
    "train_all_models_main": (".orchestration", "main"),
    "train_and_calibrate_flow": (".flow", "train_and_calibrate_flow"),
    "train_baseline_model": (".baselines", "train_baseline_model"),
    "train_mode_classifier": (".mode", "train_mode_classifier"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
