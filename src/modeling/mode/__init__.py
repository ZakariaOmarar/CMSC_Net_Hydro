"""Mode-model library surface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ModeArtifactBundle",
    "ModeTrainingArtifacts",
    "load_mode_artifact",
    "train_mode_classifier",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "ModeArtifactBundle": ("..flow.eval", "ModeArtifactBundle"),
    "ModeTrainingArtifacts": (".train", "ModeTrainingArtifacts"),
    "load_mode_artifact": ("..flow.eval", "load_mode_artifact"),
    "train_mode_classifier": (".train", "train_mode_classifier"),
}


if TYPE_CHECKING:  # pragma: no cover
    from ..flow.eval import ModeArtifactBundle as ModeArtifactBundle
    from ..flow.eval import load_mode_artifact as load_mode_artifact
    from .train import ModeTrainingArtifacts as ModeTrainingArtifacts
    from .train import train_mode_classifier as train_mode_classifier


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
