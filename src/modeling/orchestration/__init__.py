"""Model orchestration package."""

from __future__ import annotations

from importlib import import_module

__all__ = ["main"]


def __getattr__(name: str):
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".train_all", __name__)
    value = getattr(module, "main")
    globals()[name] = value
    return value
