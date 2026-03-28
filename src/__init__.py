"""Thesis hydropower starter package.

Top-level imports are resolved lazily so submodule entry-points like
``python -m src.modeling.flow.infer`` do not pull optional feature-stack
dependencies unless the symbols are actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DataSegment",
    "FeatureFrame",
]


def __getattr__(name: str) -> Any:
    if name == "DataSegment":
        from .data import DataSegment

        return DataSegment
    if name == "FeatureFrame":
        from .features import FeatureFrame

        return FeatureFrame
    raise AttributeError(f"module 'src' has no attribute {name!r}")
