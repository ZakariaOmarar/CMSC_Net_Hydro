"""Reporting package wrappers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["collect_model_reports", "to_payload"]


_EXPORTS: dict[str, tuple[str, str]] = {
    "collect_model_reports": (".report", "collect_model_reports"),
    "to_payload": (".report", "to_payload"),
}


if TYPE_CHECKING:
    from .report import collect_model_reports as collect_model_reports
    from .report import to_payload as to_payload


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
