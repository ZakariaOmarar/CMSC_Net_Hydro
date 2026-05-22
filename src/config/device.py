"""Device resolution: honour an explicit device, otherwise pick CUDA when available."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device spec to a concrete torch.device.

    - "auto" / None / "cuda" without a GPU available -> falls back to "cpu".
    - Any other explicit string (e.g. "cuda:1", "cpu", "mps") is honoured.
    """
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        spec = device.type if device.index is None else f"{device.type}:{device.index}"
    else:
        spec = str(device).strip().lower()

    if spec in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)
