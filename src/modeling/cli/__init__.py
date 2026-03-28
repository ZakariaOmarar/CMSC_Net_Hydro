"""CLI helpers for modeling package."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_config(config_path: Path) -> dict[str, object]:
    """Load a YAML mapping and normalize keys to argparse-style names."""
    path = Path(config_path)

    candidates: list[Path] = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".yaml"))

    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        configs_dir = repo_root / "configs"
        base_names = [path]
        if path.suffix == "":
            base_names.append(path.with_suffix(".yaml"))
        for base in (repo_root, configs_dir):
            for name in base_names:
                candidates.append(base / name)

    resolved: Path | None = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            resolved = candidate
            break

    if resolved is None:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"Config file not found. Tried: {tried}")

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a key-value mapping")

    cfg: dict[str, object] = {}
    for key, value in raw.items():
        normalized = str(key).strip().lstrip("-").replace("-", "_")
        cfg[normalized] = value
    return cfg


__all__ = ["load_yaml_config"]
