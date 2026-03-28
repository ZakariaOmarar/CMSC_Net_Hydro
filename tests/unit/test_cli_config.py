from __future__ import annotations

import pytest

from src.modeling.cli import load_yaml_config


def test_load_yaml_config_normalizes_dashed_keys(tmp_path) -> None:
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("hidden-dim: 256\nno-class-weights: true\n", encoding="utf-8")

    loaded = load_yaml_config(cfg)

    assert loaded["hidden_dim"] == 256
    assert loaded["no_class_weights"] is True


def test_load_yaml_config_rejects_non_mapping(tmp_path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="key-value mapping"):
        load_yaml_config(cfg)


def test_load_yaml_config_missing_file_raises(tmp_path) -> None:
    missing = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml_config(missing)
