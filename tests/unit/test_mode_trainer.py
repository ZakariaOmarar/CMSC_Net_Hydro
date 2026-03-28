from __future__ import annotations

import json

import numpy as np
import pytest


pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.mode.train import train_mode_classifier


def _write_latent(path, *, recording_id: str, center: float, n: int = 60) -> None:
    rng = np.random.default_rng(abs(hash((str(path), recording_id))) % (2**32))
    z = rng.normal(center, 0.25, size=(n, 8)).astype(np.float32)
    c = rng.normal(center, 0.25, size=(n, 4)).astype(np.float32)
    rid = np.asarray([recording_id] * n, dtype=str)
    tr = np.zeros(n, dtype=bool)
    np.savez_compressed(path, z=z, c=c, recording_id=rid, is_transition_window=tr)


def test_mode_trainer_exports_epoch_accuracy_and_predictions(tmp_path) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent(latent_dir / "pump.npz", recording_id="Pump", center=-2.0)
    _write_latent(latent_dir / "turbine.npz", recording_id="Turbine", center=2.0)
    _write_latent(latent_dir / "standstill.npz", recording_id="Standstil", center=0.0)
    _write_latent(
        latent_dir / "randomfault.npz", recording_id="RandomFault", center=1.8
    )

    out_dir = tmp_path / "mode_artifacts"
    artifacts = train_mode_classifier(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=out_dir,
        epochs=4,
        batch_size=32,
        hidden_dim=32,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        device="cpu",
        quiet=True,
    )

    assert artifacts.artifact_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.predictions_path.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["trained_epochs"] == 4
    assert len(summary["history"]) == 4
    assert "Standstill" in summary["classes"]
    assert 0.0 <= float(summary["test_accuracy"]) <= 1.0
    assert 0.0 <= float(summary["test_macro_f1"]) <= 1.0
    assert summary["selection_metric"] in {"val_accuracy", "val_macro_f1"}

    for row in summary["history"]:
        assert 0.0 <= float(row["train_accuracy"]) <= 1.0
        assert 0.0 <= float(row["val_accuracy"]) <= 1.0
        assert 0.0 <= float(row["train_macro_f1"]) <= 1.0
        assert 0.0 <= float(row["val_macro_f1"]) <= 1.0

    preds = json.loads(artifacts.predictions_path.read_text(encoding="utf-8"))
    assert preds["n_test"] > 0
    assert len(preds["items"]) == preds["n_test"]
    first = preds["items"][0]
    assert "y_true" in first
    assert "y_pred" in first
    assert "recording_id" in first


def test_mode_trainer_can_exclude_randomfault_from_mode_split(tmp_path) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent(latent_dir / "pump.npz", recording_id="Pump", center=-2.0)
    _write_latent(latent_dir / "turbine.npz", recording_id="Turbine", center=2.0)
    _write_latent(latent_dir / "standstill.npz", recording_id="Standstil", center=0.0)
    _write_latent(
        latent_dir / "randomfault.npz", recording_id="RandomFault", center=1.8
    )

    out_dir = tmp_path / "mode_artifacts_no_fault"
    artifacts = train_mode_classifier(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=out_dir,
        epochs=3,
        batch_size=32,
        hidden_dim=32,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        exclude_randomfault=True,
        device="cpu",
        quiet=True,
    )

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["exclude_randomfault"] is True
    assert int(summary["excluded_randomfault_windows"]) > 0
    assert "Turbine" in summary["classes"]
    assert "Standstill" in summary["classes"]
    assert summary["use_class_weights"] is True

    preds = json.loads(artifacts.predictions_path.read_text(encoding="utf-8"))
    assert all(
        "randomfault" not in item["recording_id"].lower() for item in preds["items"]
    )
