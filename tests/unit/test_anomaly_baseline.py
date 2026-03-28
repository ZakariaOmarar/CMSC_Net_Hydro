from __future__ import annotations

import json

import numpy as np
import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)
pytest.importorskip("sklearn", exc_type=ImportError)

from src.modeling.baselines.train import (
    build_anomaly_events,
    build_mode_predictions,
    infer_baseline_model,
    train_baseline_model,
)
from src.modeling.core.artifact_contracts import stamp_artifact_metadata
from src.modeling.flow.data import load_latent_dataset
from src.modeling.models import ModeMLP


def _write_latent(path, *, recording_id: str, center: float, n: int = 56) -> None:
    rng = np.random.default_rng(abs(hash((str(path), recording_id))) % (2**32))
    z = rng.normal(center, 0.25, size=(n, 8)).astype(np.float32)
    c = rng.normal(center, 0.25, size=(n, 4)).astype(np.float32)
    rid = np.asarray([recording_id] * n, dtype=str)
    tr = np.zeros(n, dtype=bool)
    tr[::9] = True
    np.savez_compressed(path, z=z, c=c, recording_id=rid, is_transition_window=tr)


@pytest.mark.parametrize("model_type", ["ocsvm", "lstm_ae", "cnn_ae"])
def test_baseline_train_and_infer_roundtrip(tmp_path, model_type: str) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent(latent_dir / "pump.npz", recording_id="Pump", center=-2.0)
    _write_latent(latent_dir / "turbine.npz", recording_id="Turbine", center=1.7)
    _write_latent(
        latent_dir / "standstill.npz",
        recording_id="Standstil",
        center=-0.2,
    )
    _write_latent(
        latent_dir / "randomfault.npz",
        recording_id="RandomFault",
        center=4.0,
    )

    out_dir = tmp_path / "baseline_artifacts"
    kwargs = {
        "latent_paths": sorted(latent_dir.glob("*.npz")),
        "output_dir": out_dir,
        "model_type": model_type,
        "feature_set": "zc",
        "val_ratio": 0.2,
        "score_percentile": 95.0,
        "exclude_randomfault": True,
        "seed": 42,
    }
    if model_type in {"lstm_ae", "cnn_ae"}:
        kwargs.update(
            {
                "seq_len": 12,
                "seq_stride": 4,
                "hidden_dim": 24,
                "latent_dim": 12,
                "epochs": 3,
                "batch_size": 32,
                "patience": 2,
                "device": "cpu",
            }
        )

    artifacts = train_baseline_model(**kwargs)

    assert artifacts.artifact_path.exists()
    assert artifacts.summary_path.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["model_type"] == model_type
    assert float(summary["threshold"]) > 0.0
    assert "healthy_val_score_mean" in summary
    assert "full_anomaly_rate" in summary
    assert 0.0 <= float(summary["full_anomaly_rate"]) <= 1.0

    result = infer_baseline_model(
        artifact_path=artifacts.artifact_path,
        latent_paths=sorted(latent_dir.glob("*.npz")),
        device="cpu",
    )

    n_total = 56 * 4
    assert result.scores.shape == (n_total,)
    assert result.flags.shape == (n_total,)
    assert result.thresholds.shape == (n_total,)
    assert np.all(np.isfinite(result.scores))
    assert np.sum(result.flags) > 0


def test_baseline_infer_can_emit_mode_predictions(tmp_path) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent(latent_dir / "pump.npz", recording_id="Pump", center=-1.2, n=40)
    _write_latent(latent_dir / "turbine.npz", recording_id="Turbine", center=1.2, n=40)

    baseline_out = tmp_path / "baseline_artifacts"
    artifacts = train_baseline_model(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=baseline_out,
        model_type="ocsvm",
        feature_set="zc",
        val_ratio=0.2,
        score_percentile=95.0,
        exclude_randomfault=True,
        seed=42,
    )

    mode_model = ModeMLP(input_dim=12, hidden_dim=8, n_classes=3)
    mode_artifact = tmp_path / "mode_classifier.pt"
    torch.save(
        {
            "_meta": stamp_artifact_metadata(artifact_type="mode"),
            "state_dict": mode_model.state_dict(),
            "input_dim": 12,
            "hidden_dim": 8,
            "classes": ["Pump", "Turbine", "Standstill"],
            "mean": np.zeros((12,), dtype=np.float32),
            "std": np.ones((12,), dtype=np.float32),
        },
        mode_artifact,
    )

    result = infer_baseline_model(
        artifact_path=artifacts.artifact_path,
        latent_paths=sorted(latent_dir.glob("*.npz")),
        device="cpu",
    )
    anomaly_events = build_anomaly_events(result, window_s=5.0, overlap=0.5)
    dataset = load_latent_dataset(sorted(latent_dir.glob("*.npz")))
    mode_payload = build_mode_predictions(
        artifact_path=mode_artifact,
        dataset=dataset,
        anomaly_events=anomaly_events,
        mode_consistency_window=3,
        device="cpu",
    )

    n_windows = int(result.scores.shape[0])
    assert mode_payload["mode_detection_enabled"] is True
    mode_labels = mode_payload["mode_labels"]
    mode_labels_smoothed = mode_payload["mode_labels_smoothed"]
    mode_confidence = mode_payload["mode_confidence"]
    mode_run_lengths = mode_payload["mode_run_lengths"]

    assert isinstance(mode_labels, list)
    assert isinstance(mode_labels_smoothed, list)
    assert isinstance(mode_confidence, list)
    assert isinstance(mode_run_lengths, list)
    assert len(mode_labels) == n_windows
    assert len(mode_labels_smoothed) == n_windows
    assert len(mode_confidence) == n_windows
    assert len(mode_run_lengths) == n_windows
