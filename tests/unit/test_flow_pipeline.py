from __future__ import annotations

import json
import numpy as np
import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.flow.infer import load_flow_artifact, score_with_context_smoothing
from src.modeling.flow.data import load_latent_dataset
from src.modeling.flow.train import FlowTrainingArtifacts, train_and_calibrate_flow


def _write_latent_file(
    path, *, recording_id: str, n: int = 32, d_model: int = 8, d_ctx: int = 4
) -> None:
    rng = np.random.default_rng(abs(hash((str(path), recording_id))) % (2**32))
    z = rng.normal(0.0, 1.0, size=(n, d_model)).astype(np.float32)
    c = rng.normal(0.0, 1.0, size=(n, d_ctx)).astype(np.float32)
    rid = np.asarray([recording_id] * n, dtype=str)
    is_transition = np.zeros(n, dtype=bool)
    is_transition[::7] = True

    np.savez_compressed(
        path,
        z=z,
        c=c,
        recording_id=rid,
        is_transition_window=is_transition,
    )


def test_flow_trainer_and_infer_roundtrip(tmp_path) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent_file(latent_dir / "pump_a.npz", recording_id="Pump")
    _write_latent_file(latent_dir / "turbine_a.npz", recording_id="Turbine")
    _write_latent_file(latent_dir / "fault_a.npz", recording_id="RandomFault")

    out_dir = tmp_path / "flow_artifacts"
    artifacts = train_and_calibrate_flow(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=out_dir,
        epochs=3,
        batch_size=16,
        n_coupling_layers=4,
        hidden_dim=32,
        val_ratio=0.25,
        score_percentile=95.0,
        device="cpu",
    )

    assert artifacts.artifact_path.exists()
    assert artifacts.threshold_path.exists()
    assert artifacts.summary_path.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert "threshold" in summary
    assert "healthy_val_score_mean" in summary
    assert "full_anomaly_rate" in summary
    assert float(summary["threshold"]) > 0.0
    assert 0.0 <= float(summary["full_anomaly_rate"]) <= 1.0

    flow, threshold = load_flow_artifact(artifacts.artifact_path, device="cpu")
    dataset = load_latent_dataset([latent_dir / "pump_a.npz"])

    result = score_with_context_smoothing(
        flow,
        dataset,
        threshold=threshold,
        smoother_k=8,
        smoother_decay=0.9,
        transition_policy="context_only",
        device="cpu",
    )

    assert result.scores.shape[0] == dataset.z.shape[0]
    assert result.flags.shape == result.scores.shape
    assert result.thresholds.shape == result.scores.shape
    assert np.all(np.isfinite(result.scores))


def test_flow_trainer_resume_from_checkpoint(tmp_path) -> None:
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()

    _write_latent_file(latent_dir / "pump_a.npz", recording_id="Pump")
    _write_latent_file(latent_dir / "turbine_a.npz", recording_id="Turbine")
    _write_latent_file(latent_dir / "fault_a.npz", recording_id="RandomFault")

    out_dir = tmp_path / "flow_artifacts"
    checkpoint_path = tmp_path / "checkpoints" / "flow_resume.pt"

    train_and_calibrate_flow(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=out_dir,
        epochs=2,
        pretrain_context_epochs=1,
        batch_size=16,
        n_coupling_layers=3,
        hidden_dim=24,
        val_ratio=0.25,
        score_percentile=95.0,
        device="cpu",
        quiet=True,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
    )

    assert checkpoint_path.exists()

    resumed = train_and_calibrate_flow(
        latent_paths=sorted(latent_dir.glob("*.npz")),
        output_dir=out_dir,
        epochs=2,
        pretrain_context_epochs=1,
        batch_size=16,
        n_coupling_layers=3,
        hidden_dim=24,
        val_ratio=0.25,
        score_percentile=95.0,
        device="cpu",
        quiet=True,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=True,
        checkpoint_every=1,
    )

    summary = json.loads(resumed.summary_path.read_text(encoding="utf-8"))
    checkpoint = summary["checkpoint"]
    assert checkpoint["resume_enabled"] is True
    assert int(checkpoint["pretrain_epoch_done"]) == 1
    assert int(checkpoint["flow_epoch_done"]) == 2
    assert int(summary["trained_epochs"]) == len(summary["history"])
    assert len(summary["history"]) >= 3
