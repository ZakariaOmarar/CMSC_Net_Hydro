from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.modeling.orchestration.train_all as train_all_module
from src.modeling.orchestration.train_all import _build_jobs, _build_report_job, main


def _write_minimal_latents(latent_dir: Path) -> None:
    latent_dir.mkdir(parents=True, exist_ok=True)
    z = np.zeros((8, 4), dtype=np.float32)
    c = np.zeros((8, 2), dtype=np.float32)
    rid = np.asarray(["Pump"] * 8, dtype=str)
    tr = np.zeros((8,), dtype=bool)
    np.savez_compressed(
        latent_dir / "sample.npz",
        z=z,
        c=c,
        recording_id=rid,
        is_transition_window=tr,
    )


def _write_minimal_configs(config_dir: Path, *, latent_root: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    latent = latent_root.as_posix()

    (config_dir / "anomaly_train.yaml").write_text(
        f"latent_root: {latent}\n",
        encoding="utf-8",
    )
    (config_dir / "anomaly_baseline_train.yaml").write_text(
        f"latent_root: {latent}\nmodel_type: ocsvm\n",
        encoding="utf-8",
    )
    (config_dir / "mode_train.yaml").write_text(
        f"latent_root: {latent}\n",
        encoding="utf-8",
    )
    (config_dir / "anomaly_infer_randomfault.yaml").write_text(
        (f"latent_root: {latent}\n" f"healthy_latent_root: {latent}\n"),
        encoding="utf-8",
    )
    (config_dir / "anomaly_baseline_infer_randomfault.yaml").write_text(
        f"latent_root: {latent}\n",
        encoding="utf-8",
    )


def test_build_jobs_uses_single_artifacts_root(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    latent_dir = tmp_path / "latents"
    artifacts_root = tmp_path / "results"
    _write_minimal_latents(latent_dir)
    _write_minimal_configs(config_dir, latent_root=latent_dir)

    jobs = _build_jobs(config_dir=config_dir, artifacts_root=artifacts_root)

    assert len(jobs) == 18
    assert all(callable(j.fn) for j in jobs)

    for job in jobs:
        if job.output_dir is None:
            continue
        output_dir = Path(job.output_dir)
        assert str(output_dir).startswith(str(artifacts_root))

    mode_jobs = [j for j in jobs if j.stage == "mode_train"]
    assert {j.name for j in mode_jobs} == {
        "CNF mode",
        "OC-SVM mode",
        "LSTM-AE mode",
        "CNN-AE mode",
    }


def test_build_jobs_anomaly_families_and_per_family_mode(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    latent_dir = tmp_path / "latents"
    artifacts_root = tmp_path / "results"
    _write_minimal_latents(latent_dir)
    _write_minimal_configs(config_dir, latent_root=latent_dir)

    jobs = _build_jobs(config_dir=config_dir, artifacts_root=artifacts_root)

    anomaly_jobs = [j for j in jobs if j.stage == "anomaly_train"]
    mode_jobs = [j for j in jobs if j.stage == "mode_train"]
    infer_jobs = [j for j in jobs if j.stage == "anomaly_infer"]

    expected_families = {"cnf", "ocsvm", "lstm_ae", "cnn_ae"}

    anomaly_families = {
        Path(j.output_dir).relative_to(artifacts_root).parts[0]
        for j in anomaly_jobs
        if j.output_dir is not None
    }
    assert len(mode_jobs) == 4
    assert len(infer_jobs) == 7
    assert anomaly_families.issuperset(expected_families)

    mode_by_family: dict[str, list[train_all_module.TrainJob]] = {
        fam: [] for fam in expected_families
    }
    for job in mode_jobs:
        if job.output_dir is None:
            continue
        fam = Path(job.output_dir).relative_to(artifacts_root).parts[0]
        if fam in mode_by_family:
            mode_by_family[fam].append(job)

    for family in expected_families:
        assert len(mode_by_family[family]) == 1
        assert "train_mode_classifier" in mode_by_family[family][0].description
        assert mode_by_family[family][0].output_dir == str(
            artifacts_root / family / "mode"
        )

    anomaly_by_family: dict[str, list[train_all_module.TrainJob]] = {
        fam: [] for fam in expected_families
    }
    for job in anomaly_jobs:
        if job.output_dir is None:
            continue
        fam = Path(job.output_dir).relative_to(artifacts_root).parts[0]
        if fam in anomaly_by_family:
            anomaly_by_family[fam].append(job)

    for family in expected_families:
        assert family in anomaly_by_family and anomaly_by_family[family]

        for anomaly_job in anomaly_by_family[family]:
            assert anomaly_job.output_dir is not None
            assert Path(anomaly_job.output_dir).parts[-1].startswith("anomaly")

        if family == "cnf":
            assert (
                "train_and_calibrate_flow" in anomaly_by_family[family][0].description
            )
        else:
            assert all(
                "train_baseline_model" in j.description
                for j in anomaly_by_family[family]
            )

    infer_by_family: dict[str, list[train_all_module.TrainJob]] = {
        fam: [] for fam in expected_families
    }
    for job in infer_jobs:
        if job.output_dir is None:
            continue
        fam = Path(job.output_dir).relative_to(artifacts_root).parts[0]
        if fam in infer_by_family:
            infer_by_family[fam].append(job)

    for family in expected_families:
        assert infer_by_family[family]
        if family == "cnf":
            assert all("flow_infer" in j.description for j in infer_by_family[family])
        else:
            assert all(
                "baseline_infer" in j.description for j in infer_by_family[family]
            )


def test_build_report_job_constructs_callable(tmp_path) -> None:
    artifacts_root = tmp_path / "results"
    report_output = artifacts_root / "reports" / "model_report.json"
    manifest_path = artifacts_root / "reports" / "train_all_manifest.json"

    job = _build_report_job(
        artifacts_root=artifacts_root,
        report_output=report_output,
        manifest_path=manifest_path,
    )

    assert job.stage == "reporting"
    assert callable(job.fn)
    assert "collect_model_reports" in job.description
    assert job.output_dir == str(report_output.parent)


def test_main_dry_run_writes_manifest(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    latent_dir = tmp_path / "latents"
    artifacts_root = tmp_path / "results"
    _write_minimal_latents(latent_dir)
    _write_minimal_configs(config_dir, latent_root=latent_dir)

    rc = main(
        [
            "--config-dir",
            str(config_dir),
            "--artifacts-root",
            str(artifacts_root),
            "--dry-run",
        ]
    )

    assert rc == 0

    manifest_path = artifacts_root / "reports" / "train_all_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "dry_run"
    assert payload["n_failed_jobs"] == 0
    assert payload["n_jobs"] == 19
    assert payload["artifacts_root"] == str(artifacts_root)
    assert payload["report_output"] == str(
        artifacts_root / "reports" / "model_report.json"
    )
    assert "stage_summary" in payload
    assert "anomaly_train" in payload["stage_summary"]
    assert "mode_train" in payload["stage_summary"]
    assert "anomaly_infer" in payload["stage_summary"]
    assert "reporting" in payload["stage_summary"]
    assert "runtime" in payload
    assert "orchestrator_peak_memory_mb" in payload["runtime"]

    statuses = [job["status"] for job in payload["jobs"]]
    assert statuses
    assert set(statuses) == {"dry_run"}
    assert payload["stage_summary"]["mode_train"]["n_jobs"] == 4
    mode_job_names = [
        job["name"] for job in payload["jobs"] if job.get("stage") == "mode_train"
    ]
    assert mode_job_names == ["CNF mode", "OC-SVM mode", "LSTM-AE mode", "CNN-AE mode"]


def test_main_resume_skips_successful_jobs(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    latent_dir = tmp_path / "latents"
    artifacts_root = tmp_path / "results"
    _write_minimal_latents(latent_dir)
    _write_minimal_configs(config_dir, latent_root=latent_dir)

    resume_manifest = tmp_path / "previous_manifest.json"
    resume_manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "CNF anomaly",
                        "status": "ok",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--config-dir",
            str(config_dir),
            "--artifacts-root",
            str(artifacts_root),
            "--dry-run",
            "--resume-from-manifest",
            str(resume_manifest),
        ]
    )

    assert rc == 0

    manifest_path = artifacts_root / "reports" / "train_all_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["resume_skip_ok"] is True
    assert payload["n_skipped_jobs"] >= 1

    statuses = [job["status"] for job in payload["jobs"]]
    assert "skipped" in statuses


def test_run_job_retries_until_success() -> None:
    calls = {"count": 0}

    def _flaky() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")

    result = train_all_module._run_job(
        job=train_all_module.TrainJob(name="job", fn=_flaky, description="test"),
        dry_run=False,
        max_retries=2,
        retry_backoff_s=0.0,
    )

    assert result.status == "ok"
    assert result.attempt_count == 2
    assert result.retried is True


def test_find_repo_root_resolves_by_markers(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "modeling" / "orchestration"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    resolved = train_all_module._find_repo_root(nested / "train_all.py")
    assert resolved == repo_root


def test_find_repo_root_raises_when_markers_missing(tmp_path) -> None:
    probe = tmp_path / "a" / "b" / "c"
    probe.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Unable to locate project root"):
        train_all_module._find_repo_root(probe / "train_all.py")
