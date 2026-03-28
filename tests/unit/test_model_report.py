from __future__ import annotations

import json
from pathlib import Path

from src.modeling.reporting.report import collect_model_reports, to_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_collect_model_reports_aggregates_and_ranks(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"

    _write_json(
        artifacts / "cnf" / "anomaly" / "training_summary.json",
        {
            "threshold": 123.0,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 80.0,
            "healthy_val_score_std": 5.0,
            "full_score_mean": 90.0,
            "full_score_std": 10.0,
            "n_full_anomalies": 12,
            "full_anomaly_rate": 0.12,
        },
    )
    _write_json(
        artifacts / "cnf" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.80,
            "validation_macro_f1": 0.78,
            "test_accuracy": 0.82,
            "test_macro_f1": 0.79,
        },
    )
    _write_json(
        artifacts / "ocsvm" / "anomaly" / "anomaly_summary.json",
        {
            "model_type": "ocsvm",
            "artifact_path": str(artifacts / "ocsvm" / "anomaly" / "anomaly_model.pkl"),
            "threshold": 5.5,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 2.1,
            "healthy_val_score_std": 0.8,
            "full_score_mean": 3.5,
            "full_score_std": 1.4,
            "n_full_anomalies": 21,
            "full_anomaly_rate": 0.21,
        },
    )
    _write_json(
        artifacts / "ocsvm" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.72,
            "validation_macro_f1": 0.70,
            "test_accuracy": 0.74,
            "test_macro_f1": 0.71,
        },
    )

    rows = collect_model_reports(artifacts_root=artifacts)

    assert len(rows) == 4
    names = [r.model_name for r in rows]
    assert set(names) == {"cnf", "ocsvm", "lstm_ae", "cnn_ae"}

    cnf_row = next(r for r in rows if r.model_name == "cnf")
    assert cnf_row.mode_ranking_metric == "test_macro_f1"
    assert cnf_row.anomaly_threshold == 123.0
    assert cnf_row.anomaly_n_full_anomalies == 12
    assert cnf_row.mode_test_macro_f1 == 0.79

    ocsvm_row = next(r for r in rows if r.model_name == "ocsvm")
    assert ocsvm_row.anomaly_threshold == 5.5
    assert ocsvm_row.anomaly_full_anomaly_rate == 0.21
    assert ocsvm_row.mode_test_accuracy == 0.74
    assert ocsvm_row.mode_checkpoint_found is False

    lstm_row = next(r for r in rows if r.model_name == "lstm_ae")
    assert lstm_row.anomaly_checkpoint_found is False
    assert lstm_row.anomaly_threshold is None
    assert lstm_row.mode_test_accuracy is None


def test_collect_model_reports_prefers_manifest_when_provided(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"

    _write_json(
        artifacts / "cnf" / "anomaly" / "training_summary.json",
        {
            "threshold": 111.0,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 10.0,
            "healthy_val_score_std": 2.0,
            "full_score_mean": 12.0,
            "full_score_std": 3.0,
            "n_full_anomalies": 7,
            "full_anomaly_rate": 0.07,
        },
    )
    _write_json(
        artifacts / "flow" / "training_summary.json",
        {
            "threshold": 999.0,
        },
    )

    _write_json(
        artifacts / "cnf" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.88,
            "validation_macro_f1": 0.86,
            "test_accuracy": 0.87,
            "test_macro_f1": 0.85,
        },
    )

    manifest_path = artifacts / "reports" / "train_all_manifest.json"
    _write_json(
        manifest_path,
        {
            "jobs": [
                {
                    "name": "CNF anomaly",
                    "status": "ok",
                    "output_dir": str(artifacts / "cnf" / "anomaly"),
                },
                {
                    "name": "CNF mode",
                    "status": "ok",
                    "output_dir": str(artifacts / "cnf" / "mode"),
                },
            ]
        },
    )

    rows = collect_model_reports(
        artifacts_root=artifacts,
        manifest_path=manifest_path,
    )

    cnf_row = next(r for r in rows if r.model_name == "cnf")
    assert cnf_row.anomaly_threshold == 111.0
    assert cnf_row.anomaly_summary_path is not None
    assert "cnf" in cnf_row.anomaly_summary_path.replace("\\", "/")
    assert cnf_row.mode_summary_path is not None
    assert "cnf/mode" in cnf_row.mode_summary_path.replace("\\", "/")


def test_collect_model_reports_uses_family_mode_fallback_without_manifest(
    tmp_path,
) -> None:
    artifacts = tmp_path / "artifacts"

    _write_json(
        artifacts / "cnf" / "anomaly" / "training_summary.json",
        {
            "threshold": 123.0,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 80.0,
            "healthy_val_score_std": 5.0,
            "full_score_mean": 90.0,
            "full_score_std": 10.0,
            "n_full_anomalies": 12,
            "full_anomaly_rate": 0.12,
        },
    )
    _write_json(
        artifacts / "cnf" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.81,
            "validation_macro_f1": 0.80,
            "test_accuracy": 0.83,
            "test_macro_f1": 0.82,
        },
    )
    _write_json(
        artifacts / "ocsvm" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.71,
            "validation_macro_f1": 0.70,
            "test_accuracy": 0.72,
            "test_macro_f1": 0.71,
        },
    )

    rows = collect_model_reports(artifacts_root=artifacts)

    cnf_row = next(r for r in rows if r.model_name == "cnf")
    assert cnf_row.mode_summary_path is not None
    assert "cnf/mode" in cnf_row.mode_summary_path.replace("\\", "/")
    assert cnf_row.mode_test_macro_f1 == 0.82

    ocsvm_row = next(r for r in rows if r.model_name == "ocsvm")
    assert ocsvm_row.mode_summary_path is not None
    assert "ocsvm/mode" in ocsvm_row.mode_summary_path.replace("\\", "/")


def test_collect_model_reports_uses_family_mode_from_manifest(
    tmp_path,
) -> None:
    artifacts = tmp_path / "artifacts"

    _write_json(
        artifacts / "cnf" / "anomaly" / "training_summary.json",
        {
            "threshold": 111.0,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 10.0,
            "healthy_val_score_std": 2.0,
            "full_score_mean": 12.0,
            "full_score_std": 3.0,
            "n_full_anomalies": 7,
            "full_anomaly_rate": 0.07,
        },
    )
    _write_json(
        artifacts / "cnf" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.88,
            "validation_macro_f1": 0.86,
            "test_accuracy": 0.87,
            "test_macro_f1": 0.85,
        },
    )

    manifest_path = artifacts / "reports" / "train_all_manifest.json"
    _write_json(
        manifest_path,
        {
            "jobs": [
                {
                    "name": "CNF anomaly",
                    "status": "ok",
                    "output_dir": str(artifacts / "cnf" / "anomaly"),
                },
                {
                    "name": "CNF mode",
                    "status": "ok",
                    "output_dir": str(artifacts / "cnf" / "mode"),
                },
            ]
        },
    )

    rows = collect_model_reports(
        artifacts_root=artifacts,
        manifest_path=manifest_path,
    )

    cnf_row = next(r for r in rows if r.model_name == "cnf")
    assert cnf_row.mode_summary_path is not None
    assert "cnf/mode" in cnf_row.mode_summary_path.replace("\\", "/")
    assert cnf_row.mode_test_macro_f1 == 0.85

    ocsvm_row = next(r for r in rows if r.model_name == "ocsvm")
    assert ocsvm_row.mode_summary_path is None


def test_to_payload_keeps_mode_metrics_per_model(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"

    _write_json(
        artifacts / "cnf" / "anomaly" / "training_summary.json",
        {
            "threshold": 111.0,
            "score_percentile": 99.0,
            "healthy_val_score_mean": 10.0,
            "healthy_val_score_std": 2.0,
            "full_score_mean": 12.0,
            "full_score_std": 3.0,
            "n_full_anomalies": 7,
            "full_anomaly_rate": 0.07,
        },
    )
    _write_json(
        artifacts / "cnf" / "mode" / "mode_training_summary.json",
        {
            "validation_accuracy": 0.88,
            "validation_macro_f1": 0.86,
            "test_accuracy": 0.87,
            "test_macro_f1": 0.85,
        },
    )

    rows = collect_model_reports(artifacts_root=artifacts)
    payload = to_payload(rows)

    assert "shared_mode" not in payload

    cnf_row = next(r for r in payload["models"] if r["model_name"] == "cnf")
    assert cnf_row["mode_test_accuracy"] == 0.87
    assert cnf_row["mode_ranking_metric"] == "test_macro_f1"
