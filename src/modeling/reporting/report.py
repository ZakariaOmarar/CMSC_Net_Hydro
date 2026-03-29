"""Compile a comparison table across all anomaly model families.

After a full training run, this module reads the checkpoint paths, training
summaries, and run manifest to construct one ModelReportRow per model family
(cnf, ocsvm, lstm_ae, cnn_ae). The rows capture both anomaly-detection and
mode-classification metrics side-by-side, enabling a single model_report.json
that can drive figures and tables in the thesis without further post-processing.

The preferred input is the train_all_manifest.json produced by train_all.py,
which provides deterministic job-to-path mapping. When no manifest is available,
the module falls back to scanning the artifacts directory by convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelReportRow:
    """One row in the model comparison table, covering one anomaly model family.

    Anomaly fields (anomaly_*) come from the model's training summary and
    threshold file. Mode fields (mode_*) come from the mode classifier trained
    alongside that anomaly model. Fields are None when the corresponding
    artifact was not found.
    """

    model_name: str
    anomaly_checkpoint_path: str | None
    anomaly_summary_path: str | None
    anomaly_threshold: float | None
    anomaly_score_percentile: float | None
    anomaly_healthy_val_score_mean: float | None
    anomaly_healthy_val_score_std: float | None
    anomaly_full_score_mean: float | None
    anomaly_full_score_std: float | None
    anomaly_n_full_anomalies: int | None
    anomaly_full_anomaly_rate: float | None
    anomaly_healthy_fpr: float | None
    anomaly_rf_detection_rate: float | None
    mode_checkpoint_path: str | None
    mode_summary_path: str | None
    mode_validation_accuracy: float | None
    mode_validation_macro_f1: float | None
    mode_cv_accuracy_std: float | None
    mode_cv_macro_f1_std: float | None
    mode_cv_folds: int | None
    mode_test_accuracy: float | None
    mode_test_macro_f1: float | None
    mode_checkpoint_found: bool
    mode_score_for_ranking: float
    mode_ranking_metric: str
    anomaly_checkpoint_found: bool


_EXPECTED_MODELS = ("cnf", "ocsvm", "lstm_ae", "cnn_ae")
_ANOMALY_JOB_BY_MODEL = {
    "cnf": "CNF anomaly",
    "ocsvm": "OC-SVM anomaly",
    "lstm_ae": "LSTM-AE anomaly",
    "cnn_ae": "CNN-AE anomaly",
}
_MODE_JOB_BY_MODEL = {
    "cnf": "CNF mode",
    "ocsvm": "OC-SVM mode",
    "lstm_ae": "LSTM-AE mode",
    "cnn_ae": "CNN-AE mode",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return raw


def _read_manifest_jobs(manifest_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(Path(manifest_path))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"Manifest {manifest_path} must contain a jobs list")

    normalized: list[dict[str, Any]] = []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized.append(item)
    return normalized


def _job_lookup(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for job in jobs:
        name_raw = job.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            continue
        out[name_raw] = job
    return out


def _summary_tuple_from_manifest(
    *,
    job_lookup: dict[str, dict[str, Any]],
    job_name: str,
    summary_name: str,
    checkpoint_name: str,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    job = job_lookup.get(job_name)
    if job is None:
        return None, None, None

    output_dir_raw = job.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw.strip():
        return None, None, None

    output_dir = Path(output_dir_raw)
    summary_path = output_dir / summary_name
    checkpoint_path = output_dir / checkpoint_name

    if summary_path.exists() and summary_path.is_file():
        return summary_path, checkpoint_path, _read_json(summary_path)

    return summary_path, checkpoint_path, None


def _resolve_mode_summary_from_manifest(
    *,
    job_lookup: dict[str, dict[str, Any]],
    model_name: str,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    return _summary_tuple_from_manifest(
        job_lookup=job_lookup,
        job_name=_MODE_JOB_BY_MODEL[model_name],
        summary_name="mode_training_summary.json",
        checkpoint_name="mode_classifier.pt",
    )


def _collect_model_reports_from_manifest(
    *, manifest_path: Path
) -> list[ModelReportRow]:
    jobs = _read_manifest_jobs(Path(manifest_path))
    lookup = _job_lookup(jobs)

    rows: list[ModelReportRow] = []
    for model_name in _EXPECTED_MODELS:
        anomaly_summary_path: Path | None
        anomaly_ckpt_path: Path | None
        anomaly_summary: dict[str, Any] | None

        mode_summary_path: Path | None
        mode_ckpt_path: Path | None
        mode_summary: dict[str, Any] | None

        if model_name == "cnf":
            anomaly_summary_path, anomaly_ckpt_path, anomaly_summary = (
                _summary_tuple_from_manifest(
                    job_lookup=lookup,
                    job_name=_ANOMALY_JOB_BY_MODEL[model_name],
                    summary_name="training_summary.json",
                    checkpoint_name="flow.pt",
                )
            )
        else:
            anomaly_summary_path, anomaly_ckpt_path, anomaly_summary = (
                _summary_tuple_from_manifest(
                    job_lookup=lookup,
                    job_name=_ANOMALY_JOB_BY_MODEL[model_name],
                    summary_name="anomaly_model_summary.json",
                    checkpoint_name="anomaly_model.pkl",
                )
            )

        mode_summary_path, mode_ckpt_path, mode_summary = (
            _resolve_mode_summary_from_manifest(
                job_lookup=lookup,
                model_name=model_name,
            )
        )

        rows.append(
            _build_row(
                model_name=model_name,
                anomaly_checkpoint_path=anomaly_ckpt_path,
                anomaly_summary_path=anomaly_summary_path,
                anomaly_summary=anomaly_summary,
                mode_checkpoint_path=mode_ckpt_path,
                mode_summary_path=mode_summary_path,
                mode_summary=mode_summary,
            )
        )

    rows.sort(
        key=lambda r: (
            1 if r.anomaly_checkpoint_found else 0,
            1 if r.mode_checkpoint_found else 0,
            r.mode_score_for_ranking,
            r.mode_test_accuracy or -1.0,
        ),
        reverse=True,
    )
    return rows


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating, str)):
        return float(value)
    return None


def _ranking_score(
    *,
    cv_macro_f1_mean: float | None,
    test_macro_f1: float | None,
    test_accuracy: float | None,
    validation_macro_f1: float | None,
    validation_accuracy: float | None,
) -> tuple[float, str]:
    if cv_macro_f1_mean is not None:
        return float(cv_macro_f1_mean), "cv_macro_f1_mean"
    if test_macro_f1 is not None:
        return float(test_macro_f1), "test_macro_f1"
    if test_accuracy is not None:
        return float(test_accuracy), "test_accuracy"
    if validation_macro_f1 is not None:
        return float(validation_macro_f1), "validation_macro_f1"
    if validation_accuracy is not None:
        return float(validation_accuracy), "validation_accuracy"
    return float("-inf"), "none"


def _build_row(
    *,
    model_name: str,
    anomaly_checkpoint_path: Path | None,
    anomaly_summary_path: Path | None,
    anomaly_summary: dict[str, Any] | None,
    mode_checkpoint_path: Path | None,
    mode_summary_path: Path | None,
    mode_summary: dict[str, Any] | None,
) -> ModelReportRow:
    anomaly_summary = anomaly_summary or {}
    mode_summary = mode_summary or {}

    mode_val_acc = _as_float_or_none(mode_summary.get("validation_accuracy"))
    if mode_val_acc is None:
        mode_val_acc = _as_float_or_none(mode_summary.get("best_val_accuracy"))

    mode_val_f1 = _as_float_or_none(mode_summary.get("validation_macro_f1"))
    if mode_val_f1 is None:
        mode_val_f1 = _as_float_or_none(mode_summary.get("best_val_macro_f1"))

    cv = mode_summary.get("cv")
    cv_macro_f1_mean: float | None = None
    cv_macro_f1_std: float | None = None
    cv_accuracy_std: float | None = None
    cv_folds: int | None = None
    if isinstance(cv, dict):
        cv_macro_f1_mean = _as_float_or_none(cv.get("macro_f1_mean"))
        cv_macro_f1_std = _as_float_or_none(cv.get("macro_f1_std"))
        cv_accuracy_std = _as_float_or_none(cv.get("accuracy_std"))
        cv_folds_raw = cv.get("n_folds")
        if isinstance(cv_folds_raw, (int, float, np.integer, np.floating, str)):
            cv_folds = int(cv_folds_raw)

    mode_test_acc = _as_float_or_none(mode_summary.get("test_accuracy"))
    mode_test_f1 = _as_float_or_none(mode_summary.get("test_macro_f1"))
    score, metric = _ranking_score(
        cv_macro_f1_mean=cv_macro_f1_mean,
        test_macro_f1=mode_test_f1,
        test_accuracy=mode_test_acc,
        validation_macro_f1=mode_val_f1,
        validation_accuracy=mode_val_acc,
    )

    n_full_anomalies_raw = anomaly_summary.get("n_full_anomalies")
    n_full_anomalies = (
        int(n_full_anomalies_raw)
        if isinstance(n_full_anomalies_raw, (int, float, np.integer, np.floating, str))
        else None
    )

    return ModelReportRow(
        model_name=model_name,
        anomaly_checkpoint_path=(
            str(anomaly_checkpoint_path)
            if anomaly_checkpoint_path is not None
            else None
        ),
        anomaly_summary_path=(
            str(anomaly_summary_path) if anomaly_summary_path is not None else None
        ),
        anomaly_threshold=_as_float_or_none(anomaly_summary.get("threshold")),
        anomaly_score_percentile=_as_float_or_none(
            anomaly_summary.get("score_percentile")
        ),
        anomaly_healthy_val_score_mean=_as_float_or_none(
            anomaly_summary.get("healthy_val_score_mean")
        ),
        anomaly_healthy_val_score_std=_as_float_or_none(
            anomaly_summary.get("healthy_val_score_std")
        ),
        anomaly_full_score_mean=_as_float_or_none(
            anomaly_summary.get("full_score_mean")
        ),
        anomaly_full_score_std=_as_float_or_none(anomaly_summary.get("full_score_std")),
        anomaly_n_full_anomalies=n_full_anomalies,
        anomaly_full_anomaly_rate=_as_float_or_none(
            anomaly_summary.get("full_anomaly_rate")
        ),
        anomaly_healthy_fpr=_as_float_or_none(anomaly_summary.get("healthy_fpr")),
        anomaly_rf_detection_rate=_as_float_or_none(
            anomaly_summary.get("rf_detection_rate")
        ),
        mode_checkpoint_path=(
            str(mode_checkpoint_path) if mode_checkpoint_path is not None else None
        ),
        mode_summary_path=(
            str(mode_summary_path) if mode_summary_path is not None else None
        ),
        mode_validation_accuracy=mode_val_acc,
        mode_validation_macro_f1=mode_val_f1,
        mode_cv_accuracy_std=cv_accuracy_std,
        mode_cv_macro_f1_std=cv_macro_f1_std,
        mode_cv_folds=cv_folds,
        mode_test_accuracy=mode_test_acc,
        mode_test_macro_f1=mode_test_f1,
        mode_checkpoint_found=bool(
            mode_checkpoint_path is not None and mode_checkpoint_path.exists()
        ),
        mode_score_for_ranking=float(score),
        mode_ranking_metric=metric,
        anomaly_checkpoint_found=bool(
            anomaly_checkpoint_path is not None and anomaly_checkpoint_path.exists()
        ),
    )


def _collect_baseline_models(
    *, artifacts_root: Path
) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    out: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    model_names = ("ocsvm", "lstm_ae", "cnn_ae")
    for model_name in model_names:
        candidates: list[Path] = []

        modern_dir = artifacts_root / model_name / "anomaly"
        preferred_modern = modern_dir / "anomaly_summary.json"
        if preferred_modern.exists() and preferred_modern.is_file():
            candidates.append(preferred_modern)
        if modern_dir.exists() and modern_dir.is_dir():
            for path in sorted(modern_dir.glob("*_summary.json")):
                if path not in candidates:
                    candidates.append(path)

        for summary_path in candidates:
            summary = _read_json(summary_path)
            summary_model = str(summary.get("model_type", model_name)).strip().lower()
            if summary_model != model_name:
                continue

            ckpt_raw = summary.get("artifact_path")
            if summary_path.parent.name == "anomaly":
                local_checkpoint = summary_path.parent / "anomaly_model.pkl"
                if local_checkpoint.exists() and local_checkpoint.is_file():
                    checkpoint_path = local_checkpoint
                elif ckpt_raw is not None:
                    checkpoint_path = Path(str(ckpt_raw))
                else:
                    checkpoint_path = local_checkpoint
            elif ckpt_raw is not None:
                checkpoint_path = Path(str(ckpt_raw))
            else:
                checkpoint_path = summary_path.with_name(
                    summary_path.stem.replace("_summary", "") + ".pkl"
                )

            out[model_name] = (summary_path, checkpoint_path, summary)
            break

    return out


def _resolve_cnf_anomaly_artifacts(
    *, artifacts_root: Path
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    summary_path = artifacts_root / "cnf" / "anomaly" / "training_summary.json"
    checkpoint_path = artifacts_root / "cnf" / "anomaly" / "flow.pt"
    if summary_path.exists() and summary_path.is_file():
        return summary_path, checkpoint_path, _read_json(summary_path)
    return None, None, None


def _resolve_mode_artifacts(
    *,
    artifacts_root: Path,
    model_name: str,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    candidate_summaries = [
        artifacts_root / model_name / "mode" / "mode_training_summary.json",
    ]
    for summary_path in candidate_summaries:
        if summary_path.exists() and summary_path.is_file():
            mode_summary = _read_json(summary_path)
            mode_ckpt = summary_path.parent / "mode_classifier.pt"
            return summary_path, mode_ckpt, mode_summary
    return None, None, None


def collect_model_reports(
    *,
    artifacts_root: Path,
    manifest_path: Path | None = None,
) -> list[ModelReportRow]:
    if manifest_path is not None:
        p = Path(manifest_path)
        if p.exists() and p.is_file():
            return _collect_model_reports_from_manifest(manifest_path=p)

    root = Path(artifacts_root)
    rows: list[ModelReportRow] = []
    baseline_models = _collect_baseline_models(artifacts_root=root)

    for model_name in _EXPECTED_MODELS:
        anomaly_summary_path: Path | None = None
        anomaly_ckpt_path: Path | None = None
        anomaly_summary: dict[str, Any] | None = None

        if model_name == "cnf":
            anomaly_summary_path, anomaly_ckpt_path, anomaly_summary = (
                _resolve_cnf_anomaly_artifacts(artifacts_root=root)
            )
        else:
            found = baseline_models.get(model_name)
            if found is not None:
                anomaly_summary_path, anomaly_ckpt_path, anomaly_summary = found

        mode_summary_path, mode_ckpt_path, mode_summary = _resolve_mode_artifacts(
            artifacts_root=root,
            model_name=model_name,
        )

        rows.append(
            _build_row(
                model_name=model_name,
                anomaly_checkpoint_path=anomaly_ckpt_path,
                anomaly_summary_path=anomaly_summary_path,
                anomaly_summary=anomaly_summary,
                mode_checkpoint_path=mode_ckpt_path,
                mode_summary_path=mode_summary_path,
                mode_summary=mode_summary,
            )
        )

    rows.sort(
        key=lambda r: (
            1 if r.anomaly_checkpoint_found else 0,
            1 if r.mode_checkpoint_found else 0,
            r.mode_score_for_ranking,
            r.mode_test_accuracy or -1.0,
        ),
        reverse=True,
    )
    return rows


def to_payload(rows: list[ModelReportRow]) -> dict[str, Any]:
    n_trained = int(sum(1 for r in rows if r.anomaly_checkpoint_found))
    return {
        "n_models": int(len(rows)),
        "n_trained_models": n_trained,
        "models": [
            {
                "rank": int(i + 1),
                "model_name": r.model_name,
                "anomaly_checkpoint_path": r.anomaly_checkpoint_path,
                "anomaly_summary_path": r.anomaly_summary_path,
                "anomaly_threshold": r.anomaly_threshold,
                "anomaly_score_percentile": r.anomaly_score_percentile,
                "anomaly_healthy_val_score_mean": r.anomaly_healthy_val_score_mean,
                "anomaly_healthy_val_score_std": r.anomaly_healthy_val_score_std,
                "anomaly_full_score_mean": r.anomaly_full_score_mean,
                "anomaly_full_score_std": r.anomaly_full_score_std,
                "anomaly_n_full_anomalies": r.anomaly_n_full_anomalies,
                "anomaly_full_anomaly_rate": r.anomaly_full_anomaly_rate,
                "anomaly_healthy_fpr": r.anomaly_healthy_fpr,
                "anomaly_rf_detection_rate": r.anomaly_rf_detection_rate,
                "mode_checkpoint_path": r.mode_checkpoint_path,
                "mode_summary_path": r.mode_summary_path,
                "mode_validation_accuracy": r.mode_validation_accuracy,
                "mode_validation_macro_f1": r.mode_validation_macro_f1,
                "mode_cv_accuracy_std": r.mode_cv_accuracy_std,
                "mode_cv_macro_f1_std": r.mode_cv_macro_f1_std,
                "mode_cv_folds": r.mode_cv_folds,
                "mode_test_accuracy": r.mode_test_accuracy,
                "mode_test_macro_f1": r.mode_test_macro_f1,
                "mode_checkpoint_found": r.mode_checkpoint_found,
                "mode_score_for_ranking": r.mode_score_for_ranking,
                "mode_ranking_metric": r.mode_ranking_metric,
                "anomaly_checkpoint_found": r.anomaly_checkpoint_found,
            }
            for i, r in enumerate(rows)
        ],
    }
