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
    anomaly_threshold: float | None
    anomaly_score_percentile: float | None
    anomaly_healthy_val_score_mean: float | None
    anomaly_healthy_val_score_std: float | None
    anomaly_n_full_anomalies: int | None
    anomaly_healthy_fpr: float | None
    fault_n_windows_total: int | None
    fault_n_detected: int | None
    fault_detection_rate: float | None
    mode_test_accuracy: float | None
    mode_test_macro_f1: float | None
    mode_score_for_ranking: float


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
    *,
    manifest_path: Path,
    fault_positions_root: Path | None = None,
) -> list[ModelReportRow]:
    jobs = _read_manifest_jobs(Path(manifest_path))
    lookup = _job_lookup(jobs)

    # Derive artifacts_root from the manifest location for flat-layout fallback
    manifest_dir = Path(manifest_path).parent
    artifacts_root_fallback = manifest_dir.parent  # reports/ -> artifacts_root/

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

        if fault_positions_root is not None:
            fn_total, fn_det, fn_rate = _collect_fault_stats_nested(
                Path(fault_positions_root), model_name
            )
        else:
            fn_total, fn_det, fn_rate = _collect_fault_stats_flat(
                artifacts_root_fallback, model_name
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
                fault_n_windows_total=fn_total,
                fault_n_detected=fn_det,
                fault_detection_rate=fn_rate,
            )
        )

    rows.sort(
        key=lambda r: (
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
    test_macro_f1: float | None,
    test_accuracy: float | None,
) -> float:
    if test_macro_f1 is not None:
        return float(test_macro_f1)
    if test_accuracy is not None:
        return float(test_accuracy)
    return float("-inf")


def _aggregate_fault_stats(
    infer_paths: list[Path],
) -> tuple[int | None, int | None, float | None]:
    """Sum n_windows and n_anomalies across a list of inference result files.

    Returns (total_windows, total_detected, detection_rate) or (None, None, None)
    if no readable files are found.
    """
    total_windows = 0
    total_detected = 0
    found_any = False
    for p in infer_paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_win = blob.get("n_windows")
        n_anom = blob.get("n_anomalies")
        if isinstance(n_win, (int, float)) and isinstance(n_anom, (int, float)):
            total_windows += int(n_win)
            total_detected += int(n_anom)
            found_any = True
    if not found_any:
        return None, None, None
    rate = float(total_detected) / float(total_windows) if total_windows > 0 else 0.0
    return total_windows, total_detected, rate


# Maps model family -> infer file names to look for in each fault position dir
_FAULT_INFER_FILES: dict[str, list[str]] = {
    "cnf": ["cnf_infer.json"],
    "ocsvm": ["ocsvm_anomaly_infer.json", "infer_randomfault.json"],
    "lstm_ae": ["lstm_ae_anomaly_infer.json", "infer_randomfault.json"],
    "cnn_ae": ["cnn_ae_anomaly_infer.json", "infer_randomfault.json"],
}


def _collect_fault_stats_nested(
    fault_positions_root: Path, model_name: str
) -> tuple[int | None, int | None, float | None]:
    """Aggregate fault detection stats from per-position subdirectories."""
    candidates = _FAULT_INFER_FILES.get(model_name, [])
    paths: list[Path] = []
    if fault_positions_root.is_dir():
        for pos_dir in sorted(fault_positions_root.iterdir()):
            if not pos_dir.is_dir():
                continue
            for name in candidates:
                p = pos_dir / name
                if p.exists():
                    paths.append(p)
                    break
    return _aggregate_fault_stats(paths)


def _collect_fault_stats_flat(
    artifacts_root: Path, model_name: str
) -> tuple[int | None, int | None, float | None]:
    """Aggregate fault detection stats from flat anomaly dirs (first dataset)."""
    candidates = [
        "randomfault_infer_with_timestamps.json",
        "infer_randomfault.json",
    ]
    paths: list[Path] = []
    anomaly_dirs = sorted((artifacts_root / model_name).glob("anomaly*"))
    for anomaly_dir in anomaly_dirs:
        for name in candidates:
            p = anomaly_dir / name
            if p.exists():
                paths.append(p)
                break
    return _aggregate_fault_stats(paths)


def _build_row(
    *,
    model_name: str,
    anomaly_checkpoint_path: Path | None,
    anomaly_summary_path: Path | None,
    anomaly_summary: dict[str, Any] | None,
    mode_checkpoint_path: Path | None,
    mode_summary_path: Path | None,
    mode_summary: dict[str, Any] | None,
    fault_n_windows_total: int | None = None,
    fault_n_detected: int | None = None,
    fault_detection_rate: float | None = None,
) -> ModelReportRow:
    anomaly_summary = anomaly_summary or {}
    mode_summary = mode_summary or {}

    mode_test_acc = _as_float_or_none(mode_summary.get("test_accuracy"))
    mode_test_f1 = _as_float_or_none(mode_summary.get("test_macro_f1"))
    score = _ranking_score(
        test_macro_f1=mode_test_f1,
        test_accuracy=mode_test_acc,
    )

    n_full_anomalies_raw = anomaly_summary.get("n_full_anomalies")
    n_full_anomalies = (
        int(n_full_anomalies_raw)
        if isinstance(n_full_anomalies_raw, (int, float, np.integer, np.floating, str))
        else None
    )

    return ModelReportRow(
        model_name=model_name,
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
        anomaly_n_full_anomalies=n_full_anomalies,
        anomaly_healthy_fpr=_as_float_or_none(anomaly_summary.get("healthy_fpr")),
        fault_n_windows_total=fault_n_windows_total,
        fault_n_detected=fault_n_detected,
        fault_detection_rate=fault_detection_rate,
        mode_test_accuracy=mode_test_acc,
        mode_test_macro_f1=mode_test_f1,
        mode_score_for_ranking=float(score),
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
    fault_positions_root: Path | None = None,
) -> list[ModelReportRow]:
    if manifest_path is not None:
        p = Path(manifest_path)
        if p.exists() and p.is_file():
            return _collect_model_reports_from_manifest(
                manifest_path=p,
                fault_positions_root=fault_positions_root,
            )

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

        if fault_positions_root is not None:
            fn_total, fn_det, fn_rate = _collect_fault_stats_nested(
                Path(fault_positions_root), model_name
            )
        else:
            fn_total, fn_det, fn_rate = _collect_fault_stats_flat(root, model_name)

        rows.append(
            _build_row(
                model_name=model_name,
                anomaly_checkpoint_path=anomaly_ckpt_path,
                anomaly_summary_path=anomaly_summary_path,
                anomaly_summary=anomaly_summary,
                mode_checkpoint_path=mode_ckpt_path,
                mode_summary_path=mode_summary_path,
                mode_summary=mode_summary,
                fault_n_windows_total=fn_total,
                fault_n_detected=fn_det,
                fault_detection_rate=fn_rate,
            )
        )

    rows.sort(
        key=lambda r: (
            r.mode_score_for_ranking,
            r.mode_test_accuracy or -1.0,
        ),
        reverse=True,
    )
    return rows


def to_payload(rows: list[ModelReportRow]) -> dict[str, Any]:
    n_trained = int(sum(1 for r in rows if r.anomaly_threshold is not None))
    return {
        "n_models": int(len(rows)),
        "n_trained_models": n_trained,
        "models": [
            {
                "rank": int(i + 1),
                "model_name": r.model_name,
                "anomaly_threshold": r.anomaly_threshold,
                "anomaly_score_percentile": r.anomaly_score_percentile,
                "anomaly_healthy_val_score_mean": r.anomaly_healthy_val_score_mean,
                "anomaly_healthy_val_score_std": r.anomaly_healthy_val_score_std,
                "anomaly_n_full_anomalies": r.anomaly_n_full_anomalies,
                "anomaly_healthy_fpr": r.anomaly_healthy_fpr,
                "fault_n_windows_total": r.fault_n_windows_total,
                "fault_n_detected": r.fault_n_detected,
                "fault_detection_rate": r.fault_detection_rate,
                "mode_test_accuracy": r.mode_test_accuracy,
                "mode_test_macro_f1": r.mode_test_macro_f1,
            }
            for i, r in enumerate(rows)
        ],
    }
