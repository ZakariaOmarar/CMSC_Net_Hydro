"""Full pipeline orchestrator: trains and evaluates all model families in one command.

Runs the following sequence using configurations from the configs/ directory:
1. CNF anomaly model training (conditional RealNVP flow).
2. OC-SVM, LSTM-AE, and CNN-AE baseline anomaly model training.
3. One mode classifier per anomaly model family.
4. Inference over the RandomFault recordings for all families.
5. Model report and run manifest generation.

Supports resume / retry / fail-fast controls so partial runs can be continued
without re-running expensive training stages. Progress and timing are written to
train_all_manifest.json alongside model_report.json.

Run with: python -m train_all_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..cli import load_yaml_config
from ..core import JobResult, JobStatus, StageRollup
from ..core.runtime_utils import (
    compute_window_step_s as _compute_window_step_s,
    resolve_latent_paths as _resolve_latent_paths,
)
from ..flow.data import load_latent_dataset as _load_latent_dataset
from ..flow.infer import (
    FlowInferenceResult,
    apply_mode_aware_fault_logic as _flow_apply_mode_aware_fault_logic,
    build_anomaly_events as _flow_build_anomaly_events,
    filter_healthy_dataset as _flow_filter_healthy_dataset,
    healthy_mode_score_stats as _flow_healthy_mode_score_stats,
    load_mode_artifact as _flow_load_mode_artifact,
    predict_modes as _flow_predict_modes,
    load_flow_artifact as _load_flow_artifact,
    score_with_context_smoothing as _score_with_context_smoothing,
)
from ..flow.train import train_and_calibrate_flow as _train_and_calibrate_flow
from ..baselines.train import (
    build_anomaly_events as _baseline_build_anomaly_events,
    build_mode_predictions as _build_baseline_mode_predictions,
    infer_baseline_model as _infer_baseline_model,
    load_artifact as _load_baseline_artifact,
    train_baseline_model as _train_baseline_model,
)
from ..mode.train import train_mode_classifier as _train_mode_classifier
from ..reporting.report import (
    collect_model_reports as _collect_model_reports,
    to_payload as _report_to_payload,
)


@dataclass(frozen=True)
class TrainJob:
    name: str
    fn: Callable[[], object]
    description: str = ""
    output_dir: str | None = None
    stage: str = "train"


_MANIFEST_SCHEMA_VERSION = "2.0"
_ARTIFACT_LAYOUT_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Repo helpers
# ---------------------------------------------------------------------------


def _has_repo_markers(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src").is_dir()


def _find_repo_root(start: Path) -> Path:
    root = Path(start).resolve()
    for candidate in (root, *root.parents):
        if _has_repo_markers(candidate):
            return candidate
    raise FileNotFoundError(
        "Unable to locate project root from "
        f"{root}. Expected markers: pyproject.toml and src/."
    )


def _repo_root() -> Path:
    return _find_repo_root(Path(__file__).resolve())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _job_signature(jobs: list[TrainJob]) -> str:
    h = hashlib.sha256()
    for job in jobs:
        h.update(job.name.encode("utf-8"))
        h.update(b"\n")
        h.update(job.description.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _snapshot_config_files(config_files: list[Path]) -> list[dict[str, object]]:
    snapshot: list[dict[str, object]] = []
    for path in config_files:
        p = Path(path)
        stat = p.stat()
        snapshot.append(
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(p),
            }
        )
    return snapshot


def _snapshot_data_signature(data_root: Path) -> dict[str, object]:
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        return {
            "path": str(root),
            "exists": False,
            "n_files": 0,
            "total_bytes": 0,
            "signature": None,
        }

    total_bytes = 0
    file_count = 0
    lines: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rel = str(path.relative_to(root)).replace("\\", "/")
        file_count += 1
        total_bytes += int(stat.st_size)
        lines.append(f"{rel}|{int(stat.st_size)}|{int(stat.st_mtime_ns)}")

    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return {
        "path": str(root),
        "exists": True,
        "n_files": int(file_count),
        "total_bytes": int(total_bytes),
        "signature": digest,
    }


def _load_manifest(path: Path) -> dict[str, object]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Resume manifest not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid manifest format in {p}: expected JSON object")
    return raw


def _successful_job_names(manifest: dict[str, object]) -> set[str]:
    names: set[str] = set()
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return names
    for item in jobs:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        name = item.get("name")
        if status == "ok" and isinstance(name, str) and name.strip():
            names.add(name)
    return names


def _make_skipped_result(*, job: TrainJob, reason: str) -> JobResult:
    now = _utc_now_iso()
    return JobResult(
        name=job.name,
        module=job.description or job.name,
        stage=job.stage,
        command=[],
        status="skipped",
        returncode=0,
        started_at_utc=now,
        finished_at_utc=now,
        duration_s=0.0,
        orchestrator_peak_memory_mb=0.0,
        output_dir=job.output_dir,
        attempt_count=0,
        retried=False,
        skipped_reason=reason,
    )


def _run_job(
    *,
    job: TrainJob,
    dry_run: bool,
    max_retries: int,
    retry_backoff_s: float,
) -> JobResult:
    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    started_tracing_here = False
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        started_tracing_here = True

    print(f"[train-all] {job.name}")
    if job.description:
        print(f"[train-all] {job.description}")

    if dry_run:
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = round(float(peak) / (1024.0 * 1024.0), 3)
        if started_tracing_here:
            tracemalloc.stop()
        return JobResult(
            name=job.name,
            module=job.description or job.name,
            stage=job.stage,
            command=[],
            status="dry_run",
            returncode=0,
            started_at_utc=started_at,
            finished_at_utc=_utc_now_iso(),
            duration_s=0.0,
            orchestrator_peak_memory_mb=float(peak_mb),
            output_dir=job.output_dir,
            attempt_count=1,
            retried=False,
            skipped_reason=None,
        )

    attempts = 0
    returncode = 1
    last_exc: BaseException | None = None

    while True:
        attempts += 1
        try:
            job.fn()
            returncode = 0
            break
        except Exception as exc:
            last_exc = exc
            returncode = 1

        if attempts > int(max_retries):
            break

        print(
            f"[train-all] retry {attempts}/{max_retries} after failure in {job.name} "
            f"({type(last_exc).__name__}: {last_exc})"
        )
        if retry_backoff_s > 0.0:
            time.sleep(float(retry_backoff_s) * float(attempts))

    if returncode != 0 and last_exc is not None:
        print(
            f"[train-all] job failed: {job.name}: "
            f"{type(last_exc).__name__}: {last_exc}"
        )

    duration_s = round(float(time.perf_counter() - t0), 3)
    _, peak = tracemalloc.get_traced_memory()
    peak_mb = round(float(peak) / (1024.0 * 1024.0), 3)
    if started_tracing_here:
        tracemalloc.stop()

    status: JobStatus = "ok" if returncode == 0 else "failed"
    return JobResult(
        name=job.name,
        module=job.description or job.name,
        stage=job.stage,
        command=[],
        status=status,
        returncode=int(returncode),
        started_at_utc=started_at,
        finished_at_utc=_utc_now_iso(),
        duration_s=duration_s,
        orchestrator_peak_memory_mb=float(peak_mb),
        output_dir=job.output_dir,
        attempt_count=int(attempts),
        retried=bool(attempts > 1),
        skipped_reason=None,
    )


# ---------------------------------------------------------------------------
# In-process runners
# ---------------------------------------------------------------------------


def _run_flow_infer_job(
    *,
    artifact_path: Path,
    latent_root: Path,
    output_path: Path,
    mode_artifact_path: Path | None = None,
    healthy_latent_root: Path | None = None,
    window_s: float = 5.0,
    overlap: float = 0.5,
    smoother_k: int = 20,
    smoother_decay: float = 0.9,
    transition_policy: str = "context_only",
    transition_factor: float = 1.5,
    mode_consistency_window: int = 5,
    mode_stable_min_windows: int = 3,
    mode_z_threshold: float = 3.0,
    device: str = "cpu",
) -> None:
    from typing import cast as _cast
    from ..flow.infer import TransitionPolicy

    flow, threshold = _load_flow_artifact(artifact_path, device=device)
    dataset = _load_latent_dataset(_resolve_latent_paths(latent_root))
    result = _score_with_context_smoothing(
        flow,
        dataset,
        threshold=threshold,
        smoother_k=smoother_k,
        smoother_decay=smoother_decay,
        transition_policy=_cast(TransitionPolicy, transition_policy),
        transition_factor=transition_factor,
        device=device,
    )
    anomaly_events = _flow_build_anomaly_events(
        scores=result.scores,
        flags=result.flags,
        thresholds=result.thresholds,
        window_s=window_s,
        overlap=overlap,
    )

    mode_payload: dict[str, object] = {}
    if mode_artifact_path is not None:
        if healthy_latent_root is None:
            raise ValueError(
                "healthy_latent_root is required when mode_artifact_path is provided"
            )
        mode_bundle = _flow_load_mode_artifact(mode_artifact_path, device=device)
        mode_labels, mode_probabilities = _flow_predict_modes(
            mode_bundle, dataset, device=device
        )
        healthy_dataset_all = _load_latent_dataset(
            _resolve_latent_paths(healthy_latent_root)
        )
        healthy_dataset = _flow_filter_healthy_dataset(healthy_dataset_all)
        healthy_mode_stats = _flow_healthy_mode_score_stats(
            flow,
            healthy_dataset,
            mode_bundle,
            smoother_k=smoother_k,
            smoother_decay=smoother_decay,
            device=device,
        )
        (
            mode_flags,
            mode_z,
            mode_ref_thresholds,
            mode_confidence,
            mode_labels_smoothed,
            mode_run_lengths,
        ) = _flow_apply_mode_aware_fault_logic(
            base_scores=result.scores,
            base_flags=result.flags.astype(bool),
            threshold_array=result.thresholds,
            mode_labels=mode_labels,
            mode_probabilities=mode_probabilities,
            healthy_mode_stats=healthy_mode_stats,
            mode_consistency_window=mode_consistency_window,
            mode_stable_min_windows=mode_stable_min_windows,
            mode_z_threshold=mode_z_threshold,
        )
        mode_result = FlowInferenceResult(
            scores=result.scores,
            flags=mode_flags,
            thresholds=mode_ref_thresholds,
        )
        mode_events = _flow_build_anomaly_events(
            scores=mode_result.scores,
            flags=mode_result.flags,
            thresholds=mode_result.thresholds,
            window_s=window_s,
            overlap=overlap,
        )
        mode_payload = {
            "mode_aware_enabled": True,
            "mode_artifact_path": str(mode_artifact_path),
            "healthy_latent_root": str(healthy_latent_root),
            "mode_consistency_window": int(mode_consistency_window),
            "mode_stable_min_windows": int(mode_stable_min_windows),
            "mode_z_threshold": float(mode_z_threshold),
            "mode_labels": mode_labels.tolist(),
            "mode_labels_smoothed": mode_labels_smoothed.tolist(),
            "mode_confidence": mode_confidence.tolist(),
            "mode_run_lengths": mode_run_lengths.astype(int).tolist(),
            "mode_healthy_stats": {
                key: {"mean": float(v[0]), "std": float(v[1])}
                for key, v in healthy_mode_stats.items()
            },
            "mode_z_scores": mode_z.tolist(),
            "mode_reference_thresholds": mode_ref_thresholds.tolist(),
            "mode_aware_flags": mode_flags.astype(int).tolist(),
            "mode_aware_n_anomalies": int(np.sum(mode_flags)),
            "mode_aware_indices": [int(e["window_index"]) for e in mode_events],
            "mode_aware_timestamps_s": [float(e["timestamp_s"]) for e in mode_events],
            "mode_aware_events": mode_events,
        }

    payload: dict[str, object] = {
        "n_windows": int(result.scores.shape[0]),
        "threshold_default": float(threshold),
        "n_anomalies": int(np.sum(result.flags)),
        "window_step_s": _compute_window_step_s(window_s=window_s, overlap=overlap),
        "anomaly_indices": [int(e["window_index"]) for e in anomaly_events],
        "anomaly_timestamps_s": [float(e["timestamp_s"]) for e in anomaly_events],
        "anomaly_events": anomaly_events,
        "scores": result.scores.tolist(),
        "thresholds": result.thresholds.tolist(),
        "flags": result.flags.astype(int).tolist(),
        **mode_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_baseline_infer_job(
    *,
    artifact_path: Path,
    latent_root: Path,
    output_path: Path,
    mode_artifact_path: Path | None = None,
    window_s: float = 5.0,
    overlap: float = 0.5,
    mode_consistency_window: int = 5,
    score_threshold: float | None = None,
    device: str = "cpu",
) -> None:
    latent_paths = _resolve_latent_paths(latent_root)
    result = _infer_baseline_model(
        artifact_path=artifact_path,
        latent_paths=latent_paths,
        score_threshold=score_threshold,
        device=device,
    )
    artifact = _load_baseline_artifact(artifact_path)
    anomaly_events = _baseline_build_anomaly_events(
        result, window_s=window_s, overlap=overlap
    )

    mode_payload: dict[str, object] = {"mode_detection_enabled": False}
    if mode_artifact_path is not None:
        dataset = _load_latent_dataset(latent_paths)
        mode_payload = _build_baseline_mode_predictions(
            artifact_path=mode_artifact_path,
            dataset=dataset,
            anomaly_events=anomaly_events,
            mode_consistency_window=mode_consistency_window,
            device=device,
        )

    payload: dict[str, object] = {
        "model_type": str(artifact.get("model_type")),
        "n_windows": int(result.scores.shape[0]),
        "threshold_default": (
            float(result.thresholds[0]) if result.thresholds.size else 0.0
        ),
        "n_anomalies": int(np.sum(result.flags)),
        "window_step_s": _compute_window_step_s(window_s=window_s, overlap=overlap),
        "anomaly_indices": [int(e["window_index"]) for e in anomaly_events],
        "anomaly_timestamps_s": [float(e["timestamp_s"]) for e in anomaly_events],
        "anomaly_events": anomaly_events,
        "scores": result.scores.tolist(),
        "thresholds": result.thresholds.tolist(),
        "flags": result.flags.astype(int).tolist(),
        **mode_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_report_job(
    *,
    artifacts_root: Path,
    manifest_path: Path,
    output_path: Path,
) -> None:
    p = Path(manifest_path)
    rows = _collect_model_reports(
        artifacts_root=artifacts_root,
        manifest_path=p if p.exists() else None,
    )
    payload = _report_to_payload(rows)
    payload["source_manifest_path"] = str(p) if p.exists() else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config extraction helpers
# ---------------------------------------------------------------------------


def _g(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def _flow_train_kwargs(cfg: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    return {
        "latent_paths": _resolve_latent_paths(Path(str(_g(cfg, "latent_root", "")))),
        "output_dir": output_dir,
        "epochs": int(_g(cfg, "epochs", 100)),
        "batch_size": int(_g(cfg, "batch_size", 128)),
        "lr": float(_g(cfg, "lr", 1e-4)),
        "weight_decay": float(_g(cfg, "weight_decay", 1e-5)),
        "grad_clip": float(_g(cfg, "grad_clip", 1.0)),
        "n_coupling_layers": int(_g(cfg, "n_coupling_layers", 8)),
        "n_layers": (
            int(_g(cfg, "n_layers")) if _g(cfg, "n_layers") is not None else None
        ),
        "hidden_dim": int(_g(cfg, "hidden_dim", 256)),
        "context_dim": int(_g(cfg, "context_dim", 32)),
        "dropout": float(_g(cfg, "dropout", 0.0)),
        "patience": int(_g(cfg, "patience", 0)),
        "val_ratio": float(_g(cfg, "val_ratio", 0.2)),
        "test_ratio": float(_g(cfg, "test_ratio", 0.2)),
        "score_percentile": float(_g(cfg, "score_percentile", 99.0)),
        "pretrain_context_epochs": int(_g(cfg, "pretrain_context_epochs", 25)),
        "joint_finetune_epochs": int(_g(cfg, "joint_finetune_epochs", 0)),
        "joint_ntxent_weight": float(_g(cfg, "joint_ntxent_weight", 0.1)),
        "contrastive_temperature": float(_g(cfg, "contrastive_temperature", 0.2)),
        "seed": int(_g(cfg, "seed", 42)),
        "device": str(_g(cfg, "device", "cpu")),
        "log_every": int(_g(cfg, "log_every", 10)),
        "quiet": bool(_g(cfg, "quiet", False)),
        "checkpoint_path": (
            Path(str(_g(cfg, "checkpoint_path")))
            if _g(cfg, "checkpoint_path")
            else None
        ),
        "resume_checkpoint": bool(_g(cfg, "resume_checkpoint", False)),
        "checkpoint_every": int(_g(cfg, "checkpoint_every", 5)),
    }


def _baseline_train_kwargs(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    model_type: str,
    ocsvm_nu: float | None = None,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    return {
        "latent_paths": _resolve_latent_paths(Path(str(_g(cfg, "latent_root", "")))),
        "output_dir": output_dir,
        "model_type": model_type,
        "feature_set": str(_g(cfg, "feature_set", "zc")),
        "val_ratio": float(_g(cfg, "val_ratio", 0.2)),
        "score_percentile": float(_g(cfg, "score_percentile", 99.0)),
        "exclude_randomfault": bool(_g(cfg, "exclude_randomfault", True)),
        "ocsvm_kernel": str(_g(cfg, "ocsvm_kernel", "rbf")),
        "ocsvm_gamma": _g(cfg, "ocsvm_gamma", "scale"),
        "ocsvm_nu": (
            float(ocsvm_nu)
            if ocsvm_nu is not None
            else float(_g(cfg, "ocsvm_nu", 0.05))
        ),
        "seq_len": int(_g(cfg, "seq_len", 16)),
        "seq_stride": int(_g(cfg, "seq_stride", 4)),
        "hidden_dim": int(_g(cfg, "hidden_dim", 128)),
        "latent_dim": int(_g(cfg, "latent_dim", 32)),
        "n_layers": int(_g(cfg, "n_layers", 2)),
        "epochs": int(_g(cfg, "epochs", 50)),
        "batch_size": int(_g(cfg, "batch_size", 128)),
        "lr": float(_g(cfg, "lr", 1e-3)),
        "weight_decay": float(_g(cfg, "weight_decay", 1e-5)),
        "patience": int(_g(cfg, "patience", 10)),
        "cnn_spec_n_fft": int(_g(cfg, "cnn_spec_n_fft", 64)),
        "cnn_spec_hop_length": int(_g(cfg, "cnn_spec_hop_length", 16)),
        "device": str(_g(cfg, "device", "cpu")),
        "seed": int(_g(cfg, "seed", 42)),
        "artifact_name": artifact_name,
    }


def _mode_train_kwargs(
    cfg: dict[str, Any], *, output_dir: Path, seed: int
) -> dict[str, Any]:
    return {
        "latent_paths": _resolve_latent_paths(Path(str(_g(cfg, "latent_root", "")))),
        "output_dir": output_dir,
        "epochs": int(_g(cfg, "epochs", 200)),
        "batch_size": int(_g(cfg, "batch_size", 128)),
        "lr": float(_g(cfg, "lr", 5e-4)),
        "weight_decay": float(_g(cfg, "weight_decay", 1e-5)),
        "hidden_dim": int(_g(cfg, "hidden_dim", 256)),
        "classifier_dropout": float(_g(cfg, "classifier_dropout", 0.2)),
        "train_ratio": float(_g(cfg, "train_ratio", 0.8)),
        "val_ratio": float(_g(cfg, "val_ratio", 0.1)),
        "test_ratio": float(_g(cfg, "test_ratio", 0.1)),
        "cv_folds": int(_g(cfg, "cv_folds", 5)),
        "patience": int(_g(cfg, "patience", 30)),
        "label_smoothing": float(_g(cfg, "label_smoothing", 0.1)),
        "feature_augment": not bool(_g(cfg, "disable_feature_augment", False)),
        "augment_noise_std": float(_g(cfg, "augment_noise_std", 0.02)),
        "augment_scale_min": float(_g(cfg, "augment_scale_min", 0.95)),
        "augment_scale_max": float(_g(cfg, "augment_scale_max", 1.05)),
        "augment_dropout_p": float(_g(cfg, "augment_dropout_p", 0.05)),
        "mixup_alpha": float(_g(cfg, "mixup_alpha", 0.2)),
        "exclude_randomfault": bool(_g(cfg, "exclude_randomfault", False)),
        "use_class_weights": not bool(_g(cfg, "no_class_weights", False)),
        "selection_metric": str(_g(cfg, "selection_metric", "val_macro_f1")),
        "device": str(_g(cfg, "device", "cpu")),
        "log_every": int(_g(cfg, "log_every", 10)),
        "quiet": bool(_g(cfg, "quiet", False)),
        "seed": seed,
    }


def _flow_infer_base_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    healthy_raw = _g(cfg, "healthy_latent_root")
    return {
        "latent_root": Path(str(_g(cfg, "latent_root", ""))),
        "window_s": float(_g(cfg, "window_s", 5.0)),
        "overlap": float(_g(cfg, "overlap", 0.5)),
        "smoother_k": int(_g(cfg, "smoother_k", 20)),
        "smoother_decay": float(_g(cfg, "smoother_decay", 0.9)),
        "transition_policy": str(_g(cfg, "transition_policy", "context_only")),
        "transition_factor": float(_g(cfg, "transition_factor", 1.5)),
        "mode_consistency_window": int(_g(cfg, "mode_consistency_window", 5)),
        "mode_stable_min_windows": int(_g(cfg, "mode_stable_min_windows", 3)),
        "mode_z_threshold": float(_g(cfg, "mode_z_threshold", 3.0)),
        "device": str(_g(cfg, "device", "cpu")),
        "healthy_latent_root": Path(str(healthy_raw)) if healthy_raw else None,
    }


def _baseline_infer_base_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "latent_root": Path(str(_g(cfg, "latent_root", ""))),
        "window_s": float(_g(cfg, "window_s", 5.0)),
        "overlap": float(_g(cfg, "overlap", 0.5)),
        "mode_consistency_window": int(_g(cfg, "mode_consistency_window", 5)),
        "device": str(_g(cfg, "device", "cpu")),
    }


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

_MODE_DISPLAY: dict[str, str] = {
    "cnf": "CNF",
    "ocsvm": "OC-SVM",
    "lstm_ae": "LSTM-AE",
    "cnn_ae": "CNN-AE",
}


def _build_jobs(*, config_dir: Path, artifacts_root: Path) -> list[TrainJob]:
    anomaly_cfg = load_yaml_config(config_dir / "anomaly_train.yaml")
    baseline_cfg = load_yaml_config(config_dir / "anomaly_baseline_train.yaml")
    mode_cfg = load_yaml_config(config_dir / "mode_train.yaml")
    flow_infer_cfg = load_yaml_config(config_dir / "anomaly_infer_randomfault.yaml")
    baseline_infer_cfg = load_yaml_config(
        config_dir / "anomaly_baseline_infer_randomfault.yaml"
    )

    root = Path(artifacts_root)
    fi_base = _flow_infer_base_kwargs(flow_infer_cfg)
    bi_base = _baseline_infer_base_kwargs(baseline_infer_cfg)

    jobs: list[TrainJob] = []

    # CNF anomaly training
    flow_out = root / "cnf" / "anomaly"
    flow_train_kw = _flow_train_kwargs(anomaly_cfg, output_dir=flow_out)
    jobs.append(
        TrainJob(
            name="CNF anomaly",
            fn=lambda kw=flow_train_kw: _train_and_calibrate_flow(**kw),
            description=f"train_and_calibrate_flow -> {flow_out}",
            output_dir=str(flow_out),
            stage="anomaly_train",
        )
    )

    # OC-SVM variants
    ocsvm_variants: list[tuple[float | None, str, str]] = [
        (0.01, " (nu=0.01)", "anomaly_nu_001"),
        (0.03, " (nu=0.03)", "anomaly_nu_003"),
        (None, "", "anomaly"),
        (0.1, " (nu=0.1)", "anomaly_nu_01"),
    ]
    for nu, suffix, dir_name in ocsvm_variants:
        out = root / "ocsvm" / dir_name
        kw = _baseline_train_kwargs(
            baseline_cfg,
            output_dir=out,
            model_type="ocsvm",
            ocsvm_nu=nu,
            artifact_name="anomaly_model",
        )
        jobs.append(
            TrainJob(
                name=f"OC-SVM anomaly{suffix}",
                fn=lambda k=kw: _train_baseline_model(**k),
                description=f"train_baseline_model(ocsvm, nu={nu}) -> {out}",
                output_dir=str(out),
                stage="anomaly_train",
            )
        )

    # LSTM-AE and CNN-AE
    for ae_type, ae_display in [("lstm_ae", "LSTM-AE"), ("cnn_ae", "CNN-AE")]:
        out = root / ae_type / "anomaly"
        kw = _baseline_train_kwargs(
            baseline_cfg,
            output_dir=out,
            model_type=ae_type,
            artifact_name="anomaly_model",
        )
        jobs.append(
            TrainJob(
                name=f"{ae_display} anomaly",
                fn=lambda k=kw: _train_baseline_model(**k),
                description=f"train_baseline_model({ae_type}) -> {out}",
                output_dir=str(out),
                stage="anomaly_train",
            )
        )

    # Mode training
    mode_seeds = {"cnf": 41, "ocsvm": 42, "lstm_ae": 43, "cnn_ae": 44}
    for model_name, seed in mode_seeds.items():
        out = root / model_name / "mode"
        kw = _mode_train_kwargs(mode_cfg, output_dir=out, seed=seed)
        display = _MODE_DISPLAY[model_name]
        jobs.append(
            TrainJob(
                name=f"{display} mode",
                fn=lambda k=kw: _train_mode_classifier(**k),
                description=f"train_mode_classifier(seed={seed}) -> {out}",
                output_dir=str(out),
                stage="mode_train",
            )
        )

    # CNF inference
    flow_infer_out = root / "cnf" / "anomaly" / "randomfault_infer_with_timestamps.json"
    jobs.append(
        TrainJob(
            name="CNF anomaly infer (RandomFault)",
            fn=lambda: _run_flow_infer_job(
                artifact_path=root / "cnf" / "anomaly" / "flow.pt",
                mode_artifact_path=root / "cnf" / "mode" / "mode_classifier.pt",
                output_path=flow_infer_out,
                **fi_base,
            ),
            description=f"flow_infer -> {flow_infer_out}",
            output_dir=str(root / "cnf" / "anomaly"),
            stage="anomaly_infer",
        )
    )

    # Baseline inference
    baseline_infer_specs: list[tuple[str, str, str, str]] = [
        ("OC-SVM anomaly (nu=0.01)", "ocsvm", "anomaly_nu_001", "ocsvm"),
        ("OC-SVM anomaly (nu=0.03)", "ocsvm", "anomaly_nu_003", "ocsvm"),
        ("OC-SVM anomaly", "ocsvm", "anomaly", "ocsvm"),
        ("OC-SVM anomaly (nu=0.1)", "ocsvm", "anomaly_nu_01", "ocsvm"),
        ("LSTM-AE anomaly", "lstm_ae", "anomaly", "lstm_ae"),
        ("CNN-AE anomaly", "cnn_ae", "anomaly", "cnn_ae"),
    ]
    for job_base_name, model_family, dir_name, mode_family in baseline_infer_specs:
        artifact_path = root / model_family / dir_name / "anomaly_model.pkl"
        mode_artifact_path = root / mode_family / "mode" / "mode_classifier.pt"
        out = root / model_family / dir_name / "infer_randomfault.json"
        jobs.append(
            TrainJob(
                name=f"{job_base_name} infer (RandomFault)",
                fn=lambda a=artifact_path, m=mode_artifact_path, o=out: (
                    _run_baseline_infer_job(
                        artifact_path=a,
                        mode_artifact_path=m,
                        output_path=o,
                        **bi_base,
                    )
                ),
                description=f"baseline_infer({model_family}/{dir_name}) -> {out}",
                output_dir=str(root / model_family / dir_name),
                stage="anomaly_infer",
            )
        )

    return jobs


def _build_report_job(
    *,
    artifacts_root: Path,
    report_output: Path,
    manifest_path: Path,
) -> TrainJob:
    return TrainJob(
        name="Consolidated report",
        fn=lambda: _run_report_job(
            artifacts_root=artifacts_root,
            manifest_path=manifest_path,
            output_path=report_output,
        ),
        description=f"collect_model_reports -> {report_output}",
        output_dir=str(report_output.parent),
        stage="reporting",
    )


def _default_manifest_path(*, artifacts_root: Path) -> Path:
    return Path(artifacts_root) / "reports" / "train_all_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train all anomaly/mode models and run RandomFault inference.",
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--artifacts-root", default="results")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--report-output")
    parser.add_argument("--manifest-path")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume-from-manifest")
    parser.add_argument("--no-resume-skip-ok", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--data-root", default="data/All")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = _repo_root()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = root / config_dir

    required = [
        config_dir / "anomaly_train.yaml",
        config_dir / "anomaly_baseline_train.yaml",
        config_dir / "mode_train.yaml",
        config_dir / "anomaly_infer_randomfault.yaml",
        config_dir / "anomaly_baseline_infer_randomfault.yaml",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required config files: " + ", ".join(missing))

    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = root / artifacts_root
    artifacts_root.mkdir(parents=True, exist_ok=True)

    report_output = (
        Path(args.report_output)
        if args.report_output
        else artifacts_root / "reports" / "model_report.json"
    )
    if not report_output.is_absolute():
        report_output = root / report_output

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else _default_manifest_path(artifacts_root=artifacts_root)
    )
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    resume_manifest_path: Path | None = None
    if args.resume_from_manifest:
        resume_manifest_path = Path(args.resume_from_manifest)
        if not resume_manifest_path.is_absolute():
            resume_manifest_path = root / resume_manifest_path

    if int(args.max_retries) < 0:
        raise ValueError("--max-retries must be >= 0")
    if float(args.retry_backoff_s) < 0:
        raise ValueError("--retry-backoff-s must be >= 0")

    fail_fast = bool(args.fail_fast) or not bool(args.continue_on_error)
    resume_skip_ok = bool(args.resume_from_manifest) and not bool(
        args.no_resume_skip_ok
    )

    started_at_utc = _utc_now_iso()
    run_start = time.perf_counter()

    resume_ok_names: set[str] = set()
    if resume_manifest_path is not None:
        resume_manifest = _load_manifest(resume_manifest_path)
        resume_ok_names = _successful_job_names(resume_manifest)

    jobs = _build_jobs(config_dir=config_dir, artifacts_root=artifacts_root)
    job_signature = _job_signature(jobs)

    required_configs = [
        config_dir / "anomaly_train.yaml",
        config_dir / "anomaly_baseline_train.yaml",
        config_dir / "mode_train.yaml",
        config_dir / "anomaly_infer_randomfault.yaml",
        config_dir / "anomaly_baseline_infer_randomfault.yaml",
    ]
    config_snapshot = _snapshot_config_files(required_configs)
    data_snapshot = _snapshot_data_signature(data_root)

    job_results: list[JobResult] = []

    for job in jobs:
        if resume_skip_ok and job.name in resume_ok_names:
            result = _make_skipped_result(job=job, reason="resume_skip_ok")
        else:
            result = _run_job(
                job=job,
                dry_run=bool(args.dry_run),
                max_retries=int(args.max_retries),
                retry_backoff_s=float(args.retry_backoff_s),
            )
        job_results.append(result)

        if result.status == "failed" and fail_fast:
            break

    any_failed = any(r.status == "failed" for r in job_results)
    should_run_report = (not bool(args.skip_report)) and (
        not any_failed or not fail_fast
    )

    if should_run_report:
        report_job = _build_report_job(
            artifacts_root=artifacts_root,
            report_output=report_output,
            manifest_path=manifest_path,
        )
        if resume_skip_ok and report_job.name in resume_ok_names:
            report_result = _make_skipped_result(
                job=report_job, reason="resume_skip_ok"
            )
        else:
            report_result = _run_job(
                job=report_job,
                dry_run=bool(args.dry_run),
                max_retries=int(args.max_retries),
                retry_backoff_s=float(args.retry_backoff_s),
            )
        job_results.append(report_result)
        if report_result.status == "failed":
            any_failed = True

    finished_at_utc = _utc_now_iso()
    duration_s = round(float(time.perf_counter() - run_start), 3)
    failed_job_names = [r.name for r in job_results if r.status == "failed"]
    skipped_job_names = [r.name for r in job_results if r.status == "skipped"]
    retried_job_names = [r.name for r in job_results if r.retried]

    stage_summary: dict[str, StageRollup] = {}
    for result in job_results:
        bucket = stage_summary.setdefault(str(result.stage), StageRollup())
        bucket.n_jobs += 1
        bucket.duration_s = round(
            float(bucket.duration_s) + float(result.duration_s), 3
        )
        if result.status == "ok":
            bucket.n_ok += 1
        elif result.status == "failed":
            bucket.n_failed += 1
        elif result.status == "skipped":
            bucket.n_skipped += 1
        elif result.status == "dry_run":
            bucket.n_dry_run += 1
        peak_mb = float(result.orchestrator_peak_memory_mb or 0.0)
        bucket.orchestrator_peak_memory_mb = max(
            float(bucket.orchestrator_peak_memory_mb), peak_mb
        )

    overall_peak_mb = (
        max(float(s.orchestrator_peak_memory_mb) for s in stage_summary.values())
        if stage_summary
        else 0.0
    )

    if bool(args.dry_run):
        overall_status = "dry_run"
    elif failed_job_names:
        overall_status = "partial_success" if not fail_fast else "failed"
    elif skipped_job_names:
        overall_status = "resumed_success"
    else:
        overall_status = "success"

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_layout_version": _ARTIFACT_LAYOUT_VERSION,
        "run_id": (
            str(args.run_id)
            if args.run_id
            else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_s": duration_s,
        "overall_status": overall_status,
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "fail_fast": bool(fail_fast),
        "resume_from_manifest": (
            str(resume_manifest_path) if resume_manifest_path is not None else None
        ),
        "resume_skip_ok": bool(resume_skip_ok),
        "max_retries": int(args.max_retries),
        "retry_backoff_s": float(args.retry_backoff_s),
        "config_dir": str(config_dir),
        "artifacts_root": str(artifacts_root),
        "data_root": str(data_root),
        "report_output": str(report_output),
        "manifest_path": str(manifest_path),
        "job_signature": job_signature,
        "config_snapshot": config_snapshot,
        "data_snapshot_signature": data_snapshot,
        "metrics_summary": {
            "n_jobs_total": int(len(job_results)),
            "n_jobs_ok": int(sum(1 for r in job_results if r.status == "ok")),
            "n_jobs_failed": int(len(failed_job_names)),
            "n_jobs_skipped": int(len(skipped_job_names)),
            "n_jobs_dry_run": int(sum(1 for r in job_results if r.status == "dry_run")),
            "n_jobs_retried": int(len(retried_job_names)),
            "total_job_duration_s": float(
                round(sum(r.duration_s for r in job_results), 3)
            ),
        },
        "runtime": {
            "python_version": str(platform.python_version()),
            "platform": str(platform.platform()),
            "orchestrator_peak_memory_mb": float(round(overall_peak_mb, 3)),
        },
        "stage_summary": {
            name: asdict(rollup) for name, rollup in stage_summary.items()
        },
        "n_jobs": int(len(job_results)),
        "n_failed_jobs": int(len(failed_job_names)),
        "n_skipped_jobs": int(len(skipped_job_names)),
        "failed_jobs": failed_job_names,
        "skipped_jobs": skipped_job_names,
        "retried_jobs": retried_job_names,
        "jobs": [asdict(r) for r in job_results],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_status": overall_status,
                "manifest_path": str(manifest_path),
                "n_jobs": manifest["n_jobs"],
                "n_failed_jobs": manifest["n_failed_jobs"],
            },
            indent=2,
        )
    )
    return 0 if not failed_job_names else 1


if __name__ == "__main__":
    raise SystemExit(main())
