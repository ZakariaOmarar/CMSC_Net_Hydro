"""Inference-only orchestrator: runs anomaly inference on all trained models.

Requires all model artifacts to have been produced by ``python -m train_all_models``
first.  The command re-scores the RandomFault latents through every model family,
applies mode-aware fault logic where a mode classifier is present, writes per-model
JSON result files, and regenerates ``model_report.json``.

Missing artifacts are skipped with a warning rather than failing the whole run.

Run with:  python -m infer_all_models
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cli import load_yaml_config
from ..core import StageRollup
from .train_all import (
    TrainJob,
    _ARTIFACT_LAYOUT_VERSION,
    _MANIFEST_SCHEMA_VERSION,
    _baseline_infer_base_kwargs,
    _build_report_job,
    _default_manifest_path,
    _flow_infer_base_kwargs,
    _g,
    _job_signature,
    _repo_root,
    _run_baseline_infer_job,
    _run_flow_infer_job,
    _run_job,
    _snapshot_config_files,
    _utc_now_iso,
)


def _build_infer_jobs(*, config_dir: Path, artifacts_root: Path) -> list[TrainJob]:
    """Build inference-only jobs — no model training, no mode training."""
    flow_infer_cfg = load_yaml_config(config_dir / "anomaly_infer_randomfault.yaml")
    baseline_infer_cfg = load_yaml_config(
        config_dir / "anomaly_baseline_infer_randomfault.yaml"
    )

    root = Path(artifacts_root)
    fi_base = _flow_infer_base_kwargs(flow_infer_cfg)
    bi_base = _baseline_infer_base_kwargs(baseline_infer_cfg)

    jobs: list[TrainJob] = []

    # CNF inference
    flow_artifact = root / "cnf" / "anomaly" / "flow.pt"
    flow_mode_artifact = root / "cnf" / "mode" / "mode_classifier.pt"
    flow_infer_out = root / "cnf" / "anomaly" / "randomfault_infer_with_timestamps.json"
    if flow_artifact.exists():
        jobs.append(
            TrainJob(
                name="CNF anomaly infer (RandomFault)",
                fn=lambda a=flow_artifact, m=flow_mode_artifact, o=flow_infer_out: (
                    _run_flow_infer_job(
                        artifact_path=a,
                        mode_artifact_path=m if m.exists() else None,
                        output_path=o,
                        **fi_base,
                    )
                ),
                description=f"flow_infer -> {flow_infer_out}",
                output_dir=str(root / "cnf" / "anomaly"),
                stage="anomaly_infer",
            )
        )
    else:
        print(f"[infer-all] skipping CNF infer: artifact not found ({flow_artifact})")

    # Baseline inference specs: (job name, model family, dir name, mode family)
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
        if not artifact_path.exists():
            print(
                f"[infer-all] skipping {job_base_name} infer: "
                f"artifact not found ({artifact_path})"
            )
            continue
        jobs.append(
            TrainJob(
                name=f"{job_base_name} infer (RandomFault)",
                fn=lambda a=artifact_path, m=mode_artifact_path, o=out: (
                    _run_baseline_infer_job(
                        artifact_path=a,
                        mode_artifact_path=m if m.exists() else None,
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


# ---------------------------------------------------------------------------
# Cross-model window comparison
# ---------------------------------------------------------------------------

# Each entry: (human label, path-fn relative to artifacts_root, summary-path-fn)
_COMPARISON_SOURCES: list[tuple[str, str, str]] = [
    (
        "cnf",
        "cnf/anomaly/randomfault_infer_with_timestamps.json",
        "cnf/anomaly/training_summary.json",
    ),
    (
        "ocsvm",
        "ocsvm/anomaly/infer_randomfault.json",
        "ocsvm/anomaly/anomaly_model_summary.json",
    ),
    (
        "ocsvm_nu001",
        "ocsvm/anomaly_nu_001/infer_randomfault.json",
        "ocsvm/anomaly_nu_001/anomaly_model_summary.json",
    ),
    (
        "ocsvm_nu003",
        "ocsvm/anomaly_nu_003/infer_randomfault.json",
        "ocsvm/anomaly_nu_003/anomaly_model_summary.json",
    ),
    (
        "ocsvm_nu01",
        "ocsvm/anomaly_nu_01/infer_randomfault.json",
        "ocsvm/anomaly_nu_01/anomaly_model_summary.json",
    ),
    (
        "lstm_ae",
        "lstm_ae/anomaly/infer_randomfault.json",
        "lstm_ae/anomaly/anomaly_model_summary.json",
    ),
    (
        "cnn_ae",
        "cnn_ae/anomaly/infer_randomfault.json",
        "cnn_ae/anomaly/anomaly_model_summary.json",
    ),
]


def _build_window_comparison(*, artifacts_root: Path, output_path: Path) -> None:
    """Merge all per-model inference results into a single per-window comparison JSON.

    Output schema (results/reports/randomfault_window_comparison.json):
    {
      "n_windows": 248,
      "models": ["cnf", "ocsvm", ...],
      "windows": [
        {
          "window_index": 0,
          "timestamp_s": 0.0,
          "cnf_mode_label": "Turbine",  // operating mode from CNF's mode classifier
          "n_models_flagged": 3,
          "predictions": {           // per-model prediction label
            "cnf":   "Turbine",      //   flag=0 -> cnf_mode_label (normal window)
            "ocsvm": "Anomaly",      //   flag=1 -> "Anomaly"
            ...
          }
        },
        ...
      ]
    }
    """
    root = Path(artifacts_root)

    # Load all available model results and their training summaries
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for label, infer_rel, summary_rel in _COMPARISON_SOURCES:
        infer_path = root / infer_rel
        summary_path = root / summary_rel
        if not infer_path.exists():
            continue
        infer_data = json.loads(infer_path.read_text(encoding="utf-8"))
        summary_data: dict[str, Any] = {}
        if summary_path.exists():
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        loaded[label] = (infer_data, summary_data)

    if not loaded:
        raise ValueError(
            "No inference result files found under "
            f"{root}. Run python -m infer_all_models first."
        )

    # Use the first available result to establish the window count and timestamps
    first_data = next(iter(loaded.values()))[0]
    n_windows: int = int(first_data["n_windows"])
    window_step_s: float = float(first_data.get("window_step_s", 2.5))

    # Build per-window rows
    windows: list[dict[str, Any]] = []
    for idx in range(n_windows):
        timestamp_s = round(float(idx * window_step_s), 4)

        # Operating mode from CNF's mode classifier (most accurate in tests)
        cnf_mode_label: str = ""
        if "cnf" in loaded:
            cnf_d = loaded["cnf"][0]
            mode_labels = cnf_d.get("mode_labels_smoothed") or cnf_d.get("mode_labels")
            if mode_labels and idx < len(mode_labels):
                cnf_mode_label = str(mode_labels[idx])
        if not cnf_mode_label:
            for lbl, (d, _) in loaded.items():
                ml = d.get("mode_labels_smoothed") or d.get("mode_labels")
                if ml and idx < len(ml):
                    cnf_mode_label = str(ml[idx])
                    break

        predictions: dict[str, str] = {}
        for label, (infer_data, _summary_data) in loaded.items():
            flags = infer_data.get("flags", [])
            flag = int(flags[idx]) if idx < len(flags) else 0
            predictions[label] = "Anomaly" if flag else cnf_mode_label

        n_flagged = int(sum(1 for p in predictions.values() if p == "Anomaly"))
        windows.append(
            {
                "window_index": idx,
                "timestamp_s": timestamp_s,
                "cnf_mode_label": cnf_mode_label,
                "n_models_flagged": n_flagged,
                "predictions": predictions,
            }
        )

    payload: dict[str, Any] = {
        "n_windows": n_windows,
        "n_models": len(loaded),
        "models": list(loaded.keys()),
        "window_step_s": window_step_s,
        "windows": windows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_comparison_job(*, artifacts_root: Path, output_path: Path) -> TrainJob:
    return TrainJob(
        name="Window comparison table (RandomFault)",
        fn=lambda: _build_window_comparison(
            artifacts_root=artifacts_root, output_path=output_path
        ),
        description=f"per-window cross-model comparison -> {output_path}",
        output_dir=str(output_path.parent),
        stage="reporting",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on all trained anomaly models without retraining. "
            "Artifacts must already exist (run python -m train_all_models first)."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing anomaly_infer_*.yaml files (default: configs)",
    )
    parser.add_argument(
        "--artifacts-root",
        default="results",
        help="Root directory of trained model artifacts (default: results)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip regenerating model_report.json after inference",
    )
    parser.add_argument(
        "--report-output",
        help="Override path for model_report.json output",
    )
    parser.add_argument(
        "--manifest-path",
        help="Override path for infer_all_manifest.json output",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining jobs after a failure",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (default behaviour unless --continue-on-error)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without executing them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = _repo_root()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = root / config_dir

    required_configs = [
        config_dir / "anomaly_infer_randomfault.yaml",
        config_dir / "anomaly_baseline_infer_randomfault.yaml",
    ]
    missing = [str(p) for p in required_configs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required config files: " + ", ".join(missing))

    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = root / artifacts_root

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
        else artifacts_root / "reports" / "infer_all_manifest.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    # Report collection uses the train manifest so model_report.json continues
    # to include training-time metrics (threshold, val scores, etc.)
    train_manifest_path = _default_manifest_path(artifacts_root=artifacts_root)

    fail_fast = bool(args.fail_fast) or not bool(args.continue_on_error)

    started_at_utc = _utc_now_iso()
    run_start = time.perf_counter()

    jobs = _build_infer_jobs(config_dir=config_dir, artifacts_root=artifacts_root)
    if not jobs:
        print(
            "[infer-all] No inference jobs to run. "
            "Have you trained the models? Run: python -m train_all_models"
        )
        return 1

    job_signature_val = _job_signature(jobs)
    config_snapshot = _snapshot_config_files(required_configs)

    job_results = []
    for job in jobs:
        result = _run_job(
            job=job,
            dry_run=bool(args.dry_run),
            max_retries=0,
            retry_backoff_s=0.0,
        )
        job_results.append(result)

        if result.status == "failed" and fail_fast:
            break

    any_failed = any(r.status == "failed" for r in job_results)
    should_run_report = not bool(args.skip_report) and (not any_failed or not fail_fast)

    if should_run_report:
        report_job = _build_report_job(
            artifacts_root=artifacts_root,
            report_output=report_output,
            manifest_path=train_manifest_path,
        )
        report_result = _run_job(
            job=report_job,
            dry_run=bool(args.dry_run),
            max_retries=0,
            retry_backoff_s=0.0,
        )
        job_results.append(report_result)
        if report_result.status == "failed":
            any_failed = True

    # Always build the comparison table as long as the inference jobs didn't all fail
    comparison_output = (
        artifacts_root / "reports" / "randomfault_window_comparison.json"
    )
    comparison_job = _build_comparison_job(
        artifacts_root=artifacts_root,
        output_path=comparison_output,
    )
    comparison_result = _run_job(
        job=comparison_job,
        dry_run=bool(args.dry_run),
        max_retries=0,
        retry_backoff_s=0.0,
    )
    job_results.append(comparison_result)
    if comparison_result.status == "failed":
        any_failed = True

    finished_at_utc = _utc_now_iso()
    duration_s = round(float(time.perf_counter() - run_start), 3)

    failed_job_names = [r.name for r in job_results if r.status == "failed"]
    skipped_job_names = [r.name for r in job_results if r.status == "skipped"]

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

    if bool(args.dry_run):
        overall_status = "dry_run"
    elif failed_job_names:
        overall_status = "failed"
    else:
        overall_status = "success"

    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_layout_version": _ARTIFACT_LAYOUT_VERSION,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_s": duration_s,
        "overall_status": overall_status,
        "dry_run": bool(args.dry_run),
        "config_dir": str(config_dir),
        "artifacts_root": str(artifacts_root),
        "report_output": str(report_output),
        "manifest_path": str(manifest_path),
        "job_signature": job_signature_val,
        "config_snapshot": config_snapshot,
        "metrics_summary": {
            "n_jobs_total": int(len(job_results)),
            "n_jobs_ok": int(sum(1 for r in job_results if r.status == "ok")),
            "n_jobs_failed": int(len(failed_job_names)),
            "n_jobs_skipped": int(len(skipped_job_names)),
            "n_jobs_dry_run": int(sum(1 for r in job_results if r.status == "dry_run")),
        },
        "runtime": {
            "python_version": str(platform.python_version()),
            "platform": str(platform.platform()),
        },
        "stage_summary": {
            name: asdict(rollup) for name, rollup in stage_summary.items()
        },
        "n_jobs": int(len(job_results)),
        "n_failed_jobs": int(len(failed_job_names)),
        "n_skipped_jobs": int(len(skipped_job_names)),
        "failed_jobs": failed_job_names,
        "skipped_jobs": skipped_job_names,
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
