"""Second test dataset full pipeline orchestrator — mirrors train_all.py.

Runs the complete sequence on data/second_test_dataset/:

  Stage 0  Latent build       — features extracted from Pump/Turbine/Standstill
                                 recordings (5-mic, 5-accel) → artifacts/latents_second
  Stage 1  Fault latents      — per-position latents from RandomFault/ subfolders
                                 → artifacts/latents_second_fault/<pos_folder>/
  Stage 2  Anomaly train      — CNF, OC-SVM (×4), LSTM-AE, CNN-AE on normal latents
  Stage 3  Mode train         — one mode classifier per anomaly model family
  Stage 4  Anomaly infer      — per-fault-position inference for all model families
  Stage 5  Localization       — GCC-PHAT + SRP-PHAT per fault-position folder
  Stage 6  Reporting          — consolidated JSON + manifest

Dataset layout (data_root defaults to data/second_test_dataset):

  data_root/
    node_position.txt
    Pump/         recorded_{D,E,F,G,I}.wav  +  vibration_{A,B,C,D,E}.csv
    Turbine/      (same layout)
    Standstill/   (same layout)
    RandomFault/
      pos_(x,y,z)_<mode>[_<mode>]/  ← ground-truth position encoded in name
        recorded_{D,E,F,G,I}.wav
        vibration_{A,B,C,D,E}.csv

Sensor geometry: S2_MIC_XYZ / S2_VIB_XYZ / S2_FAULT_POSITIONS_M
 in src/modeling/localization/localization_head.py — all in metres.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
import tracemalloc
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...ingestion.adapters import WavVibrationAdapter
from ...modeling.localization.localization_head import (
    MIC_PAIRS_S2,
    N_PAIRS_S2,
    S2_FAULT_POSITIONS_M,
    S2_MIC_XYZ,
    _S2_MAX_MIC_DIST_M,
    gcc_phat,
)
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
from .train_all import (
    TrainJob,
    _MANIFEST_SCHEMA_VERSION,
    _ARTIFACT_LAYOUT_VERSION,
    _MODE_DISPLAY,
    _run_job,
    _run_report_job,
    _flow_train_kwargs,
    _baseline_train_kwargs,
    _mode_train_kwargs,
    _build_report_job,
    _repo_root,
    _utc_now_iso,
    _job_signature,
    _snapshot_config_files,
    _snapshot_data_signature,
    _load_manifest,
    _successful_job_names,
    _make_skipped_result,
)
from ..latent.builder import build_latent_cache as _build_latent_cache

# ---------------------------------------------------------------------------
# Folder name parsing helpers
# ---------------------------------------------------------------------------

_POS_RE = re.compile(
    r"^pos_\((?P<x>[\d.]+),(?P<y>[\d.]+),(?P<z>[\d.]+)\)_(?P<modes>.+)$",
    re.IGNORECASE,
)


def _parse_position_folder(name: str) -> tuple[np.ndarray | None, list[str]]:
    m = _POS_RE.match(name)
    if m is None:
        return None, []
    xyz = np.array(
        [float(m.group("x")), float(m.group("y")), float(m.group("z"))],
        dtype=np.float64,
    )
    modes = [s.strip() for s in m.group("modes").split("_") if s.strip()]
    return xyz, modes


def _folder_key(name: str) -> str:
    m = _POS_RE.match(name)
    if m is None:
        return name
    return f"pos_({m.group('x')},{m.group('y')},{m.group('z')})"


def _pos_dirs(rf_root: Path) -> list[Path]:
    if not rf_root.is_dir():
        return []
    return sorted(d for d in rf_root.iterdir() if d.is_dir() and _POS_RE.match(d.name))


# ---------------------------------------------------------------------------
# GCC-PHAT / SRP-PHAT helpers for 5-mic array
# ---------------------------------------------------------------------------


def _compute_gcc_stack_s2(
    mic_data: np.ndarray,
    fs: float,
    max_dist_m: float = _S2_MAX_MIC_DIST_M,
    c: float = 343.0,
) -> np.ndarray:
    max_delay = int(max_dist_m / c * fs)
    rows: list[np.ndarray] = []
    for i, j in MIC_PAIRS_S2:
        rows.append(gcc_phat(mic_data[i], mic_data[j], max_delay))
    return np.stack(rows, axis=0)


def _srp_phat_3d(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    fs: float,
    c: float = 343.0,
) -> np.ndarray:
    max_delay = gcc_stack.shape[1] // 2
    srp = np.zeros((len(grid_x), len(grid_y), len(grid_z)), dtype=np.float32)
    for ix, gx in enumerate(grid_x):
        for iy, gy in enumerate(grid_y):
            for iz, gz in enumerate(grid_z):
                pos = np.array([gx, gy, gz])
                score = 0.0
                for k, (i, j) in enumerate(MIC_PAIRS_S2):
                    di = float(np.linalg.norm(mic_xyz[i] - pos))
                    dj = float(np.linalg.norm(mic_xyz[j] - pos))
                    expected = int(round((di - dj) / c * fs)) + max_delay
                    if 0 <= expected < gcc_stack.shape[1]:
                        score += float(gcc_stack[k, expected])
                srp[ix, iy, iz] = score
    return srp


def _estimate_position_from_wav(
    mic_data: np.ndarray,
    fs: float,
    *,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
) -> tuple[np.ndarray, float]:
    mic_xyz = S2_MIC_XYZ[: min(5, mic_data.shape[0])]
    gcc_stack = _compute_gcc_stack_s2(mic_data[: mic_xyz.shape[0]], fs=fs, c=c)

    margin = 0.10
    lo = mic_xyz.min(axis=0) - margin
    hi = mic_xyz.max(axis=0) + margin
    grid_x = np.arange(lo[0], hi[0] + grid_resolution_m, grid_resolution_m)
    grid_y = np.arange(lo[1], hi[1] + grid_resolution_m, grid_resolution_m)
    grid_z = np.arange(lo[2], hi[2] + grid_resolution_m, grid_resolution_m)

    srp = _srp_phat_3d(gcc_stack, mic_xyz, grid_x, grid_y, grid_z, fs=fs, c=c)
    peak_idx = np.unravel_index(int(np.argmax(srp)), srp.shape)
    estimated = np.array(
        [grid_x[peak_idx[0]], grid_y[peak_idx[1]], grid_z[peak_idx[2]]],
        dtype=np.float64,
    )
    return estimated, float(srp[peak_idx])


# ---------------------------------------------------------------------------
# S2-adapter factory  (5 mics, 5 accels)
# ---------------------------------------------------------------------------


def _s2_adapter() -> WavVibrationAdapter:
    return WavVibrationAdapter(allowed_mic_counts=(5,), expected_accel_count=5)


# ---------------------------------------------------------------------------
# In-process runners specific to second dataset
# ---------------------------------------------------------------------------


def _run_latent_build_job_s2(
    *,
    data_root: Path,
    output_dir: Path,
    mode: str = "mic_vibration",
    acoustic_rep: str = "mfcc",
    window_s: float = 5.0,
    overlap: float = 0.5,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> None:
    _build_latent_cache(
        data_root=data_root,
        output_dir=output_dir,
        mode=mode,
        acoustic_rep=acoustic_rep,
        window_s=window_s,
        overlap=overlap,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        adapter=_s2_adapter(),
    )


def _run_flow_infer_job_s2(
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
    if mode_artifact_path is not None and healthy_latent_root is not None:
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
            "mode_labels": mode_labels.tolist(),
            "mode_labels_smoothed": mode_labels_smoothed.tolist(),
            "mode_confidence": mode_confidence.tolist(),
            "mode_run_lengths": mode_run_lengths.astype(int).tolist(),
            "mode_z_scores": mode_z.tolist(),
            "mode_reference_thresholds": mode_ref_thresholds.tolist(),
            "mode_aware_flags": mode_flags.astype(int).tolist(),
            "mode_aware_n_anomalies": int(np.sum(mode_flags)),
            "mode_aware_events": mode_events,
        }

    payload: dict[str, object] = {
        "n_windows": int(result.scores.shape[0]),
        "threshold_default": float(threshold),
        "n_anomalies": int(np.sum(result.flags)),
        "window_step_s": _compute_window_step_s(window_s=window_s, overlap=overlap),
        "anomaly_events": anomaly_events,
        "scores": result.scores.tolist(),
        "thresholds": result.thresholds.tolist(),
        "flags": result.flags.astype(int).tolist(),
        **mode_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_baseline_infer_job_s2(
    *,
    artifact_path: Path,
    latent_root: Path,
    output_path: Path,
    mode_artifact_path: Path | None = None,
    window_s: float = 5.0,
    overlap: float = 0.5,
    mode_consistency_window: int = 5,
    device: str = "cpu",
) -> None:
    latent_paths = _resolve_latent_paths(latent_root)
    result = _infer_baseline_model(
        artifact_path=artifact_path,
        latent_paths=latent_paths,
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
        "anomaly_events": anomaly_events,
        "scores": result.scores.tolist(),
        "thresholds": result.thresholds.tolist(),
        "flags": result.flags.astype(int).tolist(),
        **mode_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_localization_job(
    *,
    pos_dir: Path,
    output_path: Path,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
) -> None:
    adapter = _s2_adapter()
    segment = adapter.read_recording_directory(pos_dir)
    mic_data = segment.mic_data
    fs = float(segment.mic_sample_rate)

    estimated_m, srp_peak = _estimate_position_from_wav(
        mic_data, fs, grid_resolution_m=grid_resolution_m, c=c
    )
    estimated_cm = estimated_m * 100.0

    gt_xyz_cm, modes = _parse_position_folder(pos_dir.name)

    error_m: float | None = None
    error_cm: float | None = None
    if gt_xyz_cm is not None:
        error_m = float(np.linalg.norm(estimated_m - gt_xyz_cm / 100.0))
        error_cm = error_m * 100.0

    best_k: str | None = None
    best_d = float("inf")
    for k, known_m in S2_FAULT_POSITIONS_M.items():
        d = float(np.linalg.norm(estimated_m - known_m))
        if d < best_d:
            best_d = d
            best_k = k

    payload: dict[str, object] = {
        "status": "ok",
        "method": "srp_phat_3d",
        "pos_folder": pos_dir.name,
        "n_mic_pairs": N_PAIRS_S2,
        "mic_channels_used": mic_data.shape[0],
        "srp_peak_power": srp_peak,
        "estimated_m": estimated_m.tolist(),
        "estimated_cm": estimated_cm.tolist(),
        "ground_truth_cm": gt_xyz_cm.tolist() if gt_xyz_cm is not None else None,
        "ground_truth_m": (
            (gt_xyz_cm / 100.0).tolist() if gt_xyz_cm is not None else None
        ),
        "error_m": error_m,
        "error_cm": error_cm,
        "operating_modes": modes,
        "nearest_known_position": best_k,
        "nearest_known_error_m": float(best_d) if best_d < float("inf") else None,
        "grid_resolution_m": grid_resolution_m,
        "fs_hz": fs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Job builder for second dataset
# ---------------------------------------------------------------------------


def _build_jobs_s2(
    *,
    config_dir: Path,
    artifacts_root: Path,
    data_root: Path,
    latent_root: Path,
    latent_root_fault: Path,
    window_s: float = 5.0,
    overlap: float = 0.5,
    device: str = "cpu",
    grid_resolution_m: float = 0.02,
) -> list[TrainJob]:
    anomaly_cfg = load_yaml_config(config_dir / "anomaly_train.yaml")
    baseline_cfg = load_yaml_config(config_dir / "anomaly_baseline_train.yaml")
    mode_cfg = load_yaml_config(config_dir / "mode_train.yaml")

    # Extract typed latent hyper-params from config once (avoids Pyright dict.get -> object errors)
    _lat_mode: str = str(anomaly_cfg.get("latent_mode", "mic_vibration"))
    _lat_rep: str = str(anomaly_cfg.get("latent_acoustic_rep", "mfcc"))
    _lat_win: float = float(anomaly_cfg.get("latent_window_s", window_s))  # type: ignore[arg-type]
    _lat_ovlp: float = float(anomaly_cfg.get("latent_overlap", overlap))  # type: ignore[arg-type]
    _lat_mfcc: int = int(anomaly_cfg.get("latent_n_mfcc", 40))  # type: ignore[arg-type]
    _lat_fft: int = int(anomaly_cfg.get("latent_n_fft", 2048))  # type: ignore[arg-type]
    _lat_hop: int = int(anomaly_cfg.get("latent_hop_length", 512))  # type: ignore[arg-type]

    root = artifacts_root

    rf_root = data_root / "RandomFault"
    fault_pos_dirs = _pos_dirs(rf_root)

    jobs: list[TrainJob] = []

    # Stage latent_build: normal modes
    jobs.append(
        TrainJob(
            name="S2 latent build (normal modes)",
            fn=lambda: _run_latent_build_job_s2(
                data_root=data_root,
                output_dir=latent_root,
                mode=_lat_mode,
                acoustic_rep=_lat_rep,
                window_s=_lat_win,
                overlap=_lat_ovlp,
                n_mfcc=_lat_mfcc,
                n_fft=_lat_fft,
                hop_length=_lat_hop,
            ),
            description=f"build_latent_cache(5-mic) {data_root} -> {latent_root}",
            output_dir=str(latent_root),
            stage="latent_build",
        )
    )

    # Stage fault_latent_build: one job per fault position
    for pos_dir in fault_pos_dirs:
        pos_out = latent_root_fault / pos_dir.name
        jobs.append(
            TrainJob(
                name=f"S2 fault latents ({pos_dir.name})",
                fn=lambda p=pos_dir, o=pos_out: _run_latent_build_job_s2(
                    data_root=p,
                    output_dir=o,
                    mode=_lat_mode,
                    acoustic_rep=_lat_rep,
                    window_s=_lat_win,
                    overlap=_lat_ovlp,
                    n_mfcc=_lat_mfcc,
                    n_fft=_lat_fft,
                    hop_length=_lat_hop,
                ),
                description=f"build_latent_cache({pos_dir.name}) -> {pos_out}",
                output_dir=str(pos_out),
                stage="fault_latent_build",
            )
        )

    # Stage anomaly_train — override latent_root to second dataset path
    anomaly_cfg_s2 = dict(anomaly_cfg)
    anomaly_cfg_s2["latent_root"] = str(latent_root)
    baseline_cfg_s2 = dict(baseline_cfg)
    baseline_cfg_s2["latent_root"] = str(latent_root)
    mode_cfg_s2 = dict(mode_cfg)
    mode_cfg_s2["latent_root"] = str(latent_root)

    flow_out = root / "cnf" / "anomaly"
    jobs.append(
        TrainJob(
            name="CNF anomaly",
            fn=lambda cfg=anomaly_cfg_s2, o=flow_out: _train_and_calibrate_flow(
                **_flow_train_kwargs(cfg, output_dir=o)
            ),
            description=f"train_and_calibrate_flow -> {flow_out}",
            output_dir=str(flow_out),
            stage="anomaly_train",
        )
    )

    ocsvm_variants: list[tuple[float | None, str, str]] = [
        (0.01, " (nu=0.01)", "anomaly_nu_001"),
        (0.03, " (nu=0.03)", "anomaly_nu_003"),
        (None, "", "anomaly"),
        (0.1, " (nu=0.1)", "anomaly_nu_01"),
    ]
    for nu, suffix, dir_name in ocsvm_variants:
        out = root / "ocsvm" / dir_name
        jobs.append(
            TrainJob(
                name=f"OC-SVM anomaly{suffix}",
                fn=lambda cfg=baseline_cfg_s2, o=out, _nu=nu: _train_baseline_model(
                    **_baseline_train_kwargs(
                        cfg,
                        output_dir=o,
                        model_type="ocsvm",
                        ocsvm_nu=_nu,
                        artifact_name="anomaly_model",
                    )
                ),
                description=f"train_baseline_model(ocsvm, nu={nu}) -> {out}",
                output_dir=str(out),
                stage="anomaly_train",
            )
        )

    for ae_type, ae_display in [("lstm_ae", "LSTM-AE"), ("cnn_ae", "CNN-AE")]:
        out = root / ae_type / "anomaly"
        jobs.append(
            TrainJob(
                name=f"{ae_display} anomaly",
                fn=lambda cfg=baseline_cfg_s2, o=out, _ae=ae_type: _train_baseline_model(
                    **_baseline_train_kwargs(
                        cfg,
                        output_dir=o,
                        model_type=_ae,
                        artifact_name="anomaly_model",
                    )
                ),
                description=f"train_baseline_model({ae_type}) -> {out}",
                output_dir=str(out),
                stage="anomaly_train",
            )
        )

    # Stage mode_train
    mode_seeds = {"cnf": 41, "ocsvm": 42, "lstm_ae": 43, "cnn_ae": 44}
    for model_name, seed in mode_seeds.items():
        out = root / model_name / "mode"
        display = _MODE_DISPLAY[model_name]
        jobs.append(
            TrainJob(
                name=f"{display} mode",
                fn=lambda cfg=mode_cfg_s2, o=out, _seed=seed: _train_mode_classifier(
                    **_mode_train_kwargs(cfg, output_dir=o, seed=_seed)
                ),
                description=f"train_mode_classifier(seed={seed}) -> {out}",
                output_dir=str(out),
                stage="mode_train",
            )
        )

    # Stages anomaly_infer + localization: one set per fault position
    infer_kw = {
        "window_s": window_s,
        "overlap": overlap,
        "smoother_k": 20,
        "smoother_decay": 0.9,
        "transition_policy": "context_only",
        "transition_factor": 1.5,
        "mode_consistency_window": 5,
        "mode_stable_min_windows": 3,
        "mode_z_threshold": 3.0,
        "device": device,
    }

    for pos_dir in fault_pos_dirs:
        pos_latent = latent_root_fault / pos_dir.name
        pos_out_root = root / "fault_positions" / pos_dir.name

        cnf_infer_out = pos_out_root / "cnf_infer.json"
        jobs.append(
            TrainJob(
                name=f"S2 CNF infer ({pos_dir.name})",
                fn=lambda lr=pos_latent, o=cnf_infer_out: _run_flow_infer_job_s2(
                    artifact_path=root / "cnf" / "anomaly" / "flow.pt",
                    mode_artifact_path=root / "cnf" / "mode" / "mode_classifier.pt",
                    latent_root=lr,
                    output_path=o,
                    healthy_latent_root=latent_root,
                    **infer_kw,
                ),
                description=f"cnf_infer({pos_dir.name}) -> {cnf_infer_out}",
                output_dir=str(pos_out_root),
                stage="anomaly_infer",
            )
        )

        baseline_infer_specs: list[tuple[str, str, str, str]] = [
            ("OC-SVM (nu=0.01)", "ocsvm", "anomaly_nu_001", "ocsvm"),
            ("OC-SVM (nu=0.03)", "ocsvm", "anomaly_nu_003", "ocsvm"),
            ("OC-SVM", "ocsvm", "anomaly", "ocsvm"),
            ("OC-SVM (nu=0.1)", "ocsvm", "anomaly_nu_01", "ocsvm"),
            ("LSTM-AE", "lstm_ae", "anomaly", "lstm_ae"),
            ("CNN-AE", "cnn_ae", "anomaly", "cnn_ae"),
        ]
        for label, mf, dn, mode_fam in baseline_infer_specs:
            artifact_path = root / mf / dn / "anomaly_model.pkl"
            mode_artifact = root / mode_fam / "mode" / "mode_classifier.pt"
            out = pos_out_root / f"{mf}_{dn}_infer.json"
            jobs.append(
                TrainJob(
                    name=f"S2 {label} infer ({pos_dir.name})",
                    fn=lambda lr=pos_latent, a=artifact_path, m=mode_artifact, o=out: (
                        _run_baseline_infer_job_s2(
                            artifact_path=a,
                            latent_root=lr,
                            output_path=o,
                            mode_artifact_path=m,
                            window_s=window_s,
                            overlap=overlap,
                            mode_consistency_window=5,
                            device=device,
                        )
                    ),
                    description=f"baseline_infer({mf}/{dn}, {pos_dir.name}) -> {out}",
                    output_dir=str(pos_out_root),
                    stage="anomaly_infer",
                )
            )

        loc_out = pos_out_root / "localization.json"
        jobs.append(
            TrainJob(
                name=f"S2 localize ({pos_dir.name})",
                fn=lambda p=pos_dir, o=loc_out: _run_localization_job(
                    pos_dir=p,
                    output_path=o,
                    grid_resolution_m=grid_resolution_m,
                ),
                description=f"srp_phat_3d({pos_dir.name}) -> {loc_out}",
                output_dir=str(pos_out_root),
                stage="localization",
            )
        )

    return jobs


# ---------------------------------------------------------------------------
# Manifest path helper
# ---------------------------------------------------------------------------


def _default_manifest_path_s2(*, artifacts_root: Path) -> Path:
    return Path(artifacts_root) / "reports" / "train_second_dataset_manifest.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train all anomaly/mode models for the second test dataset "
            "and run per-position fault inference + localization."
        )
    )
    parser.add_argument("--config-dir", default="configs/second")
    parser.add_argument("--artifacts-root", default="results/second")
    parser.add_argument("--data-root", default="data/second_test_dataset")
    parser.add_argument("--latent-root", default="artifacts/latents_second")
    parser.add_argument("--latent-root-fault", default="artifacts/latents_second_fault")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--report-output")
    parser.add_argument("--manifest-path")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume-from-manifest")
    parser.add_argument("--no-resume-skip-ok", action="store_true")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--grid-resolution-m", type=float, default=0.02)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        help=(
            "Only run jobs belonging to these pipeline stages. "
            "Choices: latent_build fault_latent_build anomaly_train "
            "mode_train anomaly_infer localization. "
            "Omit to run all stages."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required config files: " + ", ".join(missing))

    artifacts_root = Path(args.artifacts_root)
    if not artifacts_root.is_absolute():
        artifacts_root = root / artifacts_root
    artifacts_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root

    latent_root = Path(args.latent_root)
    if not latent_root.is_absolute():
        latent_root = root / latent_root

    latent_root_fault = Path(args.latent_root_fault)
    if not latent_root_fault.is_absolute():
        latent_root_fault = root / latent_root_fault

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
        else _default_manifest_path_s2(artifacts_root=artifacts_root)
    )
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

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

    jobs = _build_jobs_s2(
        config_dir=config_dir,
        artifacts_root=artifacts_root,
        data_root=data_root,
        latent_root=latent_root,
        latent_root_fault=latent_root_fault,
        window_s=float(args.window_s),
        overlap=float(args.overlap),
        device=str(args.device),
        grid_resolution_m=float(args.grid_resolution_m),
    )
    job_sig = _job_signature(jobs)

    if args.stages:
        allowed = set(args.stages)
        jobs = [j for j in jobs if j.stage in allowed]

    required_configs = [
        config_dir / "anomaly_train.yaml",
        config_dir / "anomaly_baseline_train.yaml",
        config_dir / "mode_train.yaml",
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
        fault_pos_root = artifacts_root / "fault_positions"
        report_job = _build_report_job(
            artifacts_root=artifacts_root,
            report_output=report_output,
            manifest_path=manifest_path,
            fault_positions_root=fault_pos_root,
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
        "dataset": "second_test_dataset",
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
        "latent_root": str(latent_root),
        "latent_root_fault": str(latent_root_fault),
        "report_output": str(report_output),
        "manifest_path": str(manifest_path),
        "job_signature": job_sig,
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

    return 1 if any_failed else 0
