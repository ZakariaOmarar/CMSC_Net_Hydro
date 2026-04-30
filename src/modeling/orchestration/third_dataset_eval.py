"""Third test dataset full pipeline orchestrator — mirrors second_dataset_eval.py.

Runs the complete sequence on data/third_test_dataset/:

  Stage 0  Latent build       — features extracted from speed1/speed2/speed3
                                 recordings (9-mic, 4-accel) → artifacts/latents_third
  Stage 1  Fault latents      — latent for hit_between_Fl_Gr_speed1
                                 → artifacts/latents_third_fault/<fault_folder>/
  Stage 2  Anomaly train      — CNF, OC-SVM (×4), LSTM-AE, CNN-AE on normal latents
  Stage 3  Mode train         — one mode classifier per anomaly model family
  Stage 4  Localization train — train LocalizationCNNS3 on fault recording
  Stage 5  Anomaly infer      — per-fault-folder inference for all model families
  Stage 6  Localization       — SRP-PHAT(36) + TDOA(36) + S3 neural + S2 zero-shot
  Stage 7  Reporting          — consolidated JSON + manifest

Dataset layout (data_root defaults to data/third_test_dataset):

  data_root/
    position.json
    speed1/    recorded_{D_l,D_r,E,F_l,F_r,G_l,G_r,J_l,J_r}.wav
               vibration_{D,E,F,J}.csv
    speed2/    (same layout)
    speed3/    (same layout)
    hit_between_Fl_Gr_speed1/   (same layout — fault recording)

Sensor geometry: S3_MIC_XYZ / S3_VIB_XYZ in localization_head.py (metres).
9 mics (36 pairs) vs 5 mics (10 pairs) in S2 — demonstrates improved spatial
coverage in a harder environment (fan background noise).
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
    MIC_PAIRS_S3,
    N_PAIRS_S3,
    S3_MIC_XYZ,
    S3_VIB_XYZ,
    VIB_PAIRS_S3,
    N_VIB_PAIRS_S3,
    S3_HIT_FL_GR_APPROX_M,
    S3_ZERO_SHOT_MIC_INDICES,
    _S3_MAX_DELAY_SAMPLES,
    _S3_GCC_LENGTH,
    _S3_MAX_MIC_DIST_M,
    _S2_MAX_DELAY_SAMPLES,
    gcc_phat,
    compute_gcc_stack_s3_multiwindow,
    compute_gcc_stack_s2_multiwindow,
    compute_gcc_stack_structural_multiwindow,
    structural_srp_phat_3d,
    synthetic_gcc_stack,
    C_STRUCT_MS,
    LocalizationCNNS3,
    LocalizationCNNS2,
    LocalizationDualSRPNet,
    dual_srp_localization_loss,
    supervised_localization_loss_s3,
    geometric_consistency_loss_s3,
    srp_phat_3d_hierarchical,
    srp_phat_3d as _srp_phat_3d_s3_gen,
    tdoa_triangulate,
    information_fusion,
    srp_covariance,
    tdoa_covariance,
    neural_covariance,
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
# Dataset layout constants
# ---------------------------------------------------------------------------

_S3_NORMAL_MODES: frozenset[str] = frozenset({"speed1", "speed2", "speed3"})

# TDOA optimisation bounds: slightly relaxed vs S2 to accommodate S3's
# larger physical spread (max ~14 cm across mics converted to metres).
_S3_TDOA_BOUNDS: list[tuple[float, float]] = [
    (-0.05, 0.20),   # x: Fl/Fr at x=0..6 cm → 0..0.06 m + margin
    (-0.10, 0.15),   # y: range −5..5 cm → −0.05..0.05 m + margin
    (-0.02, 0.12),   # z: range 1..8 cm → 0.01..0.08 m + margin
]


def _s3_fault_dirs(data_root: Path) -> list[Path]:
    """Return all subdirectories that are NOT normal-mode folders."""
    if not data_root.is_dir():
        return []
    return sorted(
        d
        for d in data_root.iterdir()
        if d.is_dir() and d.name not in _S3_NORMAL_MODES
    )


def _parse_s3_fault_folder(name: str) -> tuple[np.ndarray | None, list[str]]:
    """Parse operating mode(s) from a S3 fault folder name.

    For hit_between_Fl_Gr_speed1 → modes = ["speed1"], approx_gt = S3_HIT_FL_GR_APPROX_M
    For unknown patterns → modes = [name], approx_gt = None.
    """
    # hit_between_Fl_Gr_speed<N> — extract operating speed from suffix
    m = re.search(r"(speed\d+)", name, re.IGNORECASE)
    modes = [m.group(1)] if m else [name]

    # Approx GT: centroid of Fl and Gr mic positions for "hit_between_Fl_Gr"
    if "Fl" in name and "Gr" in name:
        return S3_HIT_FL_GR_APPROX_M.copy(), modes
    return None, modes


# ---------------------------------------------------------------------------
# S3-adapter factory  (9 mics, 4 accels)
# ---------------------------------------------------------------------------


def _s3_adapter() -> WavVibrationAdapter:
    return WavVibrationAdapter(allowed_mic_counts=(9,), expected_accel_count=4)


# ---------------------------------------------------------------------------
# Operating-mode one-hot helpers for S3 (speed1 / speed2 / speed3)
# ---------------------------------------------------------------------------

_S3_MODE_LABELS: list[str] = ["speed1", "speed2", "speed3"]


def _s3_mode_vec(modes: list[str]) -> np.ndarray:
    """One-hot mode vector for LocalizationDualSRPNet FiLM conditioning."""
    vec = np.zeros(len(_S3_MODE_LABELS), dtype=np.float32)
    for mode in modes:
        for i, label in enumerate(_S3_MODE_LABELS):
            if label.lower() in mode.lower():
                vec[i] = 1.0
    if vec.sum() == 0.0:
        vec[0] = 1.0  # default to speed1 when mode unknown
    return vec


# ---------------------------------------------------------------------------
# Raw vibration waveform loader (native CSV sample rate, no 4 Hz resampling)
# ---------------------------------------------------------------------------

_VIB_CSV_TIME_KEYS: tuple[str, ...] = (
    "esp_time_us", "timestamp_us", "time_us", "timestamp", "time"
)
_VIB_CSV_AMP_KEYS: tuple[str, ...] = (
    "amplitude", "amp", "fft_amplitude", "peak_amplitude"
)


def _load_raw_vib_waveforms(
    vib_files: list[Path],
    fallback_sr_hz: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Read vibration CSV amplitude columns at native ESP32 sample rate.

    Returns:
        vib_data: (n_sensors, n_samples) float32 amplitude waveforms.
        sr_hz:    Native sample rate in Hz (derived from CSV timestamps).
    """
    import csv as _csv

    amp_channels: list[np.ndarray] = []
    timestamps_ref: list[int] = []

    for csv_path in sorted(vib_files):
        amps: list[float] = []
        ts: list[int] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            fieldnames = [f.strip() for f in (reader.fieldnames or [])]
            time_col = next((k for k in _VIB_CSV_TIME_KEYS if k in fieldnames), None)
            amp_col = next((k for k in _VIB_CSV_AMP_KEYS if k in fieldnames), None)
            if amp_col is None:
                raise ValueError(f"No amplitude column in {csv_path.name}")
            for i, row in enumerate(reader):
                amps.append(float(row[amp_col]))
                if time_col and row.get(time_col, ""):
                    raw_t = float(row[time_col])
                    ts.append(int(raw_t if "us" in time_col else raw_t * 1_000_000))
                else:
                    ts.append(i * 250_000)
        amp_channels.append(np.asarray(amps, dtype=np.float32))
        if not timestamps_ref:
            timestamps_ref = ts

    if not amp_channels:
        return np.zeros((0, 1), dtype=np.float32), fallback_sr_hz

    min_len = min(len(ch) for ch in amp_channels)
    vib_data = np.stack([ch[:min_len] for ch in amp_channels])

    ts_arr = np.asarray(timestamps_ref[:min_len], dtype=np.float64)
    if len(ts_arr) > 1:
        dt_us = float(np.mean(np.diff(ts_arr)))
        sr = 1_000_000.0 / dt_us if dt_us > 0 else fallback_sr_hz
    else:
        sr = fallback_sr_hz

    return vib_data, max(float(sr), fallback_sr_hz)


def _srp_phat_3d_s3(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    fs: float,
    c: float = 343.0,
) -> np.ndarray:
    """S3 wrapper: vectorised SRP-PHAT fixed to MIC_PAIRS_S3."""
    return _srp_phat_3d_s3_gen(
        gcc_stack, mic_xyz, grid_x, grid_y, grid_z, fs=fs, c=c,
        mic_pairs=MIC_PAIRS_S3,
    )


# ---------------------------------------------------------------------------
# DualSRPNet inference helper
# ---------------------------------------------------------------------------


def _run_dual_srp_inference_s3(
    srp_ac_map: np.ndarray,
    srp_str_map: np.ndarray,
    mode_vec: np.ndarray,
    artifact_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LocalizationDualSRPNet on pre-computed SRP maps.

    Returns:
        pos_m:   (3,) estimated position in metres.
        log_std: (3,) per-axis log standard deviation.
    """
    import torch

    checkpoint = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
    model_kwargs: dict[str, object] = checkpoint.get("model_kwargs", {})
    model = LocalizationDualSRPNet(**model_kwargs)  # type: ignore[arg-type]
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Normalise to [0,1] to match training preprocessing.
    ac_n = srp_ac_map / (float(srp_ac_map.max()) + 1e-6)
    str_n = srp_str_map / (float(srp_str_map.max()) + 1e-6)

    ac_t = torch.from_numpy(ac_n).unsqueeze(0).float()
    str_t = torch.from_numpy(str_n).unsqueeze(0).float()
    mv_t = torch.from_numpy(mode_vec).unsqueeze(0).float()

    with torch.no_grad():
        pos_t, log_std_t = model(ac_t, str_t, mv_t)

    return pos_t.squeeze(0).numpy().astype(np.float64), log_std_t.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# S3 GCC helper (multi-window, all 36 pairs)
# ---------------------------------------------------------------------------


def _compute_gcc_stack_s3(
    mic_data: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Wrapper: multi-window S3 GCC stack (36 pairs, shape (36, L_s3))."""
    return compute_gcc_stack_s3_multiwindow(
        mic_data, fs=fs, window_s=window_s, hop_s=hop_s, c=c
    )


def _estimate_position_s3(
    mic_data: np.ndarray,
    fs: float,
    *,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
    window_s: float = 1.0,
    hop_s: float = 0.5,
) -> tuple[np.ndarray, float]:
    """SRP-PHAT on S3's 9-mic geometry (36 pairs, hierarchical grid search)."""
    gcc_stack = _compute_gcc_stack_s3(
        mic_data[: S3_MIC_XYZ.shape[0]], fs=fs, window_s=window_s, hop_s=hop_s, c=c
    )
    margin = 0.10
    lo = S3_MIC_XYZ.min(axis=0) - margin
    hi = S3_MIC_XYZ.max(axis=0) + margin
    return srp_phat_3d_hierarchical(
        gcc_stack,
        S3_MIC_XYZ,
        lo,
        hi,
        coarse_res=grid_resolution_m,
        fine_res=grid_resolution_m / 4.0,
        fine_margin=0.06,
        fs=fs,
        c=c,
        mic_pairs=MIC_PAIRS_S3,
    )


# ---------------------------------------------------------------------------
# Latent build runner (per speed folder)
# ---------------------------------------------------------------------------


def _run_latent_build_job_s3(
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
        adapter=_s3_adapter(),
    )


# ---------------------------------------------------------------------------
# Anomaly inference runners (reuse S2 logic, identical interface)
# ---------------------------------------------------------------------------


def _run_flow_infer_job_s3(
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


def _run_baseline_infer_job_s3(
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


# ---------------------------------------------------------------------------
# LocalizationCNNS3 training
# ---------------------------------------------------------------------------


def _run_localization_train_job_s3(
    *,
    data_root: Path,
    output_path: Path,
    epochs: int = 600,
    lr: float = 1e-3,
    batch_size: int = 16,
    val_ratio: float = 0.25,
    window_s: float = 1.0,
    hop_s: float = 0.25,
    geo_consistency_weight: float = 0.25,
    d_model: int = 64,
    dropout: float = 0.1,
    patience: int = 150,
    seed: int = 42,
    device_name: str = "cpu",
) -> dict:
    """Train LocalizationCNNS3 on the third-dataset fault recording(s).

    With only one ground-truth fault position, the geometric consistency loss
    (physics-based, label-free) carries significant weight (default 0.25) to
    complement the supervised MSE signal.
    """
    import torch
    import torch.optim as optim

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    adapter = _s3_adapter()
    fault_dirs = _s3_fault_dirs(data_root)

    print(f"\n[LocalizationCNNS3 training] data_root={data_root}")
    print(f"  d_model={d_model}, window_s={window_s}s, hop_s={hop_s}s")
    print(f"  epochs={epochs}, lr={lr}, geo_weight={geo_consistency_weight}")

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for fault_dir in fault_dirs:
        approx_gt_m, _ = _parse_s3_fault_folder(fault_dir.name)
        if approx_gt_m is None:
            print(f"  [skip] {fault_dir.name}: no GT position inferred from name")
            continue
        gt_m = approx_gt_m.astype(np.float32)
        try:
            segment = adapter.read_recording_directory(fault_dir)
        except Exception as exc:
            print(f"  [skip] {fault_dir.name}: {exc}")
            continue

        mic_data = segment.mic_data
        fs = float(segment.mic_sample_rate)
        win_samples = int(window_s * fs)
        hop_samples = max(1, int(hop_s * fs))
        start = 0
        n_windows = 0
        while start + win_samples <= mic_data.shape[1]:
            frame = mic_data[:, start : start + win_samples]
            rows = [
                gcc_phat(frame[i], frame[j], _S3_MAX_DELAY_SAMPLES)
                for i, j in MIC_PAIRS_S3
            ]
            X_list.append(np.stack(rows, axis=0))  # (36, L_s3)
            y_list.append(gt_m)
            n_windows += 1
            start += hop_samples
        print(f"  {fault_dir.name}: {n_windows} windows  gt=[{gt_m * 100} cm]")

    if not X_list:
        raise RuntimeError(
            f"No fault recordings with inferable GT found under {data_root}."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    print(f"  Total windows: {len(X)}  GCC shape: {X.shape}")

    # SRP-PHAT prior for each window
    print("Computing SRP-PHAT priors ...")
    priors_list: list[np.ndarray] = []
    for fault_dir in fault_dirs:
        approx_gt_m, _ = _parse_s3_fault_folder(fault_dir.name)
        if approx_gt_m is None:
            continue
        try:
            segment = adapter.read_recording_directory(fault_dir)
        except Exception:
            continue
        mic_data = segment.mic_data
        fs = float(segment.mic_sample_rate)
        srp_peak_m, _ = _estimate_position_s3(
            mic_data, fs, window_s=window_s, hop_s=0.5
        )
        win_samples = int(window_s * fs)
        hop_samples = max(1, int(hop_s * fs))
        start = 0
        while start + win_samples <= mic_data.shape[1]:
            priors_list.append(srp_peak_m.astype(np.float32))
            start += hop_samples

    srp_priors = np.stack(priors_list, axis=0)
    print(f"  SRP priors shape: {srp_priors.shape}")

    # Train / val split
    unique_positions = np.unique(y, axis=0)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for pos in unique_positions:
        mask = np.where(np.all(y == pos, axis=1))[0]
        n_val = max(1, int(len(mask) * val_ratio))
        val_idx.extend(mask[-n_val:].tolist())
        train_idx.extend(mask[:-n_val].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    print(f"  Train: {len(train_idx)} windows  |  Val: {len(val_idx)} windows")

    device = torch.device(device_name)
    X_tr = torch.from_numpy(X[train_idx]).to(device)
    y_tr = torch.from_numpy(y[train_idx]).to(device)
    s_tr = torch.from_numpy(srp_priors[train_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    s_val = torch.from_numpy(srp_priors[val_idx]).to(device)
    mic_xyz_t = torch.tensor(S3_MIC_XYZ, dtype=torch.float32, device=device)

    model = LocalizationCNNS3(d_model=d_model, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        n_batches = max(1, len(X_tr) // batch_size)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            if len(idx) == 0:
                continue
            pred = model(X_tr[idx], s_tr[idx])
            loss_sup = supervised_localization_loss_s3(pred, y_tr[idx])
            if geo_consistency_weight > 0.0:
                loss_geo = geometric_consistency_loss_s3(
                    pred, X_tr[idx], mic_xyz_t, _S3_MAX_DELAY_SAMPLES
                )
                loss = loss_sup + geo_consistency_weight * loss_geo
            else:
                loss = loss_sup
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val, s_val)
            val_sup = supervised_localization_loss_s3(pred_val, y_val).item()
            if geo_consistency_weight > 0.0:
                val_geo = geometric_consistency_loss_s3(
                    pred_val, X_val, mic_xyz_t, _S3_MAX_DELAY_SAMPLES
                ).item()
                val_loss = val_sup + geo_consistency_weight * val_geo
            else:
                val_loss = val_sup
            mae_cm = float(torch.norm(pred_val - y_val, dim=-1).mean().item() * 100.0)

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / n_batches,
                "val_loss": val_loss,
                "val_mae_cm": mae_cm,
            }
        )
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train={epoch_loss / n_batches:.5f}  "
                f"val={val_loss:.5f}  val_mae={mae_cm:.1f} cm"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "model_kwargs": {"d_model": d_model, "dropout": dropout},
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_mae_cm": mae_cm,
                },
                str(output_path),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    artifact = {
        "state_dict": best_state,
        "model_kwargs": {"d_model": d_model, "dropout": dropout},
        "val_mae_cm": mae_cm,
        "best_val_loss": best_val_loss,
        "hp": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "window_s": window_s,
            "hop_s": hop_s,
            "geo_consistency_weight": geo_consistency_weight,
            "seed": seed,
        },
        "history": history[-50:],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, str(output_path))
    print(f"\nSaved LocalizationCNNS3: {output_path}")
    return artifact


# ---------------------------------------------------------------------------
# LocalizationDualSRPNet training for S3
# ---------------------------------------------------------------------------


def _run_dual_srp_train_job_s3(
    *,
    data_root: Path,
    output_path: Path,
    epochs: int = 600,
    lr: float = 1e-3,
    batch_size: int = 4,
    val_ratio: float = 0.25,
    grid_resolution_m: float = 0.02,
    n_modes: int = 3,
    d_film: int = 32,
    dropout: float = 0.1,
    patience: int = 100,
    seed: int = 42,
    device_name: str = "cpu",
    c: float = 343.0,
    n_augment: int = 24,
    n_synth: int = 200,
) -> dict:
    """Train LocalizationDualSRPNet on S3 fault data (acoustic + structural SRP + FiLM).

    S3 has very few fault recordings (typically one), so physics-based synthetic
    samples (n_synth=200) are generated at random grid positions to provide spatial
    diversity in training while keeping the real recording in val only.
    """
    import torch
    import torch.optim as optim

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    adapter = _s3_adapter()
    fault_dirs = _s3_fault_dirs(data_root)

    lo = S3_MIC_XYZ.min(axis=0) - 0.10
    hi = S3_MIC_XYZ.max(axis=0) + 0.10
    grid_x = np.arange(lo[0], hi[0] + grid_resolution_m, grid_resolution_m)
    grid_y = np.arange(lo[1], hi[1] + grid_resolution_m, grid_resolution_m)
    grid_z = np.arange(lo[2], hi[2] + grid_resolution_m, grid_resolution_m)

    print(f"\n[LocalizationDualSRPNet S3 training] data_root={data_root}")
    print(f"  grid: {len(grid_x)}×{len(grid_y)}×{len(grid_z)}, d_film={d_film}")
    print(f"  epochs={epochs}, lr={lr}, n_augment={n_augment}")

    ac_maps: list[np.ndarray] = []
    str_maps: list[np.ndarray] = []
    mode_vecs_list: list[np.ndarray] = []
    gt_positions: list[np.ndarray] = []

    for fault_dir in fault_dirs:
        approx_gt_m, modes = _parse_s3_fault_folder(fault_dir.name)
        if approx_gt_m is None:
            print(f"  [skip] {fault_dir.name}: no GT position")
            continue
        mode_vec = _s3_mode_vec(modes)

        try:
            segment = adapter.read_recording_directory(fault_dir)
        except Exception as exc:
            print(f"  [skip] {fault_dir.name}: {exc}")
            continue

        mic_data = segment.mic_data
        fs = float(segment.mic_sample_rate)

        gcc_ac = compute_gcc_stack_s3_multiwindow(
            mic_data[: S3_MIC_XYZ.shape[0]], fs=fs, c=c
        )
        srp_ac = _srp_phat_3d_s3(gcc_ac, S3_MIC_XYZ, grid_x, grid_y, grid_z, fs, c)

        vib_files = sorted(fault_dir.glob("vibration_*.csv"))
        try:
            vib_data_raw, vib_fs = _load_raw_vib_waveforms(vib_files)
            if vib_data_raw.shape[0] > 0:
                gcc_str = compute_gcc_stack_structural_multiwindow(
                    vib_data_raw, vib_fs, S3_VIB_XYZ, VIB_PAIRS_S3
                )
                srp_str = structural_srp_phat_3d(
                    gcc_str, S3_VIB_XYZ, grid_x, grid_y, grid_z,
                    fs=vib_fs, vib_pairs=VIB_PAIRS_S3,
                )
            else:
                srp_str = np.zeros_like(srp_ac)
        except Exception:
            srp_str = np.zeros_like(srp_ac)

        # Normalise each map to [0,1] before augmentation.
        srp_ac_n = (srp_ac / (float(srp_ac.max()) + 1e-6)).astype(np.float32)
        srp_str_n = (srp_str / (float(srp_str.max()) + 1e-6)).astype(np.float32)
        ac_maps.append(srp_ac_n)
        str_maps.append(srp_str_n)
        mode_vecs_list.append(mode_vec)
        gt_positions.append(approx_gt_m.astype(np.float32))
        print(f"  {fault_dir.name}: ac_peak={srp_ac.max():.3f}  str_peak={srp_str.max():.3f}")

    if not ac_maps:
        raise RuntimeError(f"No fault recordings with GT found under {data_root}.")

    # ── Synthetic pre-training samples ────────────────────────────────────────
    # Random positions in the grid volume → ideal GCC → SRP map via srp_phat_3d.
    # Structural SRP: uniform random noise (4 Hz vib_fs carries no TDOA).
    # Synthetic samples are always assigned to train; val uses only real recordings.
    n_real_base = len(ac_maps)
    if n_synth > 0:
        print(f"  Generating {n_synth} synthetic samples (random positions in grid) ...")
        for _ in range(n_synth):
            src = np.array(
                [rng.uniform(lo[0], hi[0]),
                 rng.uniform(lo[1], hi[1]),
                 rng.uniform(lo[2], hi[2])],
                dtype=np.float64,
            )
            gcc_syn = synthetic_gcc_stack(
                source_xyz=src, mic_xyz=S3_MIC_XYZ, mic_pairs=MIC_PAIRS_S3,
                fs=16000.0, c=c, max_delay_samples=_S3_MAX_DELAY_SAMPLES,
                sigma_samples=2.0, noise_floor=0.10,
            )
            srp_ac_syn = _srp_phat_3d_s3(gcc_syn, S3_MIC_XYZ, grid_x, grid_y, grid_z, 16000.0, c)
            srp_ac_n = (srp_ac_syn / (float(srp_ac_syn.max()) + 1e-6)).astype(np.float32)
            srp_str_n = rng.random(srp_ac_n.shape).astype(np.float32)

            m_idx = int(rng.integers(0, len(_S3_MODE_LABELS)))
            mv = np.zeros(len(_S3_MODE_LABELS), dtype=np.float32)
            mv[m_idx] = 1.0

            ac_maps.append(srp_ac_n)
            str_maps.append(srp_str_n)
            mode_vecs_list.append(mv)
            gt_positions.append(src.astype(np.float32))

    n_base = n_real_base  # val split sees only real recordings; synthetic always in train
    print(f"  Real base: {n_base}  Synthetic: {n_synth}.  Augmenting real ×{n_augment}.")

    aug_ac: list[np.ndarray] = []
    aug_str: list[np.ndarray] = []
    aug_modes: list[np.ndarray] = []
    aug_gt: list[np.ndarray] = []
    for i in range(n_base):
        for _ in range(n_augment):
            s = float(rng.uniform(0.85, 1.15))
            aug_ac.append(
                np.clip(
                    ac_maps[i] * s
                    + rng.standard_normal(ac_maps[i].shape).astype(np.float32) * 0.02,
                    0.0, None,
                )
            )
            aug_str.append(
                np.clip(
                    str_maps[i] * s
                    + rng.standard_normal(str_maps[i].shape).astype(np.float32) * 0.02,
                    0.0, None,
                )
            )
            aug_modes.append(mode_vecs_list[i])
            aug_gt.append(gt_positions[i])

    # Layout of all_ac after concatenation:
    #   indices 0..n_real_base-1                   : real base recordings
    #   indices n_real_base..n_real_total-1         : augments of real recordings
    #   indices n_real_total..n_real_total+n_synth-1 : synthetic samples
    all_ac_real = ac_maps[:n_real_base] + aug_ac
    all_str_real = str_maps[:n_real_base] + aug_str
    all_modes_real = mode_vecs_list[:n_real_base] + aug_modes
    all_gt_real = gt_positions[:n_real_base] + aug_gt

    all_ac = all_ac_real + ac_maps[n_real_base:]
    all_str = all_str_real + str_maps[n_real_base:]
    all_modes = all_modes_real + mode_vecs_list[n_real_base:]
    all_gt_pos = all_gt_real + gt_positions[n_real_base:]

    n_real_total = n_real_base * (1 + n_augment)

    # Val split: when n_base==1 use augments for train, original base for val.
    if n_base == 1:
        train_idx = list(range(n_base, n_base + n_augment))  # real augments
        val_idx = [0]                                          # real base
    else:
        n_val_pos = max(1, int(n_base * val_ratio))
        val_base_set = set(range(n_base - n_val_pos, n_base))
        train_idx = []
        val_idx = []
        for i in range(n_base):
            aug_start = n_base + i * n_augment
            bucket = val_idx if i in val_base_set else train_idx
            bucket.append(i)
            bucket.extend(range(aug_start, aug_start + n_augment))
    # Append synthetic indices to train only.
    train_idx.extend(range(n_real_total, n_real_total + n_synth))
    print(
        f"  Train: {len(train_idx)} samples "
        f"({n_real_base * (1 + n_augment) - len(val_idx)} real + {n_synth} synthetic) | "
        f"Val: {len(val_idx)} samples (real only)"
    )

    device = torch.device(device_name)

    def _t(lst: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack(lst, axis=0)).to(device)

    ac_tr = _t([all_ac[i] for i in train_idx])
    str_tr = _t([all_str[i] for i in train_idx])
    mv_tr = _t([all_modes[i] for i in train_idx])
    gt_tr = _t([all_gt_pos[i] for i in train_idx])

    ac_vl = _t([all_ac[i] for i in val_idx])
    str_vl = _t([all_str[i] for i in val_idx])
    mv_vl = _t([all_modes[i] for i in val_idx])
    gt_vl = _t([all_gt_pos[i] for i in val_idx])

    model = LocalizationDualSRPNet(n_modes=n_modes, d_film=d_film, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0
    history: list[dict] = []
    mae_cm = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(ac_tr), device=device)
        n_batches = max(1, len(ac_tr) // batch_size)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            if len(idx) == 0:
                continue
            pos_p, lsd_p = model(ac_tr[idx], str_tr[idx], mv_tr[idx])
            loss = dual_srp_localization_loss(pos_p, lsd_p, gt_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pos_vp, lsd_vp = model(ac_vl, str_vl, mv_vl)
            val_loss = dual_srp_localization_loss(pos_vp, lsd_vp, gt_vl).item()
            mae_cm = float(torch.norm(pos_vp - gt_vl, dim=-1).mean().item() * 100.0)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / n_batches,
            "val_loss": val_loss,
            "val_mae_cm": mae_cm,
        })
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train={epoch_loss / n_batches:.5f}  "
                f"val={val_loss:.5f}  val_mae={mae_cm:.1f} cm"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "model_kwargs": {"n_modes": n_modes, "d_film": d_film, "dropout": dropout},
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_mae_cm": mae_cm,
                },
                str(output_path),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nSaved DualSRPNet S3: {output_path}  (val_mae={mae_cm:.1f} cm)")
    return {
        "val_mae_cm": mae_cm,
        "best_val_loss": best_val_loss,
        "n_base_samples": n_base,
        "hp": {
            "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "n_augment": n_augment, "n_modes": n_modes, "d_film": d_film, "seed": seed,
        },
        "history": history[-50:],
    }


# ---------------------------------------------------------------------------
# Neural inference helpers
# ---------------------------------------------------------------------------


def _run_neural_localization_s3(
    mic_data: np.ndarray,
    fs: float,
    srp_prior_m: np.ndarray,
    artifact_path: Path,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Run LocalizationCNNS3 on the recording, return mean predicted position."""
    import torch

    checkpoint = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
    model_kwargs: dict[str, object] = checkpoint.get("model_kwargs", {})
    model = LocalizationCNNS3(**model_kwargs)  # type: ignore[arg-type]
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    n_samples = mic_data.shape[1]
    max_delay = _S3_MAX_DELAY_SAMPLES

    preds: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_data[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S3]
        gcc_np = np.stack(rows, axis=0)          # (36, L_s3)
        gcc_t = torch.from_numpy(gcc_np).unsqueeze(0).float()   # (1, 36, L)
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()       # (3,)
        preds.append(pred)
        start += hop_samples

    if not preds:
        gcc_stack = compute_gcc_stack_s3_multiwindow(
            mic_data, fs=fs, window_s=window_s, hop_s=hop_s, c=c
        )
        gcc_t = torch.from_numpy(gcc_stack).unsqueeze(0).float()
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()
        preds.append(pred)

    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)


def _run_neural_localization_s2_zeroshot(
    mic_data: np.ndarray,
    fs: float,
    srp_prior_m: np.ndarray,
    artifact_path: Path,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Run the S2 LocalizationCNNS2 model on the 5-mic zero-shot subset of S3.

    Extracts the five microphone channels defined by S3_ZERO_SHOT_MIC_INDICES
    (Dl=0, Dr=1, Fr=4, Gr=6, Jl=7), computes 10-pair GCC with S2 delay bounds,
    and feeds the result into the pre-trained S2 model.
    """
    import torch

    checkpoint = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
    model_kwargs: dict[str, object] = checkpoint.get("model_kwargs", {})
    model = LocalizationCNNS2(**model_kwargs)  # type: ignore[arg-type]
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 5-mic subset from S3 recording
    mic_5 = mic_data[list(S3_ZERO_SHOT_MIC_INDICES)]  # (5, N)
    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    n_samples = mic_5.shape[1]
    max_delay = _S2_MAX_DELAY_SAMPLES

    preds: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_5[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S2]
        gcc_np = np.stack(rows, axis=0)          # (10, L_s2)
        gcc_t = torch.from_numpy(gcc_np).unsqueeze(0).float()   # (1, 10, L)
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()       # (3,)
        preds.append(pred)
        start += hop_samples

    if not preds:
        gcc_stack = compute_gcc_stack_s2_multiwindow(
            mic_5, fs=fs, window_s=window_s, hop_s=hop_s, c=c
        )
        gcc_t = torch.from_numpy(gcc_stack).unsqueeze(0).float()
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()
        preds.append(pred)

    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-fault-folder localization job
# ---------------------------------------------------------------------------


def _run_localization_job_s3(
    *,
    fault_dir: Path,
    output_path: Path,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
    s3_neural_artifact: Path | None = None,
    s2_neural_artifact: Path | None = None,
    dual_srp_artifact: Path | None = None,
) -> None:
    adapter = _s3_adapter()
    segment = adapter.read_recording_directory(fault_dir)
    mic_data = segment.mic_data  # (9, N)
    fs = float(segment.mic_sample_rate)

    approx_gt_m, modes = _parse_s3_fault_folder(fault_dir.name)
    mode_vec = _s3_mode_vec(modes)

    def _err(est_m: np.ndarray) -> tuple[float | None, float | None]:
        if approx_gt_m is None:
            return None, None
        e_m = float(np.linalg.norm(est_m - approx_gt_m))
        return e_m, e_m * 100.0

    # --- Acoustic GCC stack (computed once, reused for SRP + TDOA + full grid map) ---
    gcc_ac_stack = compute_gcc_stack_s3_multiwindow(
        mic_data[: S3_MIC_XYZ.shape[0]], fs=fs, c=c
    )

    # --- Acoustic SRP-PHAT hierarchical peak estimate ---
    lo = S3_MIC_XYZ.min(axis=0) - 0.10
    hi = S3_MIC_XYZ.max(axis=0) + 0.10
    srp_estimated_m, srp_peak = srp_phat_3d_hierarchical(
        gcc_ac_stack, S3_MIC_XYZ, lo, hi,
        coarse_res=grid_resolution_m,
        fine_res=grid_resolution_m / 4.0,
        fine_margin=0.06,
        fs=fs, c=c, mic_pairs=MIC_PAIRS_S3,
    )
    srp_estimated_cm = srp_estimated_m * 100.0
    srp_err_m, srp_err_cm = _err(srp_estimated_m)

    # --- Acoustic SRP map on coarse fixed grid (for DualSRPNet) ---
    grid_x = np.arange(lo[0], hi[0] + grid_resolution_m, grid_resolution_m)
    grid_y = np.arange(lo[1], hi[1] + grid_resolution_m, grid_resolution_m)
    grid_z = np.arange(lo[2], hi[2] + grid_resolution_m, grid_resolution_m)
    srp_ac_map = _srp_phat_3d_s3(gcc_ac_stack, S3_MIC_XYZ, grid_x, grid_y, grid_z, fs, c)

    # --- Structural GCC-PHAT on raw vibration waveforms (4 accels, 6 pairs) ---
    vib_files = sorted(fault_dir.glob("vibration_*.csv"))
    str_payload: dict[str, object] = {"available": False}
    srp_str_map = np.zeros_like(srp_ac_map)
    str_estimated_m = srp_estimated_m.copy()
    str_peak_power = 0.0
    vib_fs = 4.0
    try:
        vib_data_raw, vib_fs = _load_raw_vib_waveforms(vib_files)
        if vib_data_raw.shape[0] > 0:
            gcc_str_stack = compute_gcc_stack_structural_multiwindow(
                vib_data_raw, vib_fs, S3_VIB_XYZ, VIB_PAIRS_S3
            )
            srp_str_map = structural_srp_phat_3d(
                gcc_str_stack, S3_VIB_XYZ, grid_x, grid_y, grid_z,
                fs=vib_fs, vib_pairs=VIB_PAIRS_S3,
            )
            str_peak_idx = np.unravel_index(int(np.argmax(srp_str_map)), srp_str_map.shape)
            str_estimated_m = np.array([
                grid_x[str_peak_idx[0]], grid_y[str_peak_idx[1]], grid_z[str_peak_idx[2]]
            ], dtype=np.float64)
            str_peak_power = float(srp_str_map[str_peak_idx])
            str_err_m, str_err_cm = _err(str_estimated_m)
            str_payload = {
                "available": True,
                "n_vib_pairs": N_VIB_PAIRS_S3,
                "vib_fs_hz": vib_fs,
                "c_struct_ms": C_STRUCT_MS,
                "srp_peak_power": str_peak_power,
                "estimated_m": str_estimated_m.tolist(),
                "estimated_cm": (str_estimated_m * 100.0).tolist(),
                "approx_error_m": str_err_m,
                "approx_error_cm": str_err_cm,
            }
    except Exception as exc:
        str_payload = {"available": False, "error": str(exc)}

    # --- TDOA triangulation (36-pair non-linear LS) ---
    tdoa_estimated_m, tdoa_residual = tdoa_triangulate(
        gcc_ac_stack, S3_MIC_XYZ, srp_estimated_m,
        fs=fs, c=c, mic_pairs=MIC_PAIRS_S3, bounds=_S3_TDOA_BOUNDS,
    )
    tdoa_estimated_cm = tdoa_estimated_m * 100.0
    tdoa_err_m, tdoa_err_cm = _err(tdoa_estimated_m)

    # --- S3 native neural model (LocalizationCNNS3, acoustic GCC) ---
    s3_neural_payload: dict[str, object] = {"available": False}
    s3_neural_estimated_m: np.ndarray | None = None
    if s3_neural_artifact is not None and s3_neural_artifact.is_file():
        try:
            s3_neural_estimated_m = _run_neural_localization_s3(
                mic_data=mic_data, fs=fs, srp_prior_m=srp_estimated_m,
                artifact_path=s3_neural_artifact, c=c,
            )
            s3_err_m, s3_err_cm = _err(s3_neural_estimated_m)
            s3_neural_payload = {
                "available": True,
                "method": "LocalizationCNNS3",
                "artifact": str(s3_neural_artifact),
                "estimated_m": s3_neural_estimated_m.tolist(),
                "estimated_cm": (s3_neural_estimated_m * 100.0).tolist(),
                "approx_error_m": s3_err_m,
                "approx_error_cm": s3_err_cm,
            }
        except Exception as exc:
            s3_neural_payload = {"available": False, "error": str(exc)}

    # --- S2 zero-shot neural model (5-mic subset, acoustic GCC) ---
    s2_zeroshot_payload: dict[str, object] = {"available": False}
    s2_zeroshot_estimated_m: np.ndarray | None = None
    if s2_neural_artifact is not None and s2_neural_artifact.is_file():
        try:
            s2_zeroshot_estimated_m = _run_neural_localization_s2_zeroshot(
                mic_data=mic_data, fs=fs, srp_prior_m=srp_estimated_m,
                artifact_path=s2_neural_artifact, c=c,
            )
            s2_err_m, s2_err_cm = _err(s2_zeroshot_estimated_m)
            s2_zeroshot_payload = {
                "available": True,
                "method": "LocalizationCNNS2_zeroshot",
                "artifact": str(s2_neural_artifact),
                "zero_shot_mic_indices": list(S3_ZERO_SHOT_MIC_INDICES),
                "estimated_m": s2_zeroshot_estimated_m.tolist(),
                "estimated_cm": (s2_zeroshot_estimated_m * 100.0).tolist(),
                "approx_error_m": s2_err_m,
                "approx_error_cm": s2_err_cm,
            }
        except Exception as exc:
            s2_zeroshot_payload = {"available": False, "error": str(exc)}

    # --- DualSRPNet (Cross3D, acoustic + structural maps + FiLM mode conditioning) ---
    dual_payload: dict[str, object] = {"available": False}
    dual_estimated_m: np.ndarray | None = None
    dual_log_std: np.ndarray | None = None
    if dual_srp_artifact is not None and dual_srp_artifact.is_file():
        try:
            dual_estimated_m, dual_log_std = _run_dual_srp_inference_s3(
                srp_ac_map, srp_str_map, mode_vec, dual_srp_artifact
            )
            dual_err_m, dual_err_cm = _err(dual_estimated_m)
            dual_payload = {
                "available": True,
                "method": "LocalizationDualSRPNet",
                "artifact": str(dual_srp_artifact),
                "mode_vec": mode_vec.tolist(),
                "estimated_m": dual_estimated_m.tolist(),
                "estimated_cm": (dual_estimated_m * 100.0).tolist(),
                "log_std": dual_log_std.tolist(),
                "approx_error_m": dual_err_m,
                "approx_error_cm": dual_err_cm,
            }
        except Exception as exc:
            dual_payload = {"available": False, "error": str(exc)}

    # --- Information-form fusion (replaces all hand-tuned weights) ---
    est_list: list[np.ndarray] = [srp_estimated_m, tdoa_estimated_m]
    cov_list: list[np.ndarray] = [
        srp_covariance(srp_peak),
        tdoa_covariance(tdoa_residual, N_PAIRS_S3, fs, c),
    ]
    if s3_neural_estimated_m is not None:
        est_list.append(s3_neural_estimated_m)
        cov_list.append(np.eye(3) * (0.05 ** 2))   # 5 cm isotropic prior (S3 neural actual error ~5 cm)
    if s2_zeroshot_estimated_m is not None:
        est_list.append(s2_zeroshot_estimated_m)
        cov_list.append(np.eye(3) * (0.25 ** 2))   # 25 cm prior: S2 zero-shot actual error ~38 cm, near-negligible weight
    if dual_estimated_m is not None and dual_log_std is not None:
        est_list.append(dual_estimated_m)
        cov_list.append(neural_covariance(dual_log_std))
    fused_m, fused_cov = information_fusion(est_list, cov_list)
    fused_cm = fused_m * 100.0
    fused_err_m, fused_err_cm = _err(fused_m)
    fused_std_cm = float(np.sqrt(np.trace(fused_cov)) * 100.0)

    # --- Best method by approx error ---
    best_method = "srp_phat"
    best_err_cm: float | None = srp_err_cm

    def _update_best(method: str, err_cm: float | None) -> None:
        nonlocal best_method, best_err_cm
        if err_cm is not None and (best_err_cm is None or err_cm < best_err_cm):
            best_method = method
            best_err_cm = err_cm

    _update_best("tdoa", tdoa_err_cm)
    if s3_neural_estimated_m is not None:
        _update_best("neural_cnn_s3", float(s3_neural_payload.get("approx_error_cm", float("inf"))))  # type: ignore[arg-type]
    if s2_zeroshot_estimated_m is not None:
        _update_best("neural_cnn_s2_zeroshot", float(s2_zeroshot_payload.get("approx_error_cm", float("inf"))))  # type: ignore[arg-type]
    if dual_estimated_m is not None:
        _update_best("dual_srp_net", float(dual_payload.get("approx_error_cm", float("inf"))))  # type: ignore[arg-type]
    _update_best("fused", fused_err_cm)

    best_est_map: dict[str, np.ndarray] = {
        "srp_phat": srp_estimated_m,
        "tdoa": tdoa_estimated_m,
        "fused": fused_m,
    }
    if s3_neural_estimated_m is not None:
        best_est_map["neural_cnn_s3"] = s3_neural_estimated_m
    if s2_zeroshot_estimated_m is not None:
        best_est_map["neural_cnn_s2_zeroshot"] = s2_zeroshot_estimated_m
    if dual_estimated_m is not None:
        best_est_map["dual_srp_net"] = dual_estimated_m
    best_est_m = best_est_map.get(best_method, fused_m)

    payload: dict[str, object] = {
        "status": "ok",
        "fault_folder": fault_dir.name,
        "n_mic_channels": mic_data.shape[0],
        "n_mic_pairs": N_PAIRS_S3,
        "n_vib_pairs": N_VIB_PAIRS_S3,
        "operating_modes": modes,
        "approx_gt_cm": (approx_gt_m * 100.0).tolist() if approx_gt_m is not None else None,
        "approx_gt_m": approx_gt_m.tolist() if approx_gt_m is not None else None,
        "approx_gt_note": (
            "hit at z=8cm (same height as both Fl and Gr sensors); "
            "x,y approximated as centroid — z is the reliable constraint"
            if approx_gt_m is not None
            else None
        ),
        # Acoustic SRP-PHAT (9 mics, 36 pairs)
        "srp_phat": {
            "method": "srp_phat_3d_multiwindow_hierarchical",
            "n_pairs": N_PAIRS_S3,
            "srp_peak_power": srp_peak,
            "estimated_m": srp_estimated_m.tolist(),
            "estimated_cm": srp_estimated_cm.tolist(),
            "approx_error_m": srp_err_m,
            "approx_error_cm": srp_err_cm,
            "grid_resolution_coarse_m": grid_resolution_m,
            "grid_resolution_fine_m": grid_resolution_m / 4.0,
        },
        # Structural SRP-PHAT (4 accels, 6 pairs)
        "structural_srp_phat": str_payload,
        # TDOA triangulation (36-pair non-linear LS)
        "tdoa_triangulation": {
            "method": "tdoa_lbfgsb_36pairs",
            "n_pairs": N_PAIRS_S3,
            "estimated_m": tdoa_estimated_m.tolist(),
            "estimated_cm": tdoa_estimated_cm.tolist(),
            "approx_error_m": tdoa_err_m,
            "approx_error_cm": tdoa_err_cm,
            "residual_sum_sq": float(tdoa_residual),
        },
        # S3 native neural (Transformer CNN, acoustic GCC only)
        "neural_cnn_s3": s3_neural_payload,
        # S2 zero-shot neural (5-mic subset, acoustic GCC)
        "neural_cnn_s2_zeroshot": s2_zeroshot_payload,
        # Cross3D CNN (dual SRP maps + FiLM mode conditioning)
        "dual_srp_net": dual_payload,
        # Information-form fused estimate (replaces hand-tuned weights)
        "fused": {
            "method": "information_form_fusion",
            "n_branches": len(est_list),
            "estimated_m": fused_m.tolist(),
            "estimated_cm": fused_cm.tolist(),
            "fused_std_cm": round(fused_std_cm, 3),
            "approx_error_m": fused_err_m,
            "approx_error_cm": fused_err_cm,
        },
        "best_method": best_method,
        "best_approx_error_cm": best_err_cm,
        "fs_hz": fs,
        # Legacy flat fields (best estimate)
        "method": best_method,
        "srp_peak_power": srp_peak,
        "estimated_m": best_est_m.tolist(),
        "estimated_cm": (best_est_m * 100.0).tolist(),
        "approx_error_m": best_err_cm / 100.0 if best_err_cm is not None else None,
        "approx_error_cm": best_err_cm,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Job builder for third dataset
# ---------------------------------------------------------------------------


def _build_jobs_s3(
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
    loc_epochs: int = 600,
    loc_lr: float = 1e-3,
    loc_batch_size: int = 16,
    loc_window_s: float = 1.0,
    loc_hop_s: float = 0.25,
    loc_geo_weight: float = 0.25,
    loc_patience: int = 150,
    loc_seed: int = 42,
    dual_srp_epochs: int = 200,
    dual_srp_lr: float = 1e-3,
    s2_localization_artifact: Path | None = None,
) -> list[TrainJob]:
    anomaly_cfg = load_yaml_config(config_dir / "anomaly_train.yaml")
    baseline_cfg = load_yaml_config(config_dir / "anomaly_baseline_train.yaml")
    mode_cfg = load_yaml_config(config_dir / "mode_train.yaml")

    _lat_mode: str = str(anomaly_cfg.get("latent_mode", "mic_vibration"))
    _lat_rep: str = str(anomaly_cfg.get("latent_acoustic_rep", "mfcc"))
    _lat_win: float = float(anomaly_cfg.get("latent_window_s", window_s))  # type: ignore[arg-type]
    _lat_ovlp: float = float(anomaly_cfg.get("latent_overlap", overlap))   # type: ignore[arg-type]
    _lat_mfcc: int = int(anomaly_cfg.get("latent_n_mfcc", 40))            # type: ignore[arg-type]
    _lat_fft: int = int(anomaly_cfg.get("latent_n_fft", 2048))            # type: ignore[arg-type]
    _lat_hop: int = int(anomaly_cfg.get("latent_hop_length", 512))        # type: ignore[arg-type]

    root = artifacts_root
    fault_dirs = _s3_fault_dirs(data_root)

    jobs: list[TrainJob] = []

    # ------------------------------------------------------------------
    # Stage latent_build: one job per normal-mode folder
    # ------------------------------------------------------------------
    for speed_dir in sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name in _S3_NORMAL_MODES
    ):
        jobs.append(
            TrainJob(
                name=f"S3 latent build ({speed_dir.name})",
                fn=lambda sd=speed_dir: _run_latent_build_job_s3(
                    data_root=sd,
                    output_dir=latent_root,
                    mode=_lat_mode,
                    acoustic_rep=_lat_rep,
                    window_s=_lat_win,
                    overlap=_lat_ovlp,
                    n_mfcc=_lat_mfcc,
                    n_fft=_lat_fft,
                    hop_length=_lat_hop,
                ),
                description=f"build_latent_cache(9-mic, {speed_dir.name}) -> {latent_root}",
                output_dir=str(latent_root),
                stage="latent_build",
            )
        )

    # ------------------------------------------------------------------
    # Stage fault_latent_build: one job per fault folder
    # ------------------------------------------------------------------
    for fault_dir in fault_dirs:
        fault_out = latent_root_fault / fault_dir.name
        jobs.append(
            TrainJob(
                name=f"S3 fault latents ({fault_dir.name})",
                fn=lambda fd=fault_dir, fo=fault_out: _run_latent_build_job_s3(
                    data_root=fd,
                    output_dir=fo,
                    mode=_lat_mode,
                    acoustic_rep=_lat_rep,
                    window_s=_lat_win,
                    overlap=_lat_ovlp,
                    n_mfcc=_lat_mfcc,
                    n_fft=_lat_fft,
                    hop_length=_lat_hop,
                ),
                description=f"build_latent_cache({fault_dir.name}) -> {fault_out}",
                output_dir=str(fault_out),
                stage="fault_latent_build",
            )
        )

    # ------------------------------------------------------------------
    # Stage anomaly_train (identical model families as S2)
    # ------------------------------------------------------------------
    anomaly_cfg_s3 = dict(anomaly_cfg)
    anomaly_cfg_s3["latent_root"] = str(latent_root)
    baseline_cfg_s3 = dict(baseline_cfg)
    baseline_cfg_s3["latent_root"] = str(latent_root)
    mode_cfg_s3 = dict(mode_cfg)
    mode_cfg_s3["latent_root"] = str(latent_root)

    flow_out = root / "cnf" / "anomaly"
    jobs.append(
        TrainJob(
            name="CNF anomaly",
            fn=lambda cfg=anomaly_cfg_s3, o=flow_out: _train_and_calibrate_flow(
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
                fn=lambda cfg=baseline_cfg_s3, o=out, _nu=nu: _train_baseline_model(
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
                fn=lambda cfg=baseline_cfg_s3, o=out, _ae=ae_type: _train_baseline_model(
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

    # ------------------------------------------------------------------
    # Stage mode_train (3 classes: speed1/speed2/speed3)
    # ------------------------------------------------------------------
    mode_seeds = {"cnf": 41, "ocsvm": 42, "lstm_ae": 43, "cnn_ae": 44}
    for model_name, seed in mode_seeds.items():
        out = root / model_name / "mode"
        display = _MODE_DISPLAY[model_name]
        jobs.append(
            TrainJob(
                name=f"{display} mode",
                fn=lambda cfg=mode_cfg_s3, o=out, _seed=seed: _train_mode_classifier(
                    **_mode_train_kwargs(cfg, output_dir=o, seed=_seed)
                ),
                description=f"train_mode_classifier(seed={seed}) -> {out}",
                output_dir=str(out),
                stage="mode_train",
            )
        )

    # ------------------------------------------------------------------
    # Stage localization_train: train LocalizationCNNS3
    # ------------------------------------------------------------------
    loc_artifact_s3 = artifacts_root / "localization_cnn_s3.pt"
    jobs.append(
        TrainJob(
            name="S3 LocalizationCNNS3 train",
            fn=lambda o=loc_artifact_s3: _run_localization_train_job_s3(
                data_root=data_root,
                output_path=o,
                epochs=loc_epochs,
                lr=loc_lr,
                batch_size=loc_batch_size,
                window_s=loc_window_s,
                hop_s=loc_hop_s,
                geo_consistency_weight=loc_geo_weight,
                patience=loc_patience,
                seed=loc_seed,
                device_name=device,
            ),
            description=f"train LocalizationCNNS3 -> {loc_artifact_s3}",
            output_dir=str(artifacts_root),
            stage="localization_train",
        )
    )

    # Stage localization_train: train LocalizationDualSRPNet for S3
    dual_srp_artifact_s3 = artifacts_root / "localization_dual_srp_s3.pt"
    jobs.append(
        TrainJob(
            name="S3 LocalizationDualSRPNet train",
            fn=lambda o=dual_srp_artifact_s3: _run_dual_srp_train_job_s3(
                data_root=data_root,
                output_path=o,
                epochs=dual_srp_epochs,
                lr=dual_srp_lr,
                grid_resolution_m=grid_resolution_m,
                n_modes=len(_S3_MODE_LABELS),
                seed=loc_seed,
                device_name=device,
            ),
            description=f"train LocalizationDualSRPNet S3 -> {dual_srp_artifact_s3}",
            output_dir=str(artifacts_root),
            stage="localization_train",
        )
    )

    # ------------------------------------------------------------------
    # Stages anomaly_infer + localization: one set per fault folder
    # ------------------------------------------------------------------
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

    for fault_dir in fault_dirs:
        fault_latent = latent_root_fault / fault_dir.name
        pos_out_root = root / "fault_positions" / fault_dir.name

        # CNF infer
        cnf_infer_out = pos_out_root / "cnf_infer.json"
        jobs.append(
            TrainJob(
                name=f"S3 CNF infer ({fault_dir.name})",
                fn=lambda lr=fault_latent, o=cnf_infer_out: _run_flow_infer_job_s3(
                    artifact_path=root / "cnf" / "anomaly" / "flow.pt",
                    mode_artifact_path=root / "cnf" / "mode" / "mode_classifier.pt",
                    latent_root=lr,
                    output_path=o,
                    healthy_latent_root=latent_root,
                    **infer_kw,
                ),
                description=f"cnf_infer({fault_dir.name}) -> {cnf_infer_out}",
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
                    name=f"S3 {label} infer ({fault_dir.name})",
                    fn=lambda lr=fault_latent, a=artifact_path, m=mode_artifact, o=out: (
                        _run_baseline_infer_job_s3(
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
                    description=f"baseline_infer({mf}/{dn}, {fault_dir.name}) -> {out}",
                    output_dir=str(pos_out_root),
                    stage="anomaly_infer",
                )
            )

        # Localization
        loc_out = pos_out_root / "localization.json"
        _s3_art = loc_artifact_s3
        _s2_art = s2_localization_artifact
        _da_art = dual_srp_artifact_s3
        jobs.append(
            TrainJob(
                name=f"S3 localize ({fault_dir.name})",
                fn=lambda fd=fault_dir, o=loc_out, s3a=_s3_art, s2a=_s2_art, da=_da_art: (
                    _run_localization_job_s3(
                        fault_dir=fd,
                        output_path=o,
                        grid_resolution_m=grid_resolution_m,
                        s3_neural_artifact=s3a if s3a is not None and s3a.is_file() else None,
                        s2_neural_artifact=s2a if s2a is not None and s2a.is_file() else None,
                        dual_srp_artifact=da if da.is_file() else None,
                    )
                ),
                description=f"s3_localize+dual_srp({fault_dir.name}) -> {loc_out}",
                output_dir=str(pos_out_root),
                stage="localization",
            )
        )

    return jobs


# ---------------------------------------------------------------------------
# Manifest path helper
# ---------------------------------------------------------------------------


def _default_manifest_path_s3(*, artifacts_root: Path) -> Path:
    return Path(artifacts_root) / "reports" / "train_third_dataset_manifest.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train all anomaly/mode models for the third test dataset "
            "and run per-fault fault inference + 4-method localization "
            "(SRP-PHAT 36-pair + TDOA 36-pair + S3 neural + S2 zero-shot)."
        )
    )
    parser.add_argument("--config-dir", default="configs/third")
    parser.add_argument("--artifacts-root", default="results/third")
    parser.add_argument("--data-root", default="data/third_test_dataset")
    parser.add_argument("--latent-root", default="artifacts/latents_third")
    parser.add_argument("--latent-root-fault", default="artifacts/latents_third_fault")
    parser.add_argument(
        "--s2-localization-artifact",
        default="results/second/localization_cnn_s2.pt",
        help="Path to pre-trained S2 LocalizationCNNS2 model for zero-shot transfer.",
    )
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
    # LocalizationCNNS3 training hyper-parameters
    parser.add_argument("--loc-epochs", type=int, default=600)
    parser.add_argument("--loc-lr", type=float, default=1e-3)
    parser.add_argument("--loc-batch-size", type=int, default=16)
    parser.add_argument("--loc-window-s", type=float, default=1.0)
    parser.add_argument("--loc-hop-s", type=float, default=0.25)
    parser.add_argument(
        "--loc-geo-weight",
        type=float,
        default=0.25,
        help="Geometric consistency loss weight (higher than S2 due to single GT position)",
    )
    parser.add_argument("--loc-patience", type=int, default=150)
    parser.add_argument("--loc-seed", type=int, default=42)
    parser.add_argument(
        "--dual-srp-epochs",
        type=int,
        default=200,
        help="Training epochs for LocalizationDualSRPNet S3",
    )
    parser.add_argument(
        "--dual-srp-lr",
        type=float,
        default=1e-3,
        help="Learning rate for LocalizationDualSRPNet S3",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        help=(
            "Only run jobs belonging to these pipeline stages. "
            "Choices: latent_build fault_latent_build anomaly_train "
            "mode_train localization_train anomaly_infer localization. "
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

    s2_localization_artifact = Path(args.s2_localization_artifact)
    if not s2_localization_artifact.is_absolute():
        s2_localization_artifact = root / s2_localization_artifact

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
        else _default_manifest_path_s3(artifacts_root=artifacts_root)
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

    run_start = time.perf_counter()

    resume_ok_names: set[str] = set()
    if resume_manifest_path is not None:
        resume_manifest = _load_manifest(resume_manifest_path)
        resume_ok_names = _successful_job_names(resume_manifest)

    jobs = _build_jobs_s3(
        config_dir=config_dir,
        artifacts_root=artifacts_root,
        data_root=data_root,
        latent_root=latent_root,
        latent_root_fault=latent_root_fault,
        window_s=float(args.window_s),
        overlap=float(args.overlap),
        device=str(args.device),
        grid_resolution_m=float(args.grid_resolution_m),
        loc_epochs=int(args.loc_epochs),
        loc_lr=float(args.loc_lr),
        loc_batch_size=int(args.loc_batch_size),
        loc_window_s=float(args.loc_window_s),
        loc_hop_s=float(args.loc_hop_s),
        loc_geo_weight=float(args.loc_geo_weight),
        loc_patience=int(args.loc_patience),
        loc_seed=int(args.loc_seed),
        dual_srp_epochs=int(args.dual_srp_epochs),
        dual_srp_lr=float(args.dual_srp_lr),
        s2_localization_artifact=s2_localization_artifact,
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

    n_ok = sum(1 for r in job_results if r.status in ("ok", "skipped"))
    n_fail = sum(1 for r in job_results if r.status == "failed")
    elapsed = time.perf_counter() - run_start
    print(
        f"\n[third_dataset_eval] done — {n_ok} ok / {n_fail} failed "
        f"in {elapsed:.1f}s"
    )
    return 1 if n_fail > 0 else 0
