"""Second test dataset full pipeline orchestrator — mirrors train_all.py.

Runs the complete sequence on data/second_test_dataset/:

  Stage 0  Latent build       — features extracted from Pump/Turbine/Standstill
                                 recordings (5-mic, 5-accel) → artifacts/latents_second
  Stage 1  Fault latents      — per-position latents from RandomFault/ subfolders
                                 → artifacts/latents_second_fault/<pos_folder>/
  Stage 2  Anomaly train      — CNF, OC-SVM (×4), LSTM-AE, CNN-AE on normal latents
  Stage 3  Mode train         — one mode classifier per anomaly model family
  Stage 4  Anomaly infer      — per-fault-position inference for all model families
  Stage 5  Localization train — train LocalizationCNNS2 on RandomFault recordings
  Stage 6  Localization       — GCC-PHAT + SRP-PHAT + neural per fault-position folder
  Stage 7  Reporting          — consolidated JSON + manifest

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
    _S2_MAX_DELAY_SAMPLES,
    _S2_GCC_LENGTH,
    _S2_MAX_MIC_DIST_M,
    gcc_phat,
    compute_gcc_stack_s2_multiwindow,
    LocalizationCNNS2,
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


def _srp_phat_3d_hierarchical(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    coarse_res: float = 0.02,
    fine_res: float = 0.005,
    fine_margin: float = 0.05,
    fs: float = 16000.0,
    c: float = 343.0,
) -> tuple[np.ndarray, float]:
    """Two-level hierarchical SRP-PHAT: coarse grid then fine grid around peak.

    1. Coarse search over the full bounding box at coarse_res.
    2. Fine search in a ±fine_margin cube around the coarse peak at fine_res.

    Returns:
        estimated: (3,) float64 position in metres.
        srp_peak_power: peak SRP value at the refined estimate.
    """
    # --- Coarse search ---
    grid_x = np.arange(lo[0], hi[0] + coarse_res, coarse_res)
    grid_y = np.arange(lo[1], hi[1] + coarse_res, coarse_res)
    grid_z = np.arange(lo[2], hi[2] + coarse_res, coarse_res)
    srp = _srp_phat_3d(gcc_stack, mic_xyz, grid_x, grid_y, grid_z, fs=fs, c=c)
    peak_idx = np.unravel_index(int(np.argmax(srp)), srp.shape)
    coarse_peak = np.array(
        [grid_x[peak_idx[0]], grid_y[peak_idx[1]], grid_z[peak_idx[2]]],
        dtype=np.float64,
    )

    # --- Fine search around coarse peak (clamped to the original bounding box) ---
    fine_lo = np.maximum(coarse_peak - fine_margin, lo)
    fine_hi = np.minimum(coarse_peak + fine_margin, hi)
    fine_gx = np.arange(fine_lo[0], fine_hi[0] + fine_res, fine_res)
    fine_gy = np.arange(fine_lo[1], fine_hi[1] + fine_res, fine_res)
    fine_gz = np.arange(fine_lo[2], fine_hi[2] + fine_res, fine_res)
    srp_fine = _srp_phat_3d(gcc_stack, mic_xyz, fine_gx, fine_gy, fine_gz, fs=fs, c=c)
    fine_idx = np.unravel_index(int(np.argmax(srp_fine)), srp_fine.shape)
    estimated = np.array(
        [fine_gx[fine_idx[0]], fine_gy[fine_idx[1]], fine_gz[fine_idx[2]]],
        dtype=np.float64,
    )
    return estimated, float(srp_fine[fine_idx])


def _srp_phat_3d(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    fs: float,
    c: float = 343.0,
) -> np.ndarray:
    """Vectorized SRP-PHAT over a 3-D candidate grid.

    Replaces the original Python triple-loop over grid points with NumPy
    broadcasting. Distance and TDOA arrays are (Nx, Ny, Nz) tensors; the
    loop runs only over mic pairs (N_PAIRS_S2 = 10 iterations).

    Returns: float32 array of shape (Nx, Ny, Nz).
    """
    max_delay = gcc_stack.shape[1] // 2
    L = gcc_stack.shape[1]

    # Build 3-D grid of candidate positions: (Nx, Ny, Nz, 3)
    gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_pts = np.stack([gx, gy, gz], axis=-1)  # (Nx, Ny, Nz, 3)

    srp = np.zeros((len(grid_x), len(grid_y), len(grid_z)), dtype=np.float32)

    for k, (i, j) in enumerate(MIC_PAIRS_S2):
        pi = mic_xyz[i]  # (3,)
        pj = mic_xyz[j]  # (3,)

        # Euclidean distances: (Nx, Ny, Nz)
        di = np.linalg.norm(grid_pts - pi, axis=-1)
        dj = np.linalg.norm(grid_pts - pj, axis=-1)

        # TDOA in samples → lookup index in GCC vector
        tdoa_idx = np.round((di - dj) / c * fs).astype(np.int32) + max_delay

        # Mask valid indices and look up GCC values
        valid = (tdoa_idx >= 0) & (tdoa_idx < L)
        idx_safe = np.clip(tdoa_idx, 0, L - 1)
        srp += (gcc_stack[k, idx_safe] * valid).astype(np.float32)

    return srp


def _tdoa_triangulate(
    gcc_stack: np.ndarray,
    mic_xyz: np.ndarray,
    srp_init_m: np.ndarray,
    fs: float,
    c: float = 343.0,
) -> tuple[np.ndarray, float]:
    """Refine the SRP-PHAT estimate via TDOA non-linear least-squares.

    Each of the 10 mic pairs gives a measured TDOA (GCC-PHAT peak lag). The
    source position implies a theoretical TDOA through physics. We minimise the
    residual between measured and theory-predicted TDOAs, starting from the
    SRP-PHAT peak as an initial guess.

    Unlike the grid search, this is continuous and sub-sample accurate.

    Args:
        gcc_stack:   (10, L) averaged GCC-PHAT stack.
        mic_xyz:     (5, 3) mic positions in metres.
        srp_init_m:  (3,) initial position estimate from SRP-PHAT.
        fs:          Sample rate.
        c:           Speed of sound.

    Returns:
        pos_refined: (3,) refined position in metres.
        residual:    Sum of squared TDOA residuals (in samples²).
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    max_delay = gcc_stack.shape[1] // 2

    # Measured TDOA in seconds (from GCC-PHAT peak argmax)
    measured_tdoa_s = np.array(
        [
            (float(np.argmax(gcc_stack[k])) - max_delay) / fs
            for k in range(len(MIC_PAIRS_S2))
        ]
    )

    mic_i_xyz = np.stack([mic_xyz[i] for i, j in MIC_PAIRS_S2])  # (10, 3)
    mic_j_xyz = np.stack([mic_xyz[j] for i, j in MIC_PAIRS_S2])  # (10, 3)

    def _residuals(pos: np.ndarray) -> float:
        di = np.linalg.norm(mic_i_xyz - pos, axis=-1)  # (10,)
        dj = np.linalg.norm(mic_j_xyz - pos, axis=-1)
        theory_tdoa_s = (di - dj) / c
        return float(np.sum((measured_tdoa_s - theory_tdoa_s) ** 2))

    def _jac(pos: np.ndarray) -> np.ndarray:
        di = np.linalg.norm(mic_i_xyz - pos, axis=-1, keepdims=True) + 1e-9
        dj = np.linalg.norm(mic_j_xyz - pos, axis=-1, keepdims=True) + 1e-9
        theory_tdoa_s = ((di - dj) / c).squeeze(-1)
        residuals = measured_tdoa_s - theory_tdoa_s  # (10,)
        grad_di = (pos - mic_i_xyz) / (di * c)  # (10, 3)
        grad_dj = (pos - mic_j_xyz) / (dj * c)
        grad_theory = grad_di - grad_dj  # (10, 3)  ∂theory_tdoa/∂pos
        return float(-2) * (residuals[:, None] * grad_theory).sum(axis=0)  # (3,)

    result = minimize(
        _residuals,
        x0=srp_init_m,
        jac=_jac,
        method="L-BFGS-B",
        bounds=[(-0.10, 0.30), (-0.10, 0.60), (-0.10, 0.40)],
        options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
    )
    return result.x.astype(np.float64), float(result.fun)


def _estimate_position_from_wav(
    mic_data: np.ndarray,
    fs: float,
    *,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
    window_s: float = 1.0,
    hop_s: float = 0.5,
) -> tuple[np.ndarray, float]:
    mic_xyz = S2_MIC_XYZ[: min(5, mic_data.shape[0])]

    # Multi-window averaged GCC (short-window PHAT + time averaging)
    gcc_stack = compute_gcc_stack_s2_multiwindow(
        mic_data[: mic_xyz.shape[0]],
        fs=fs,
        window_s=window_s,
        hop_s=hop_s,
        c=c,
    )

    margin = 0.10
    lo = mic_xyz.min(axis=0) - margin
    hi = mic_xyz.max(axis=0) + margin

    # Hierarchical SRP-PHAT: coarse at grid_resolution_m, fine at /4
    estimated, srp_peak = _srp_phat_3d_hierarchical(
        gcc_stack,
        mic_xyz,
        lo,
        hi,
        coarse_res=grid_resolution_m,
        fine_res=grid_resolution_m / 4.0,
        fine_margin=0.06,
        fs=fs,
        c=c,
    )
    return estimated, srp_peak


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


_LOC_TRAIN_POS_RE = re.compile(
    r"^pos_\((?P<x>[\d.]+),(?P<y>[\d.]+),(?P<z>[\d.]+)\)", re.IGNORECASE
)


def _run_localization_train_job(
    *,
    data_root: Path,
    output_path: Path,
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 16,
    val_ratio: float = 0.25,
    window_s: float = 1.0,
    hop_s: float = 0.25,
    geo_consistency_weight: float = 0.05,
    d_model: int = 64,
    dropout: float = 0.1,
    patience: int = 100,
    seed: int = 42,
    device_name: str = "cpu",
) -> dict:
    """Train LocalizationCNNS2 on second-dataset RandomFault recordings."""
    import torch
    import torch.optim as optim
    from ...modeling.localization.localization_head import (
        supervised_localization_loss_s2,
        geometric_consistency_loss_s2,
    )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    adapter = WavVibrationAdapter(allowed_mic_counts=(5,), expected_accel_count=5)
    rf_root = data_root / "RandomFault"

    print(f"\n[LocalizationCNNS2 training] data_root={data_root}")
    print(f"  d_model={d_model}, window_s={window_s}s, hop_s={hop_s}s")
    print(f"  epochs={epochs}, lr={lr}, batch_size={batch_size}")

    # Collect GCC windows from all fault positions
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for pos_dir in sorted(rf_root.iterdir()):
        if not pos_dir.is_dir():
            continue
        m = _LOC_TRAIN_POS_RE.match(pos_dir.name)
        if m is None:
            continue
        gt_m = (
            np.array(
                [float(m.group("x")), float(m.group("y")), float(m.group("z"))],
                dtype=np.float32,
            )
            / 100.0
        )
        try:
            segment = adapter.read_recording_directory(pos_dir)
        except Exception as exc:
            print(f"  [skip] {pos_dir.name}: {exc}")
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
                gcc_phat(frame[i], frame[j], _S2_MAX_DELAY_SAMPLES)
                for i, j in MIC_PAIRS_S2
            ]
            X_list.append(np.stack(rows, axis=0))
            y_list.append(gt_m)
            n_windows += 1
            start += hop_samples
        print(f"  {pos_dir.name}: {n_windows} windows  gt=[{gt_m * 100} cm]")

    if not X_list:
        raise RuntimeError(f"No recordings found under {rf_root}.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    print(f"  Total windows: {len(X)}  GCC shape: {X.shape}")

    # Compute SRP-PHAT prior for each window (whole-recording peak, stationary source)
    print("Computing SRP-PHAT priors ...")
    priors_list: list[np.ndarray] = []
    for pos_dir in sorted(rf_root.iterdir()):
        if not pos_dir.is_dir() or _LOC_TRAIN_POS_RE.match(pos_dir.name) is None:
            continue
        try:
            segment = adapter.read_recording_directory(pos_dir)
        except Exception:
            continue
        mic_data = segment.mic_data
        fs = float(segment.mic_sample_rate)
        srp_peak_m, _ = _estimate_position_from_wav(
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

    # Train/val split stratified by position
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
    mic_xyz_t = torch.tensor(S2_MIC_XYZ, dtype=torch.float32, device=device)

    model = LocalizationCNNS2(d_model=d_model, dropout=dropout).to(device)
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
            loss_sup = supervised_localization_loss_s2(pred, y_tr[idx])
            if geo_consistency_weight > 0.0:
                loss_geo = geometric_consistency_loss_s2(
                    pred, X_tr[idx], mic_xyz_t, _S2_MAX_DELAY_SAMPLES
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
            val_sup = supervised_localization_loss_s2(pred_val, y_val).item()
            if geo_consistency_weight > 0.0:
                val_geo = geometric_consistency_loss_s2(
                    pred_val, X_val, mic_xyz_t, _S2_MAX_DELAY_SAMPLES
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
        if epoch % 10 == 0 or epoch == 1:
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

    model.load_state_dict(best_state)
    model.eval()
    per_pos_errors: list[dict] = []
    with torch.no_grad():
        for pos in unique_positions:
            mask_val = [i for i, vi in enumerate(val_idx) if np.all(y[vi] == pos)]
            if not mask_val:
                continue
            mask_t = torch.tensor(mask_val, device=device)
            pred_pos = model(X_val[mask_t], s_val[mask_t])
            errs_cm = torch.norm(pred_pos - y_val[mask_t], dim=-1) * 100.0
            pos_cm = (pos * 100).round().astype(int).tolist()
            per_pos_errors.append(
                {"gt_cm": pos_cm, "mean_error_cm": float(errs_cm.mean().item())}
            )

    overall_mae = float(np.mean([e["mean_error_cm"] for e in per_pos_errors]))
    print(f"\nOverall val MAE: {overall_mae:.1f} cm")

    artifact = {
        "state_dict": best_state,
        "model_kwargs": {"d_model": d_model, "dropout": dropout},
        "val_mae_cm": overall_mae,
        "best_val_loss": best_val_loss,
        "per_position_errors": per_pos_errors,
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
    print(f"\nSaved: {output_path}")
    return artifact


def _run_neural_localization(
    mic_data: np.ndarray,
    fs: float,
    srp_prior_m: np.ndarray,
    artifact_path: Path,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Run LocalizationCNNS2 on the recording and return mean predicted position.

    Segments the recording into overlapping windows, computes per-window GCC
    stacks, runs the neural model on each, and averages the predictions.

    Args:
        mic_data:      (5, N) mic waveforms.
        fs:            Sample rate.
        srp_prior_m:   (3,) SRP-PHAT peak estimate — passed as neural prior token.
        artifact_path: Path to 'localization_cnn_s2.pt' checkpoint.
        window_s:      Window length for GCC computation.
        hop_s:         Hop between windows.
        c:             Speed of sound.

    Returns:
        mean_pred: (3,) float64 — averaged (x, y, z) prediction in metres.
    """
    import torch

    checkpoint = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
    model_kwargs: dict[str, object] = checkpoint.get("model_kwargs", {})
    model = LocalizationCNNS2(**model_kwargs)  # type: ignore[arg-type]
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    n_samples = mic_data.shape[1]
    max_delay = _S2_MAX_DELAY_SAMPLES

    preds: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_data[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S2]
        gcc_np = np.stack(rows, axis=0)  # (10, L)
        gcc_t = torch.from_numpy(gcc_np).unsqueeze(0).float()  # (1, 10, L)
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()  # (3,)
        preds.append(pred)
        start += hop_samples

    if not preds:
        # Fallback: single-window on full recording
        gcc_stack = compute_gcc_stack_s2_multiwindow(
            mic_data, fs=fs, window_s=window_s, hop_s=hop_s, c=c
        )
        gcc_t = torch.from_numpy(gcc_stack).unsqueeze(0).float()
        srp_t = torch.tensor(srp_prior_m, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(gcc_t, srp_t).squeeze(0).numpy()
        preds.append(pred)

    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float64)


def _run_localization_job(
    *,
    pos_dir: Path,
    output_path: Path,
    grid_resolution_m: float = 0.02,
    c: float = 343.0,
    neural_artifact: Path | None = None,
) -> None:
    adapter = _s2_adapter()
    segment = adapter.read_recording_directory(pos_dir)
    mic_data = segment.mic_data
    fs = float(segment.mic_sample_rate)

    # --- Improved SRP-PHAT (multi-window GCC + hierarchical grid search) ---
    srp_estimated_m, srp_peak = _estimate_position_from_wav(
        mic_data, fs, grid_resolution_m=grid_resolution_m, c=c
    )
    srp_estimated_cm = srp_estimated_m * 100.0

    # --- TDOA direct triangulation (non-linear LS on GCC argmax peaks) ---
    gcc_stack_avg = compute_gcc_stack_s2_multiwindow(
        mic_data[: S2_MIC_XYZ.shape[0]], fs=fs, c=c
    )
    tdoa_estimated_m, tdoa_residual = _tdoa_triangulate(
        gcc_stack_avg, S2_MIC_XYZ, srp_estimated_m, fs=fs, c=c
    )
    tdoa_estimated_cm = tdoa_estimated_m * 100.0

    gt_xyz_cm, modes = _parse_position_folder(pos_dir.name)
    gt_xyz_m = gt_xyz_cm / 100.0 if gt_xyz_cm is not None else None

    def _err(est_m: np.ndarray) -> tuple[float | None, float | None]:
        if gt_xyz_m is None:
            return None, None
        e_m = float(np.linalg.norm(est_m - gt_xyz_m))
        return e_m, e_m * 100.0

    srp_err_m, srp_err_cm = _err(srp_estimated_m)
    tdoa_err_m, tdoa_err_cm = _err(tdoa_estimated_m)

    # --- Neural LocalizationCNNS2 (if trained artifact available) ---
    neural_payload: dict[str, object] = {"available": False}
    neural_estimated_m: np.ndarray | None = None

    if neural_artifact is not None and neural_artifact.is_file():
        try:
            neural_estimated_m = _run_neural_localization(
                mic_data=mic_data,
                fs=fs,
                srp_prior_m=srp_estimated_m,
                artifact_path=neural_artifact,
                c=c,
            )
            neural_estimated_cm = neural_estimated_m * 100.0
            neural_err_m, neural_err_cm = _err(neural_estimated_m)
            neural_payload = {
                "available": True,
                "method": "LocalizationCNNS2",
                "artifact": str(neural_artifact),
                "estimated_m": neural_estimated_m.tolist(),
                "estimated_cm": neural_estimated_cm.tolist(),
                "error_m": neural_err_m,
                "error_cm": neural_err_cm,
            }
        except Exception as exc:
            neural_payload = {"available": False, "error": str(exc)}

    # --- Fusion: combine SRP, TDOA, and neural estimates ---
    # TDOA is only trustworthy when it agrees with SRP (< 15 cm disagreement).
    # In reverberant conditions GCC argmax is fragile; blending a bad TDOA
    # degrades the final estimate.  When TDOA diverges we fall back to SRP.
    _TDOA_TRUST_THRESH_M = 0.15  # 15 cm
    tdoa_srp_dist_m = float(np.linalg.norm(tdoa_estimated_m - srp_estimated_m))
    tdoa_trusted = tdoa_srp_dist_m < _TDOA_TRUST_THRESH_M

    if neural_estimated_m is not None:
        if tdoa_trusted:
            fused_m = (
                0.2 * srp_estimated_m
                + 0.25 * tdoa_estimated_m
                + 0.55 * neural_estimated_m
            )
        else:
            fused_m = 0.35 * srp_estimated_m + 0.65 * neural_estimated_m
    else:
        if tdoa_trusted:
            fused_m = 0.40 * srp_estimated_m + 0.60 * tdoa_estimated_m
        else:
            fused_m = srp_estimated_m.copy()

    fused_cm = fused_m * 100.0
    fused_err_m, fused_err_cm = _err(fused_m)

    # Determine best method by error (only possible when GT is known)
    best_method = "srp_phat"
    best_err_cm: float | None = srp_err_cm

    def _update_best(method: str, err_cm: float | None) -> None:
        nonlocal best_method, best_err_cm
        if err_cm is not None and (best_err_cm is None or err_cm < best_err_cm):
            best_method = method
            best_err_cm = err_cm

    _update_best("tdoa", tdoa_err_cm)
    if neural_estimated_m is not None:
        _update_best("neural_cnn", float(neural_payload.get("error_cm", float("inf"))))  # type: ignore[arg-type]
    _update_best("fused", fused_err_cm)

    # Nearest known fault position lookup (on the best estimate)
    best_est_m = {
        "fused": fused_m,
        "neural_cnn": neural_estimated_m if neural_estimated_m is not None else fused_m,
        "tdoa": tdoa_estimated_m,
        "srp_phat": srp_estimated_m,
    }.get(best_method, fused_m)
    best_k: str | None = None
    best_d = float("inf")
    for k, known_m in S2_FAULT_POSITIONS_M.items():
        d = float(np.linalg.norm(best_est_m - known_m))
        if d < best_d:
            best_d = d
            best_k = k

    payload: dict[str, object] = {
        "status": "ok",
        "pos_folder": pos_dir.name,
        "n_mic_pairs": N_PAIRS_S2,
        "mic_channels_used": mic_data.shape[0],
        "operating_modes": modes,
        "ground_truth_cm": gt_xyz_cm.tolist() if gt_xyz_cm is not None else None,
        "ground_truth_m": gt_xyz_m.tolist() if gt_xyz_m is not None else None,
        # --- SRP-PHAT (improved: multi-window GCC + hierarchical grid) ---
        "srp_phat": {
            "method": "srp_phat_3d_multiwindow_hierarchical",
            "srp_peak_power": srp_peak,
            "estimated_m": srp_estimated_m.tolist(),
            "estimated_cm": srp_estimated_cm.tolist(),
            "error_m": srp_err_m,
            "error_cm": srp_err_cm,
            "grid_resolution_coarse_m": grid_resolution_m,
            "grid_resolution_fine_m": grid_resolution_m / 4.0,
        },
        # --- TDOA triangulation (non-linear least-squares over 10 pairs) ---
        "tdoa_triangulation": {
            "method": "tdoa_lbfgsb_10pairs",
            "estimated_m": tdoa_estimated_m.tolist(),
            "estimated_cm": tdoa_estimated_cm.tolist(),
            "error_m": tdoa_err_m,
            "error_cm": tdoa_err_cm,
            "residual_sum_sq": float(tdoa_residual),
        },
        # --- Neural CNN (if trained) ---
        "neural_cnn": neural_payload,
        # --- Fused estimate ---
        "fused": {
            "method": (
                "srp_tdoa_neural_weighted_avg"
                if (neural_estimated_m is not None and tdoa_trusted)
                else (
                    "srp_neural_weighted_avg"
                    if neural_estimated_m is not None
                    else "srp_tdoa_weighted_avg" if tdoa_trusted else "srp_phat_only"
                )
            ),
            "tdoa_trusted": tdoa_trusted,
            "tdoa_srp_disagreement_cm": round(tdoa_srp_dist_m * 100.0, 2),
            "estimated_m": fused_m.tolist(),
            "estimated_cm": fused_cm.tolist(),
            "error_m": fused_err_m,
            "error_cm": fused_err_cm,
        },
        # --- Summary: best method ---
        "best_method": best_method,
        "best_error_cm": best_err_cm,
        "nearest_known_position": best_k,
        "nearest_known_error_m": float(best_d) if best_d < float("inf") else None,
        "fs_hz": fs,
        # --- Legacy flat fields (backward compatibility, reflects best estimate) ---
        "method": best_method,
        "srp_peak_power": srp_peak,
        "estimated_m": best_est_m.tolist(),
        "estimated_cm": (best_est_m * 100.0).tolist(),
        "error_m": best_err_cm / 100.0 if best_err_cm is not None else None,
        "error_cm": best_err_cm,
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
    loc_epochs: int = 400,
    loc_lr: float = 1e-3,
    loc_batch_size: int = 16,
    loc_window_s: float = 1.0,
    loc_hop_s: float = 0.25,
    loc_geo_weight: float = 0.05,
    loc_patience: int = 100,
    loc_seed: int = 42,
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

    # Stage localization_train: train LocalizationCNNS2
    loc_artifact = artifacts_root / "localization_cnn_s2.pt"
    jobs.append(
        TrainJob(
            name="S2 LocalizationCNNS2 train",
            fn=lambda o=loc_artifact: _run_localization_train_job(
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
            description=f"train LocalizationCNNS2 -> {loc_artifact}",
            output_dir=str(artifacts_root),
            stage="localization_train",
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
        _neural_artifact = artifacts_root / "localization_cnn_s2.pt"
        jobs.append(
            TrainJob(
                name=f"S2 localize ({pos_dir.name})",
                fn=lambda p=pos_dir, o=loc_out, na=_neural_artifact: _run_localization_job(
                    pos_dir=p,
                    output_path=o,
                    grid_resolution_m=grid_resolution_m,
                    neural_artifact=na if na.is_file() else None,
                ),
                description=f"srp_phat+neural({pos_dir.name}) -> {loc_out}",
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
    # LocalizationCNNS2 training hyper-parameters
    parser.add_argument("--loc-epochs", type=int, default=400)
    parser.add_argument("--loc-lr", type=float, default=1e-3)
    parser.add_argument("--loc-batch-size", type=int, default=16)
    parser.add_argument(
        "--loc-window-s",
        type=float,
        default=1.0,
        help="GCC window length for localization training (s)",
    )
    parser.add_argument(
        "--loc-hop-s",
        type=float,
        default=0.25,
        help="Hop size for localization training (s)",
    )
    parser.add_argument(
        "--loc-geo-weight",
        type=float,
        default=0.05,
        help="Geometric consistency loss weight",
    )
    parser.add_argument("--loc-patience", type=int, default=100)
    parser.add_argument("--loc-seed", type=int, default=42)
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
        loc_epochs=int(args.loc_epochs),
        loc_lr=float(args.loc_lr),
        loc_batch_size=int(args.loc_batch_size),
        loc_window_s=float(args.loc_window_s),
        loc_hop_s=float(args.loc_hop_s),
        loc_geo_weight=float(args.loc_geo_weight),
        loc_patience=int(args.loc_patience),
        loc_seed=int(args.loc_seed),
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
