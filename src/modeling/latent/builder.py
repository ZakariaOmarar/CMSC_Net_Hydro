"""Build latent cache files (.npz) from raw recordings.

This module is the bridge between the signal-processing pipeline and the ML
models. For each recording it:
1. Scans and loads raw WAV + CSV pairs via the ingestion layer.
2. Applies the preprocessing stack (calibration, DC removal, bandpass, z-score).
3. Segments the preprocessed signal into sliding windows.
4. Runs each window through the configured feature extractors.
5. Assembles the feature rows into z (diagnostic features) and c (context
   features) arrays and writes them to a .npz file under artifacts/latents/.

Downstream anomaly and mode models consume these .npz latent files directly,
so all models share identical inputs regardless of architecture.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from ...features import (
    CrossChannelExtractor,
    FrequencyDomainExtractor,
    TimeDomainExtractor,
    VibrationFrequencyExtractor,
    VibrationEnvelopeExtractor,
    compute_cwt_scalogram_stack,
    compute_mfcc_stack,
    compute_stft_stack,
)
from ...ingestion import RecordingScanner, SegmentLoader
from ...ingestion.adapters import WavVibrationAdapter
from .preprocessing import (
    ModelVariant,
    build_segmenter,
    preprocess_segment,
    select_single_mic,
    validate_variant,
)


@dataclass(frozen=True)
class LatentBuildSummary:
    """Summary returned by build_latent_cache after processing one recording.

    Used by the orchestration layer to confirm the output shape and emit
    structured progress events.
    """

    recording_id: str
    output_path: Path
    n_windows: int
    d_model: int
    d_ctx: int


_VIB_FREQ_SUFFIXES: tuple[str, ...] = (
    "fundamental_amplitude",
    "vpf_amplitude",
    "harmonic_ratio_2x",
    "harmonic_ratio_3x",
    "harmonic_ratio_4x",
    "band_energy_200_1k",
    "band_energy_1k_5k",
)


def _resolve_transition_flag(metadata: dict) -> bool:
    explicit = metadata.get("is_transition_window")
    if explicit is not None:
        return bool(explicit)

    mask = metadata.get("transition_mask")
    idx = metadata.get("window_index")
    if mask is None or idx is None:
        return False

    try:
        i = int(idx)
        if i < 0:
            return False
        return bool(mask[i])
    except Exception:
        return False


def _pad_or_trim(arr: np.ndarray, *, size: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if x.shape[0] == size:
        return x
    if x.shape[0] > size:
        return x[:size]
    out = np.zeros((size,), dtype=np.float64)
    out[: x.shape[0]] = x
    return out


def _feature_vec(
    features: dict[str, float | np.ndarray],
    key: str,
    *,
    size: int,
) -> np.ndarray:
    if key not in features:
        return np.zeros((size,), dtype=np.float64)
    return _pad_or_trim(np.asarray(features[key], dtype=np.float64), size=size)


def _feature_scalar(features: dict[str, float | np.ndarray], key: str) -> float:
    if key not in features:
        return 0.0
    arr = np.asarray(features[key], dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _build_routing_feature_names(
    *,
    n_accels: int,
    n_mfcc: int,
    n_mic_pairs: int,
    n_accel_pairs: int,
) -> tuple[list[str], list[str]]:
    z_names: list[str] = []

    for ai in range(n_accels):
        z_names.extend([f"accel_{ai}_vibration_features_{j}" for j in range(18)])

    for ai in range(n_accels):
        z_names.extend([f"accel_{ai}_vib_freq_stats_{j}" for j in range(8)])

    for ai in range(n_accels):
        for suffix in _VIB_FREQ_SUFFIXES:
            z_names.append(f"accel_{ai}_{suffix}")
        z_names.append(f"accel_{ai}_zero_crossing_rate")

    z_names.extend([f"acoustic_mfcc_mean_{j}" for j in range(max(1, int(n_mfcc)))])
    z_names.extend([f"mic_coherence_per_pair_{j}" for j in range(n_mic_pairs)])
    z_names.extend([f"accel_correlation_matrix_{j}" for j in range(n_accel_pairs)])

    c_names = [
        "transition_max_abs_gradient",
        "transition_monotonic_up_flag",
        "transition_monotonic_down_flag",
        "transition_rolling_variance_delta",
        "vib_freq_p50",
        "vib_freq_slope",
        "is_transition_window",
    ]

    return z_names, c_names


def _route_window_features(
    *,
    features: dict[str, float | np.ndarray],
    acoustic_mfcc_vec: np.ndarray,
    metadata: dict,
    n_accels: int,
    n_mfcc: int,
    n_mic_pairs: int,
    n_accel_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    vib_env_rows: list[np.ndarray] = []
    vib_freq_rows: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []

    for ai in range(n_accels):
        env_key = f"accel_{ai}_vibration_features"
        env_vec = _feature_vec(features, env_key, size=18)
        vib_env_rows.append(env_vec)
        z_parts.append(env_vec)

    for ai in range(n_accels):
        vf_key = f"accel_{ai}_vib_freq_stats"
        vf_vec = _feature_vec(features, vf_key, size=8)
        vib_freq_rows.append(vf_vec)
        z_parts.append(vf_vec)

    for ai in range(n_accels):
        scalar_vals = [
            _feature_scalar(features, f"accel_{ai}_{suffix}")
            for suffix in _VIB_FREQ_SUFFIXES
        ]
        scalar_vals.append(_feature_scalar(features, f"accel_{ai}_zero_crossing_rate"))
        z_parts.append(np.asarray(scalar_vals, dtype=np.float64))

    mfcc_mean = _pad_or_trim(
        np.asarray(acoustic_mfcc_vec, dtype=np.float64),
        size=max(1, n_mfcc),
    )
    z_parts.append(mfcc_mean)

    mic_coh = _feature_vec(features, "mic_coherence_per_pair", size=n_mic_pairs)
    accel_corr = _feature_vec(features, "accel_correlation_matrix", size=n_accel_pairs)
    z_parts.append(mic_coh)
    z_parts.append(accel_corr)

    vib_env_mat = np.stack(vib_env_rows, axis=0) if vib_env_rows else np.zeros((0, 18))
    vib_freq_mat = (
        np.stack(vib_freq_rows, axis=0)
        if vib_freq_rows
        else np.zeros((0, 8), dtype=np.float64)
    )

    if vib_env_mat.shape[0] > 0:
        transition_block = vib_env_mat[:, 14:18]
        trans_max_abs_grad = float(np.mean(transition_block[:, 0]))
        trans_up = float(np.max(transition_block[:, 1]))
        trans_down = float(np.max(transition_block[:, 2]))
        trans_var_delta = float(np.mean(transition_block[:, 3]))
    else:
        trans_max_abs_grad = 0.0
        trans_up = 0.0
        trans_down = 0.0
        trans_var_delta = 0.0

    if vib_freq_mat.shape[0] > 0:
        vib_p50 = float(np.mean(vib_freq_mat[:, 5]))
        vib_slope = float(np.mean(vib_freq_mat[:, 7]))
    else:
        vib_p50 = 0.0
        vib_slope = 0.0

    transition_flag = 1.0 if _resolve_transition_flag(metadata) else 0.0

    c_vec = np.asarray(
        [
            trans_max_abs_grad,
            trans_up,
            trans_down,
            trans_var_delta,
            vib_p50,
            vib_slope,
            transition_flag,
        ],
        dtype=np.float64,
    )

    z_vec = np.concatenate(z_parts, axis=0).astype(np.float64)
    return z_vec, c_vec


def _compute_acoustic_tensor(
    mic_data: np.ndarray,
    sample_rate: int,
    *,
    acoustic_rep: str,
    n_scales: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if acoustic_rep == "cwt":
        return compute_cwt_scalogram_stack(
            mic_data,
            sample_rate,
            n_scales=n_scales,
        )
    if acoustic_rep == "mfcc":
        return compute_mfcc_stack(
            mic_data,
            sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    if acoustic_rep == "stft":
        return compute_stft_stack(
            mic_data,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    raise ValueError("acoustic_rep must be one of: cwt, mfcc, stft")


def build_latent_cache(
    *,
    data_root: Path,
    output_dir: Path,
    mode: str = "mic_vibration",
    mic_channel_index: int = 0,
    acoustic_rep: str = "mfcc",
    window_s: float = 5.0,
    overlap: float = 0.5,
    context_len: int = 16,
    n_scales: int = 64,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: str = "cpu",
    encoder_checkpoint: Path | None = None,
    reuse_existing: bool = True,
    log_progress: bool = True,
    adapter: WavVibrationAdapter | None = None,
) -> list[LatentBuildSummary]:
    # Reserved for CLI/API compatibility with the flow trainer.
    _ = context_len, device, encoder_checkpoint

    variant: ModelVariant = validate_variant(mode)
    scanner = RecordingScanner(Path(data_root))
    groups = scanner.scan_groups()
    if not groups:
        raise FileNotFoundError(f"No recording groups found under {data_root}")

    loader = SegmentLoader(adapter=adapter)
    segmenter = build_segmenter(window_s=window_s, overlap=overlap)
    vib_extractor = VibrationEnvelopeExtractor()
    vib_freq_extractor = VibrationFrequencyExtractor()
    td_extractor = TimeDomainExtractor(channels="accel")
    fd_extractor = FrequencyDomainExtractor(channels="accel")
    cross_extractor = CrossChannelExtractor()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[LatentBuildSummary] = []

    max_mics = 1 if variant == "mic_only" else max(len(g.mic_files) for g in groups)
    max_accels = max(len(g.vibration_files) for g in groups)
    max_mic_pairs = max(0, int(max_mics * (max_mics - 1) // 2))
    max_accel_pairs = max(0, int(max_accels * (max_accels - 1) // 2))
    z_feature_names, c_feature_names = _build_routing_feature_names(
        n_accels=int(max_accels),
        n_mfcc=max(1, int(n_mfcc)),
        n_mic_pairs=int(max_mic_pairs),
        n_accel_pairs=int(max_accel_pairs),
    )

    for gi, group in enumerate(groups):
        output_name = f"{gi:03d}_{group.source_dir.name}_{group.recording_id}.npz"
        output_path = output_dir / output_name

        if reuse_existing and output_path.exists():
            try:
                with np.load(output_path, allow_pickle=False) as blob:
                    z_existing = np.asarray(blob["z"])
                    c_existing = np.asarray(blob["c"])

                if (
                    z_existing.ndim == 2
                    and c_existing.ndim == 2
                    and z_existing.shape[0] == c_existing.shape[0]
                    and z_existing.shape[0] > 0
                ):
                    summaries.append(
                        LatentBuildSummary(
                            recording_id=group.recording_id,
                            output_path=output_path,
                            n_windows=int(z_existing.shape[0]),
                            d_model=int(z_existing.shape[1]),
                            d_ctx=int(c_existing.shape[1]),
                        )
                    )
                    if log_progress:
                        print(
                            json.dumps(
                                {
                                    "event": "latent_cache_reused",
                                    "recording_id": group.recording_id,
                                    "file": str(output_path),
                                    "n_windows": int(z_existing.shape[0]),
                                }
                            )
                        )
                    continue
            except Exception:
                pass

        started = time.perf_counter()
        if log_progress:
            print(
                json.dumps(
                    {
                        "event": "latent_build_start",
                        "recording_id": group.recording_id,
                        "index": gi,
                        "total_recordings": len(groups),
                        "acoustic_rep": acoustic_rep,
                    }
                )
            )

        raw_segment = loader.load_group(group)
        segment = preprocess_segment(raw_segment)
        windows = segmenter.segment(segment)
        if not windows:
            continue

        z_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        recording_ids: list[str] = []
        transition_flags: list[bool] = []

        for win in windows:
            model_win = (
                select_single_mic(win, mic_channel_index)
                if variant == "mic_only"
                else win
            )

            features: dict[str, float | np.ndarray] = {}
            for extractor in (
                vib_extractor,
                vib_freq_extractor,
                td_extractor,
                fd_extractor,
                cross_extractor,
            ):
                frame = extractor.extract(model_win)
                features.update(frame.features)

            try:
                acoustic_tensor = _compute_acoustic_tensor(
                    model_win.mic_data,
                    model_win.mic_sample_rate,
                    acoustic_rep=str(acoustic_rep),
                    n_scales=int(n_scales),
                    n_mfcc=max(1, int(n_mfcc)),
                    n_fft=int(n_fft),
                    hop_length=int(hop_length),
                )

                if str(acoustic_rep) == "mfcc":
                    acoustic_core = acoustic_tensor[:, : max(1, int(n_mfcc)), :]
                    acoustic_summary = np.mean(acoustic_core, axis=(0, 2))
                else:
                    acoustic_summary = np.mean(
                        np.asarray(acoustic_tensor, dtype=np.float64),
                        axis=(0, 2),
                    )

                acoustic_mfcc_vec = _pad_or_trim(
                    np.asarray(acoustic_summary, dtype=np.float64),
                    size=max(1, int(n_mfcc)),
                )
            except Exception:
                acoustic_mfcc_vec = np.zeros((max(1, int(n_mfcc)),), dtype=np.float64)

            z_vec, c_vec = _route_window_features(
                features=features,
                acoustic_mfcc_vec=acoustic_mfcc_vec,
                metadata=model_win.metadata,
                n_accels=int(max_accels),
                n_mfcc=max(1, int(n_mfcc)),
                n_mic_pairs=int(max_mic_pairs),
                n_accel_pairs=int(max_accel_pairs),
            )

            z_list.append(z_vec.astype(np.float32))
            recording_ids.append(group.recording_id)
            transition_flags.append(_resolve_transition_flag(model_win.metadata))
            c_list.append(c_vec.astype(np.float32))

        if not z_list:
            continue

        z_mat = np.stack(z_list, axis=0)
        c_mat = np.stack(c_list, axis=0)

        np.savez_compressed(
            output_path,
            z=z_mat,
            c=c_mat,
            recording_id=np.asarray(recording_ids, dtype=str),
            is_transition_window=np.asarray(transition_flags, dtype=bool),
            z_feature_names=np.asarray(z_feature_names, dtype=str),
            c_feature_names=np.asarray(c_feature_names, dtype=str),
        )

        summaries.append(
            LatentBuildSummary(
                recording_id=group.recording_id,
                output_path=output_path,
                n_windows=int(z_mat.shape[0]),
                d_model=int(z_mat.shape[1]),
                d_ctx=int(c_mat.shape[1]),
            )
        )

        if log_progress:
            elapsed_s = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "latent_build_done",
                        "recording_id": group.recording_id,
                        "file": str(output_path),
                        "n_windows": int(z_mat.shape[0]),
                        "elapsed_s": round(float(elapsed_s), 2),
                    }
                )
            )

    return summaries
