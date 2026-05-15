"""V4 trainer — supervised (x, y, z) regression on labeled anomaly windows.

Pipeline:
  1. **Precompute** V4 samples from labeled `TestDatasetSegment`s:
       - SRP-PHAT volume on a fixed grid (raw mic_data → `compute_srp_phat_volume`)
       - Structure-borne TDOA tokens (raw accel_data → `compute_accel_tdoa_tokens`)
       - V2 context vector `c_t` (V2 encoder forward on log-mel/CWT features)
       - Ground-truth `(x, y, z)` from the loader's `spatial_label`
  2. **Train** the V4 head with Smooth-L1 loss on `(x, y, z)`.  V2 is frozen
     by default; `cfg.unfreeze_v2_encoder=True` enables a fine-tune path.
  3. **Report** held-out 3-D Euclidean error (mean + 95th percentile).

Three knobs that matter for Chapter 6:
  - `cfg.unconditional=True` → A3 ablation (zero c at train+infer).
  - `cfg.scada_dim>0`        → SCADA injection slot for V5.1 (D3 speed one-hot)
                                and V5.2 (top-K Allg_M1).
  - `gated_inference`        → infer only on windows flagged anomalous by V3
                                (cost/quality study; a Chapter 6 deployment row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...features.audio_spectral import compute_encoder_input_stack, compute_log_mel_spectrogram
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import TestDatasetSegment
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import V2SSLConfig, _dataset_idx
from .v4_features import (
    GridSpec,
    compute_accel_tdoa_tokens,
    compute_burst_aware_srp_phat_volume,
    compute_srp_phat_volume,
)
from .v4_loc_head import V4LocalizationHead


# ---------------------------------------------------------------------------
# Sample + config
# ---------------------------------------------------------------------------


@dataclass
class V4Sample:
    """One labeled anomaly window."""

    srp_volume: np.ndarray  # (Nx, Ny, Nz)
    tdoa_tokens: np.ndarray  # (n_pairs, 8); n_pairs may be 0
    context: np.ndarray  # (c_dim,) — V2 c_t (PMA pool of fused tokens)
    x_for_v3: np.ndarray  # (c_dim,) — mean-pool of fused tokens (V3 flow input)
    target_xyz: np.ndarray  # (3,) metres
    scada: np.ndarray | None  # (s_dim,) or None
    mode_label: str | None
    recording_id: str
    source_dir: str
    dataset_id: str  # so V3-gated cohort assembly can dispatch by source campaign


@dataclass(frozen=True)
class V4Config:
    """V4 training config."""

    # Head dims
    cnn_feature_dim: int = 128
    tdoa_feature_dim: int = 64
    hidden_dim: int = 128
    n_heads_tdoa: int = 2

    # Heatmap soft-argmax + FiLM-residual head
    soft_argmax_temperature: float = 1.0
    # Residual half-range in metres.  Was 0.05 in the second-pass audit;
    # the 2026-05-11 run showed soft-argmax centre-bias of 10–15 cm on
    # corner-of-grid positions (e.g. D4 `(-20, 0, 0)`) that the ±5 cm
    # residual could not correct.  0.20 m lets the head reach any voxel
    # in the prototype bounding box while the tanh squash + AdamW weight
    # decay still prefer small residuals when soft-argmax is accurate.
    residual_scale_m: float = 0.20

    # Conditioning
    scada_dim: int = 0  # 0 → no SCADA slot; V5.1/V5.2 set this.
    unconditional: bool = False  # A3 ablation
    # Channel-ablation modes (A5 in REVIEW.md sixth-pass audit):
    #   - "both": full V4 architecture (acoustic SRP + structure-borne TDOA).
    #   - "srp_only": zero the TDOA tokens at inference and training, so the
    #     head must regress from the SRP volume + FiLM(c) alone.
    #   - "tdoa_only": zero the SRP volume at inference and training, so the
    #     head's soft-argmax starts at the grid centroid for every window
    #     and the localization comes entirely from the FiLM(c)-conditioned
    #     residual on the TDOA tokens.
    # This ablation isolates the marginal contribution of the structure-
    # borne accelerometer TDOA pathway from the acoustic SRP pathway —
    # i.e. it answers "which modality is doing the localization work?"
    # at the V4-head input level, complementary to the A1 modality-severing
    # ablation that lives in V2.
    channel_mode: Literal["both", "srp_only", "tdoa_only"] = "both"

    # Training
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.3
    seed: int = 42
    device: str = "cpu"

    # Loss / training-time augmentation
    # Targets are in metres but on a ~10 cm prototype the smooth-L1 loss with
    # default beta=1.0 sits in the quadratic regime where ‖err‖² → 0 fast and
    # gradients vanish.  Switching the regression problem to centimetres
    # (multiply pred and target by 100 inside the loop) puts a 1 cm error at
    # loss ≈ 1.0 — well inside the linear part of smooth-L1 — and recovers
    # full gradient signal.  This is purely a numerical-conditioning fix and
    # does not change the inference output (we scale back to metres for MAE).
    train_in_centimetres: bool = True
    smooth_l1_beta: float = 1.0  # in the post-scale unit (cm if scaling on)
    # Augmentation — addresses the 6-recording labeled pool.  All three are
    # zero-mean perturbations applied only at training time:
    target_pos_noise_m: float = 0.002      # ± 2 mm Gaussian on the ground-truth
    srp_volume_noise_std: float = 0.02     # additive Gaussian on the SRP volume
    srp_volume_dropout_p: float = 0.0      # not used; reserved for mic-dropout aug
    tdoa_jitter_m: float = 0.001           # ± 1 mm Gaussian on path_diff_m
    augment: bool = True


# ---------------------------------------------------------------------------
# Sample precomputation
# ---------------------------------------------------------------------------


def _window_v2_features(
    segment: TestDatasetSegment,
    cfg: V2SSLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the V2 paired feature stacks for the segment."""
    if cfg.use_cwt:
        ac = compute_encoder_input_stack(
            segment.segment.mic_data,
            fs=int(segment.segment.mic_sample_rate),
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            cwt_n_scales=cfg.cwt_n_scales,
        )
    else:
        mels = []
        for ch in range(segment.segment.n_mic_channels):
            m = compute_log_mel_spectrogram(
                segment.segment.mic_data[ch],
                fs=int(segment.segment.mic_sample_rate),
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
            )
            mels.append(np.stack([m, m], axis=0).astype(np.float32))
        ac = np.stack(mels, axis=0)
    vib = compute_vibration_input_stack(
        segment.segment.accel_data, kurtosis_window=cfg.vib_kurtosis_window
    )
    return ac.astype(np.float32), vib.astype(np.float32)


def precompute_v4_samples(
    v2_encoder: V2FusionEncoder,
    segments: list[TestDatasetSegment],
    *,
    v2_cfg: V2SSLConfig,
    grid: GridSpec,
    spatial_label_overrides: dict[str, tuple[float, float, float]] | None = None,
    scada_lookup: dict[str, np.ndarray] | None = None,
    window_seconds: float | None = None,
    window_stride_seconds: float | None = None,
    burst_aware_srp: bool = False,
    burst_seconds: float = 0.10,
    device: torch.device | str = "cpu",
) -> list[V4Sample]:
    """Walk segments, slice labeled anomaly windows, build `V4Sample`s.

    - Spatial labels come from `segment.spatial_label` if present, with an
      optional `spatial_label_overrides` dict keyed by `recording_id` for
      smoke-test synthesis or for D3 `hit_between_*` recordings whose
      coordinates the loader reports as `None`.
    - SCADA values come from `scada_lookup[recording_id]` when provided
      (V5.1 D3 speed one-hot, V5.2 top-K Allg_M1).  Missing → no SCADA slot
      (the head is built without one).
    - Window cadence defaults to `v2_cfg.window_seconds` / `window_stride_seconds`
      so V2 features and SRP-PHAT/TDOA front-ends operate on the same window
      boundaries.
    """
    device = torch.device(device)
    v2_encoder = v2_encoder.to(device).eval()
    for p in v2_encoder.parameters():
        p.requires_grad_(False)

    win_s = window_seconds if window_seconds is not None else v2_cfg.window_seconds
    stride_s = (
        window_stride_seconds if window_stride_seconds is not None else v2_cfg.window_stride_seconds
    )
    overrides = spatial_label_overrides or {}

    samples: list[V4Sample] = []
    for s in segments:
        spatial = overrides.get(s.recording_id, s.spatial_label)
        if spatial is None:
            continue  # no ground-truth → skip (V4 needs labels)
        target = np.asarray(spatial, dtype=np.float32)

        scada = None
        if scada_lookup is not None and s.recording_id in scada_lookup:
            scada = np.asarray(scada_lookup[s.recording_id], dtype=np.float32)

        # Pre-compute V2 paired features once per segment.
        ac_feats, vib_feats = _window_v2_features(s, v2_cfg)

        ac_fs = float(s.segment.mic_sample_rate) / float(v2_cfg.hop_length)
        vib_fs = float(s.segment.accel_sample_rate)

        # Window counts (V2 frame-aligned).
        n_ac = max(2, int(round(win_s * ac_fs)))
        stride_ac = max(1, int(round(stride_s * ac_fs)))
        n_vib = max(2, int(round(win_s * vib_fs)))

        # Raw waveform window (used for SRP-PHAT + TDOA).
        mic_fs_raw = int(s.segment.mic_sample_rate)
        accel_fs_raw = int(s.segment.accel_sample_rate)
        n_mic_raw = max(8, int(round(win_s * mic_fs_raw)))
        n_acc_raw = max(2, int(round(win_s * accel_fs_raw)))

        T_ac = ac_feats.shape[-1]
        T_vib = vib_feats.shape[-1]
        T_mic = s.segment.mic_data.shape[1]
        T_acc = s.segment.accel_data.shape[1]

        if T_ac < n_ac or T_vib < n_vib or T_mic < n_mic_raw or T_acc < n_acc_raw:
            continue

        ds_idx = torch.tensor(
            [_dataset_idx(s.dataset_id)], dtype=torch.long, device=device
        )

        for start_ac in range(0, T_ac - n_ac + 1, stride_ac):
            t_start = start_ac / max(ac_fs, 1e-9)
            start_vib = int(round(t_start * vib_fs))
            if start_vib + n_vib > T_vib:
                continue
            start_mic = int(round(t_start * mic_fs_raw))
            start_acc = int(round(t_start * accel_fs_raw))
            if start_mic + n_mic_raw > T_mic or start_acc + n_acc_raw > T_acc:
                continue

            # V2 forward → c_t
            ac_win = torch.from_numpy(
                np.ascontiguousarray(ac_feats[..., start_ac : start_ac + n_ac])
            ).unsqueeze(0).float().to(device)
            vib_win = torch.from_numpy(
                np.ascontiguousarray(vib_feats[..., start_vib : start_vib + n_vib])
            ).unsqueeze(0).float().to(device)
            ac_xyz = torch.from_numpy(s.mic_positions.astype(np.float32)).unsqueeze(0).to(device)
            vib_xyz = torch.from_numpy(s.vib_positions.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = v2_encoder(ac_win, ac_xyz, vib_win, vib_xyz, ds_idx, mask_p=0.0)
            c_t = out["context"].squeeze(0).cpu().numpy().astype(np.float32)
            # V3's flow input: mean-pool of fused tokens (the same `x` V3
            # consumes during training).  Cached on the V4Sample so V3-
            # gated cohort assembly doesn't have to re-run the encoder.
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            x_for_v3 = fused.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

            # Front-end features on the raw waveform window.  Burst-aware
            # SRP crops to the highest-energy ~100 ms sub-window before
            # computing GCC-PHAT — sharper SRP peak on sparsely-anomalous
            # recordings (D4 RandomFault knocks), no worse on continuously-
            # anomalous recordings (D2 RandomFault, D3 hit).
            mic_seg = s.segment.mic_data[:, start_mic : start_mic + n_mic_raw]
            acc_seg = s.segment.accel_data[:, start_acc : start_acc + n_acc_raw]
            if burst_aware_srp:
                volume = compute_burst_aware_srp_phat_volume(
                    mic_seg, s.mic_positions, fs=mic_fs_raw, grid=grid,
                    burst_seconds=burst_seconds,
                )
            else:
                volume = compute_srp_phat_volume(
                    mic_seg, s.mic_positions, fs=mic_fs_raw, grid=grid,
                )
            tdoa = compute_accel_tdoa_tokens(
                acc_seg, s.vib_positions, fs=accel_fs_raw
            )

            samples.append(
                V4Sample(
                    srp_volume=volume,
                    tdoa_tokens=tdoa,
                    context=c_t,
                    x_for_v3=x_for_v3,
                    target_xyz=target,
                    scada=scada,
                    mode_label=s.mode_label,
                    recording_id=s.recording_id,
                    source_dir=str(s.source_dir),
                    dataset_id=s.dataset_id,
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------


def _split_samples_by_recording(
    samples: list[V4Sample], val_ratio: float, seed: int
) -> tuple[list[V4Sample], list[V4Sample]]:
    rng = np.random.default_rng(seed)
    keys = sorted({(s.source_dir, s.recording_id) for s in samples})
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train_keys = set(keys[n_val:])
    if not train_keys:
        train_keys = {val_keys.pop()}
    train = [s for s in samples if (s.source_dir, s.recording_id) in train_keys]
    val = [s for s in samples if (s.source_dir, s.recording_id) in val_keys]
    return train, val


def _stack_tdoa(samples: list[V4Sample]) -> torch.Tensor:
    """Pad TDOA token sequences to the batch's max `n_pairs` with zeros."""
    n_max = max((s.tdoa_tokens.shape[0] for s in samples), default=0)
    out = np.zeros((len(samples), max(n_max, 1), 8), dtype=np.float32)
    if n_max > 0:
        for i, s in enumerate(samples):
            n = s.tdoa_tokens.shape[0]
            if n > 0:
                out[i, :n] = s.tdoa_tokens
    return torch.from_numpy(out)


def _make_batch(
    samples: list[V4Sample],
    *,
    channel_mode: Literal["both", "srp_only", "tdoa_only"] = "both",
) -> dict:
    volumes = torch.from_numpy(np.stack([s.srp_volume for s in samples], axis=0)).float()
    tdoa = _stack_tdoa(samples).float()
    if channel_mode == "srp_only":
        # Severing the structure-borne TDOA pathway: zero the token
        # features (path_diff_m + endpoint coords + distance) so the
        # TDOA-set encoder reads a no-information input.  The encoder's
        # PMA output is still computed (so the head's input dim is
        # preserved), but it carries no spatial information.
        tdoa = torch.zeros_like(tdoa)
    elif channel_mode == "tdoa_only":
        # Severing the acoustic-SRP pathway: zero the SRP volume so the
        # 3-D CNN sees a flat-zero input.  The soft-argmax over a flat
        # logit volume collapses to the uniform centroid of the grid →
        # init_xyz is the grid centre on every window, and any
        # localization signal must come through the FiLM-conditioned
        # residual on (global_feat, tdoa_feat, init_xyz).
        volumes = torch.zeros_like(volumes)
    contexts = torch.from_numpy(np.stack([s.context for s in samples], axis=0)).float()
    targets = torch.from_numpy(np.stack([s.target_xyz for s in samples], axis=0)).float()
    scada = None
    if all(s.scada is not None for s in samples) and samples:
        scada = torch.from_numpy(np.stack([s.scada for s in samples], axis=0)).float()
    return {
        "volumes": volumes,
        "tdoa": tdoa,
        "contexts": contexts,
        "targets": targets,
        "scada": scada,
    }


@dataclass
class V4Result:
    head: V4LocalizationHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    val_mae_3d: float
    val_p95_3d: float
    val_predictions: np.ndarray  # (n_val, 3)
    val_targets: np.ndarray  # (n_val, 3)
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    unconditional: bool
    # Diagnostics for Chapter 6 / REVIEW.md analysis.
    val_init_xyz: np.ndarray  # (n_val, 3) — pure soft-argmax output
    val_residuals: np.ndarray  # (n_val, 3) — FiLM-residual contribution
    val_recording_breakdown: dict  # recording_id -> {n, mae, target, pred_mean}
    # Bootstrap 95 % CI on val MAE — percentile method, 1000 resamples at
    # the window level.  With ~ 3 val recordings and ~ 300 val windows
    # the single-point MAE has high variance; reporting `MAE ± CI` is
    # the standard small-sample regression discipline (Efron & Tibshirani,
    # 1993, "An Introduction to the Bootstrap").
    val_mae_ci_low: float = float("nan")
    val_mae_ci_high: float = float("nan")
    val_mae_ci_method: str = "percentile_bootstrap_1000"


def _grid_coords_from_spec(grid: GridSpec) -> torch.Tensor:
    """Build the `(Nx, Ny, Nz, 3)` voxel-centre tensor from a `GridSpec`."""
    ax_x, ax_y, ax_z = grid.axes()
    xx, yy, zz = np.meshgrid(ax_x, ax_y, ax_z, indexing="ij")
    coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)  # (Nx, Ny, Nz, 3)
    return torch.from_numpy(coords)


def train_v4_localization(
    samples: list[V4Sample],
    cfg: V4Config | None = None,
    *,
    grid: GridSpec,
) -> V4Result:
    """Train the V4 head supervised on labeled anomaly windows.

    `grid` must match the `GridSpec` used at sample-precompute time —
    the head's soft-argmax operates on its voxel centres, so a mismatch
    silently corrupts the regression targets.
    """
    cfg = cfg or V4Config()
    if not samples:
        raise RuntimeError("V4: no labeled anomaly samples")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    train_samples, val_samples = _split_samples_by_recording(samples, cfg.val_ratio, cfg.seed)

    c_dim = int(samples[0].context.shape[0])
    s_dim = cfg.scada_dim
    grid_coords = _grid_coords_from_spec(grid)
    head = V4LocalizationHead(
        grid_coords=grid_coords,
        cnn_feature_dim=cfg.cnn_feature_dim,
        tdoa_feature_dim=cfg.tdoa_feature_dim,
        c_dim=c_dim,
        s_dim=s_dim,
        hidden_dim=cfg.hidden_dim,
        n_heads_tdoa=cfg.n_heads_tdoa,
        residual_scale_m=cfg.residual_scale_m,
        soft_argmax_temperature=cfg.soft_argmax_temperature,
    ).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine LR schedule — small head + small labeled pool benefits from a
    # smooth lr decay rather than fixed-rate AdamW.  T_max = cfg.epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, int(cfg.epochs)), eta_min=cfg.lr * 0.01
    )

    n_train = len(train_samples)
    train_history: list[float] = []
    val_history: list[float] = []

    def _to_device(batch: dict) -> dict:
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    loss_scale = 100.0 if cfg.train_in_centimetres else 1.0  # 1 m → 100 cm

    def _augment_batch(batch: dict, gen: torch.Generator) -> dict:
        """Apply zero-mean noise to volumes / TDOA tokens / targets in place.

        Volume noise mimics SRP estimator variance from waveform jitter
        without re-running SRP.  TDOA-token noise (on the path-difference
        column only) mimics structure-borne speed uncertainty.  Target
        noise (≤ 2 mm) regularises the regression head against the small
        spatial-label inventory (~ 6 recordings) and is well below the
        physical position-measurement uncertainty.
        """
        if not cfg.augment:
            return batch
        vol = batch["volumes"]
        if cfg.srp_volume_noise_std > 0:
            noise = torch.randn(vol.shape, generator=gen, device=vol.device, dtype=vol.dtype)
            vol = (vol + cfg.srp_volume_noise_std * noise).clamp_min(0.0)
            batch["volumes"] = vol
        if cfg.tdoa_jitter_m > 0 and batch["tdoa"].numel() > 0:
            tdoa = batch["tdoa"].clone()
            jitter = torch.randn(
                tdoa.shape[0], tdoa.shape[1],
                generator=gen, device=tdoa.device, dtype=tdoa.dtype,
            ) * cfg.tdoa_jitter_m
            # Only the first feature is path_diff_m — leave positions / dist alone.
            tdoa[..., 0] = tdoa[..., 0] + jitter
            batch["tdoa"] = tdoa
        if cfg.target_pos_noise_m > 0:
            t = batch["targets"]
            tnoise = torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
            batch["targets"] = t + cfg.target_pos_noise_m * tnoise
        return batch

    aug_gen = torch.Generator(device="cpu")
    aug_gen.manual_seed(cfg.seed)

    suffix = "unconditional" if cfg.unconditional else f"scada_dim={cfg.scada_dim}"
    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V4 head ({suffix})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        head.train()
        perm = list(range(n_train))
        np.random.shuffle(perm)
        loss_sum = 0.0
        n = 0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            batch = _to_device(
                _augment_batch(_make_batch([train_samples[j] for j in idx], channel_mode=cfg.channel_mode), aug_gen)
            )
            pred = head(
                batch["volumes"],
                batch["tdoa"],
                batch["contexts"],
                batch["scada"],
                unconditional=cfg.unconditional,
            )
            loss = F.smooth_l1_loss(
                pred * loss_scale, batch["targets"] * loss_scale, beta=cfg.smooth_l1_beta
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * pred.shape[0]
            n += pred.shape[0]
        scheduler.step()
        train_history.append(loss_sum / max(1, n))

        head.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for i in range(0, len(val_samples), cfg.batch_size):
                batch = _to_device(_make_batch(val_samples[i : i + cfg.batch_size], channel_mode=cfg.channel_mode))
                pred = head(
                    batch["volumes"],
                    batch["tdoa"],
                    batch["contexts"],
                    batch["scada"],
                    unconditional=cfg.unconditional,
                )
                loss = F.smooth_l1_loss(
                    pred * loss_scale, batch["targets"] * loss_scale, beta=cfg.smooth_l1_beta
                )
                v_loss += float(loss.item()) * pred.shape[0]
                v_n += pred.shape[0]
        val_history.append(v_loss / max(1, v_n))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.4f}", val=f"{val_history[-1]:.4f}"
        )

    # Final val errors + per-recording diagnostic.  `return_components=True`
    # exposes (init_xyz, delta, pred) so Chapter 6 can attribute MAE to the
    # SRP-soft-argmax prior vs the FiLM-conditioned residual.
    head.eval()
    val_preds: list[np.ndarray] = []
    val_inits: list[np.ndarray] = []
    val_deltas: list[np.ndarray] = []
    val_tgts: list[np.ndarray] = []
    val_keys: list[str] = []
    with torch.no_grad():
        for i in range(0, len(val_samples), cfg.batch_size):
            batch_samples = val_samples[i : i + cfg.batch_size]
            batch = _to_device(_make_batch(batch_samples, channel_mode=cfg.channel_mode))
            out = head(
                batch["volumes"],
                batch["tdoa"],
                batch["contexts"],
                batch["scada"],
                unconditional=cfg.unconditional,
                return_components=True,
            )
            val_preds.append(out["pred"].cpu().numpy())
            val_inits.append(out["init_xyz"].cpu().numpy())
            val_deltas.append(out["delta"].cpu().numpy())
            val_tgts.append(batch["targets"].cpu().numpy())
            for s in batch_samples:
                val_keys.append(f"{Path(s.source_dir).name}/{s.recording_id}")

    if val_preds:
        val_predictions = np.concatenate(val_preds, axis=0)
        val_init = np.concatenate(val_inits, axis=0)
        val_delta = np.concatenate(val_deltas, axis=0)
        val_targets = np.concatenate(val_tgts, axis=0)
        errs = np.linalg.norm(val_predictions - val_targets, axis=-1)
        val_mae = float(errs.mean())
        val_p95 = float(np.percentile(errs, 95))
        # 95 % percentile-bootstrap CI on MAE, 1000 window-level resamples.
        # Standard small-sample uncertainty quantification on a regression
        # head (Efron & Tibshirani, 1993).  Resamples at the window level
        # rather than the recording level because window-level resampling
        # treats every val window as an independent draw from the head's
        # output distribution; recording-level CI would require ≥ 10 val
        # recordings, which the V4 cohort does not have.
        rng_ci = np.random.default_rng(cfg.seed)
        n_boot = 1000
        n_err = errs.shape[0]
        if n_err >= 2:
            boot_maes = np.empty(n_boot, dtype=np.float64)
            for i in range(n_boot):
                idx = rng_ci.integers(0, n_err, size=n_err)
                boot_maes[i] = float(errs[idx].mean())
            ci_low = float(np.percentile(boot_maes, 2.5))
            ci_high = float(np.percentile(boot_maes, 97.5))
        else:
            ci_low = ci_high = val_mae
        # Per-recording breakdown: mean(pred) vs target, recording-level MAE.
        breakdown: dict = {}
        for k in set(val_keys):
            mask = np.array([1 if vk == k else 0 for vk in val_keys], dtype=bool)
            if not mask.any():
                continue
            preds_k = val_predictions[mask]
            tgts_k = val_targets[mask]
            errs_k = np.linalg.norm(preds_k - tgts_k, axis=-1)
            breakdown[k] = {
                "n": int(mask.sum()),
                "mae_3d": float(errs_k.mean()),
                "p95_3d": float(np.percentile(errs_k, 95)),
                "target_xyz": tgts_k[0].astype(float).tolist(),
                "pred_xyz_mean": preds_k.mean(axis=0).astype(float).tolist(),
                "init_xyz_mean": val_init[mask].mean(axis=0).astype(float).tolist(),
                "delta_xyz_mean": val_delta[mask].mean(axis=0).astype(float).tolist(),
            }
    else:
        val_predictions = np.zeros((0, 3), dtype=np.float32)
        val_init = np.zeros((0, 3), dtype=np.float32)
        val_delta = np.zeros((0, 3), dtype=np.float32)
        val_targets = np.zeros((0, 3), dtype=np.float32)
        val_mae = float("nan")
        val_p95 = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        breakdown = {}

    def _qualify(s: V4Sample) -> str:
        return f"{Path(s.source_dir).name}/{s.recording_id}"

    return V4Result(
        head=head,
        train_loss_history=train_history,
        val_loss_history=val_history,
        val_mae_3d=val_mae,
        val_p95_3d=val_p95,
        val_predictions=val_predictions,
        val_targets=val_targets,
        train_recording_ids=sorted({_qualify(s) for s in train_samples}),
        val_recording_ids=sorted({_qualify(s) for s in val_samples}),
        unconditional=cfg.unconditional,
        val_init_xyz=val_init,
        val_residuals=val_delta,
        val_recording_breakdown=breakdown,
        val_mae_ci_low=ci_low,
        val_mae_ci_high=ci_high,
    )


__all__ = [
    "V4Config",
    "V4Result",
    "V4Sample",
    "precompute_v4_samples",
    "train_v4_localization",
]
