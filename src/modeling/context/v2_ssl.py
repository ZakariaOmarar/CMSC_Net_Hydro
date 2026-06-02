"""V2 multimodal SSL trainer — bidirectional cross-attention + Latent Masked Modeling.

Inherits V1 weights for both per-modality encoders and trains the fusion
block + context PMA jointly under two equally-weighted losses:

  1. **SimCLR contrastive on `c_t`** — two augmented views of each paired
     window pass through the full pipeline; the NT-Xent loss pulls the two
     views' context vectors together.
  2. **Latent Masked Modeling (LMM)** — at training time, a fraction of
     per-modality tokens (acoustic and vibration, independently) are replaced
     by a learned mask token before the cross-attention block.  The fused
     output at masked positions is regressed back to the **pre-mask
     per-modality tokens** via a cosine-similarity loss.  This is *not* raw
     cross-modal reconstruction (no scalogram pixels or CSV values are
     decoded) — it stays entirely in latent space, per the supervisor's note.

RQ1 metric: K-means(k=4) on `c_t` Hungarian-matched to D1+D2 folder labels.
The same evaluation function powers the A1 ablation (vibration-severed):
when `cfg.drop_vibration=True`, the vibration-feature inputs are zeroed at
the input — the cross-attention block + PMA still run, so the architecture
stays identical, but no information flows from the vibration branch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...config import describe_device, resolve_device
from ...config.architecture import (
    ACOUSTIC_CWT,
    ACOUSTIC_FEATURES,
    ENCODER,
    VIBRATION_FEATURES,
    WINDOWING,
)
from ...config.dataset_registry import REGISTRY
from ...features.audio_spectral import compute_encoder_input_stack, compute_log_mel_spectrogram
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import TestDatasetLoader, TestDatasetSegment
from .cluster_metric import cluster_purity_and_nmi
from .v2_fusion import V2FusionEncoder


def _registry_window_scales() -> dict[str, tuple[float, ...]]:
    """Per-dataset multi-scale window cadence sourced from the registry."""
    return {m.id: m.window_scales_seconds for m in REGISTRY}


# Note: `hop_for_dataset` removed 2026-05-21 after the empirical grid sweep
# (scripts/hop_length_study/analyze_hop_length_full_grid.py + chapter 3 §3.4.2) found that
# hop_length contributes < 0.005 ROC AUC across the [32, 4096] range once
# (n_fft, n_mels) are set correctly.  The previous per-dataset hop plumbing
# was based on a "cross-modal alignment" intuition that the empirical test
# refuted.  A single global hop_length=2048 (from ACOUSTIC_FEATURES) is now
# the architectural choice on all datasets.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V2SSLConfig:
    # Defaults sourced from `src.config.architecture` — change there, not here.

    # Window cadence (wall-clock seconds).  The legacy single-scale knobs
    # remain authoritative when `window_scales_seconds` is the empty tuple
    # (byte-equivalent to pre-2026-05-19).
    window_seconds: float = WINDOWING.window_seconds
    window_stride_seconds: float = WINDOWING.window_stride_seconds

    # Multi-scale window cadence (see `V1SSLConfig` for the full
    # justification).  When the per-dataset dict is populated, each batch
    # is single-dataset × single-scale via the grouped batch sampler's
    # `(channel_count, n_frames_ac, n_frames_vib)` bucket key.  Both
    # `n_ac` and `n_vib` are computed from the SAME per-batch scale so
    # the wall-clock pairing in `_PairedWindowedDataset` is preserved.
    window_scales_seconds: tuple[float, ...] = ()
    window_scales_seconds_per_dataset: dict[str, tuple[float, ...]] = field(
        default_factory=_registry_window_scales
    )
    window_scale_strategy: Literal["fixed", "uniform"] = WINDOWING.window_scale_strategy
    window_stride_ratio: float = WINDOWING.window_stride_ratio

    # Encoder dims (must match V1 checkpoints when loaded)
    feature_dim: int = ENCODER.feature_dim
    embed_dim: int = ENCODER.embed_dim
    n_heads: int = ENCODER.n_heads
    proj_dim: int = ENCODER.proj_dim

    # Training schedule — per-experiment, not centralised.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1
    val_ratio: float = 0.3

    # Acoustic features
    n_mels: int = ACOUSTIC_FEATURES.n_mels
    n_fft: int = ACOUSTIC_FEATURES.n_fft
    hop_length: int = ACOUSTIC_FEATURES.hop_length
    cwt_n_scales: int = ACOUSTIC_CWT.n_scales
    use_cwt: bool = True
    # F4 toggle — see `V1SSLConfig.standardize_acoustic`.  Must match the
    # V1 setting used to pretrain the inherited encoder weights, otherwise
    # the V2 fusion block sees an acoustic input distribution the V1 CNN
    # was never trained on.  Default False (F4 found not load-bearing).
    standardize_acoustic: bool = False

    # Vibration features — physical-time window with per-dataset
    # channel-2 statistic (kurtosis vs crest factor); see
    # `compute_vibration_input_stack` for the justification.
    vib_kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds
    vib_min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples
    vib_crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds
    vib_min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples
    # Vibration amplitude + envelope z-score — see
    # `V1SSLConfig.standardize_vibration`.
    standardize_vibration: bool = True

    # R1a — Acoustic2DCNN channel-width multiplier; must match the V1 value
    # used to pretrain the inherited V1-acoustic weights or
    # `load_v1_weights(strict=True)` will fail.  See
    # `V1SSLConfig.acoustic_cnn_width_mult`.
    acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult

    # Augmentations (feature space) — per-experiment knobs.
    gain_jitter_db: float = 6.0
    channel_dropout_p: float = 0.2
    spec_augment_freq_mask: int = 6
    spec_augment_time_mask: int = 8

    # Latent Masked Modeling — per-experiment knobs.
    lmm_mask_p: float = 0.3
    lmm_weight: float = 1.0

    # Modality dropout — independent per-modality dropout probabilities,
    # applied per batch before the cross-attention block.  Asymmetric by
    # default: the V1 / V2 evaluations show acoustic is the strong
    # mode-discriminator (purity 0.727) while vibration is weak (0.572)
    # and hurts fusion (V2 purity 0.612 < V1-acoustic 0.727).  Dropping
    # vibration more often than acoustic teaches the fusion block to
    # treat acoustic as the trunk and vibration as auxiliary, recovering
    # the V1-acoustic ceiling rather than the unweighted-mean of the two.
    # The legacy `modality_dropout_p` field still exists for back-compat:
    # when both new fields are 0.0 it falls back to the symmetric 50/50
    # coin-flip with that probability.
    # Modality dropout — asymmetric: vibration dropped 50 % to stop the
    # fusion block from diluting the dominant acoustic mode-discriminator.
    modality_dropout_p: float = 0.0  # legacy symmetric fallback
    acoustic_dropout_p: float = 0.0
    vibration_dropout_p: float = 0.5

    # Cross-modal alignment (CMA) — NT-Xent between per-modality PMA
    # summaries.  R1c publication run uses cma_weight = 0.5.
    cma_weight: float = 0.0
    cma_temperature: float = 0.1

    # Context-aggregation mode for c_t — see V2FusionEncoder.ContextMode.
    context_mode: str = "joint_pma"

    # Number of learned PMA seeds in the V2 joint-context pool — the
    # ARCHITECTURAL choice (Set Transformer §3.2, Lee et al. ICML 2019).
    num_context_seeds: int = ENCODER.num_context_seeds

    # Ablation A1 — drop vibration branch (zero its features at the input)
    drop_vibration: bool = False

    # Labels — healthy operating modes only.  RandomFault must not enter
    # SSL training (label-leakage invariant).
    healthy_modes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
        "Healthy",
    )

    # Early stopping on val total-loss (simclr + lmm*w + cma*w).  See V1SSLConfig
    # for the snapshot-strategy rationale (tensor-clone, not deepcopy).
    patience: int = 3
    restore_best: bool = True
    early_stop_min_delta: float = 1e-4

    # Cross-recording mixup — see V1SSLConfig docstring for the design.  V2
    # blends BOTH modalities with the SAME λ from the SAME partner recording,
    # preserving the cross-modal pairing inside each mixed window.  Default
    # 0.0 disables; recommended ablation values: {0.0, 0.2, 0.4}.
    mixup_alpha: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"

    extra: dict = field(default_factory=dict)


def _dataset_idx(dataset_id: str) -> int:
    """Registry-driven dataset index for embedding lookups.  See v1_ssl._dataset_idx."""
    return REGISTRY.index_of(dataset_id)


# ---------------------------------------------------------------------------
# Paired feature precomputation
# ---------------------------------------------------------------------------


@dataclass
class _PairedSegment:
    """One segment's full-duration acoustic + vibration features (paired)."""

    acoustic_features: np.ndarray  # (n_mics, 2, F, T_ac)
    acoustic_xyz: np.ndarray  # (n_mics, 3)
    acoustic_fs: float
    vibration_features: np.ndarray  # (n_vib, 3, T_vib)
    vibration_xyz: np.ndarray  # (n_vib, 3)
    vibration_fs: float
    dataset_idx: int
    dataset_id: str  # canonical string id, used to resolve per-dataset scales
    mode_label: str
    recording_id: str
    source_dir: str


def _precompute_paired(
    s: TestDatasetSegment, cfg: V2SSLConfig
) -> _PairedSegment | None:
    # Acoustic — hop_length, n_fft, n_mels come from cfg (defaults to
    # ACOUSTIC_FEATURES per chapter 3 §3.4.2 grid sweep).
    if cfg.use_cwt:
        acoustic_features = compute_encoder_input_stack(
            s.segment.mic_data,
            fs=int(s.segment.mic_sample_rate),
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            cwt_n_scales=cfg.cwt_n_scales,
            standardize=cfg.standardize_acoustic,
        )
    else:
        mels = []
        for ch in range(s.segment.n_mic_channels):
            m = compute_log_mel_spectrogram(
                s.segment.mic_data[ch],
                fs=int(s.segment.mic_sample_rate),
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
            )
            mels.append(np.stack([m, m], axis=0).astype(np.float32))
        acoustic_features = np.stack(mels, axis=0)

    vibration_features = compute_vibration_input_stack(
        s.segment.accel_data,
        sample_rate=float(s.segment.accel_sample_rate),
        kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
        min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
        crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
        min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
        standardize=cfg.standardize_vibration,
    )

    if acoustic_features.shape[-1] < 2 or vibration_features.shape[-1] < 2:
        return None

    return _PairedSegment(
        acoustic_features=acoustic_features.astype(np.float32),
        acoustic_xyz=s.mic_positions.astype(np.float32),
        acoustic_fs=float(s.segment.mic_sample_rate) / float(cfg.hop_length),
        vibration_features=vibration_features.astype(np.float32),
        vibration_xyz=s.vib_positions.astype(np.float32),
        vibration_fs=float(s.segment.accel_sample_rate),
        dataset_idx=_dataset_idx(s.dataset_id),
        dataset_id=str(s.dataset_id),
        mode_label=s.mode_label or "Unknown",
        recording_id=s.recording_id,
        source_dir=str(s.source_dir),
    )


# ---------------------------------------------------------------------------
# Paired windowed dataset
# ---------------------------------------------------------------------------


def _resolve_paired_segment_scales(
    cfg: V2SSLConfig, dataset_id: str
) -> tuple[tuple[float, ...], float]:
    """Resolve `(scales_in_seconds, stride_ratio)` for one paired segment.

    Per-dataset override → global tuple → legacy single-scale fallback —
    same precedence as :func:`v1_ssl._resolve_segment_scales`.
    """
    per_ds = cfg.window_scales_seconds_per_dataset or {}
    if per_ds.get(dataset_id):
        scales = tuple(float(s) for s in per_ds[dataset_id])
        return scales, float(cfg.window_stride_ratio)
    if cfg.window_scales_seconds:
        scales = tuple(float(s) for s in cfg.window_scales_seconds)
        return scales, float(cfg.window_stride_ratio)
    legacy_ratio = (
        float(cfg.window_stride_seconds) / float(cfg.window_seconds)
        if cfg.window_seconds > 0
        else 0.5
    )
    return (float(cfg.window_seconds),), legacy_ratio


class _PairedWindowedDataset(tud.Dataset):
    """Yields paired acoustic + vibration windows aligned by wall-clock start.

    Per-segment frame counts (acoustic + vibration) are computed from the
    segment's own feature rates so a 2-second window over D4 raw vibration
    (~376 Hz) keeps all 752 samples while a 2-second window over D1 peak
    vibration (4 Hz) keeps all 8.  When `cfg.window_scales_seconds[_per_dataset]`
    is set, every (segment, scale) pair materialises its own window list;
    BOTH `n_ac` AND `n_vib` are computed from the same per-batch scale so
    the wall-clock pairing (sample 0 of mic ↔ sample 0 of vib) is preserved
    inside each window.  The grouped batch sampler buckets by
    `(n_mics, n_vib, frames_ac, frames_vib)` so that a single `torch.stack`
    never has to reconcile different sample counts.
    """

    def __init__(
        self,
        segments: list[_PairedSegment],
        cfg: V2SSLConfig,
        *,
        enable_mixup: bool = False,
    ) -> None:
        self.cfg = cfg
        self._segments = segments
        self._mixup_enabled = bool(enable_mixup and cfg.mixup_alpha > 0.0)
        if not segments:
            self._refs: list[tuple[int, int, int, int, int]] = []
            self._partner_pool: dict[tuple, list[int]] = {}
            return

        self._refs = []
        for si, seg in enumerate(segments):
            scales, stride_ratio = _resolve_paired_segment_scales(cfg, seg.dataset_id)
            T_ac = int(seg.acoustic_features.shape[-1])
            T_vib = int(seg.vibration_features.shape[-1])
            for scale_s in scales:
                n_ac = max(2, int(round(scale_s * seg.acoustic_fs)))
                n_vib = max(2, int(round(scale_s * seg.vibration_fs)))
                stride_ac = max(1, int(round(scale_s * stride_ratio * seg.acoustic_fs)))
                if T_ac < n_ac or T_vib < n_vib:
                    continue
                for start_ac in range(0, T_ac - n_ac + 1, stride_ac):
                    t_start = start_ac / max(seg.acoustic_fs, 1e-9)
                    start_vib = int(round(t_start * seg.vibration_fs))
                    if start_vib + n_vib > T_vib:
                        continue
                    self._refs.append((si, start_ac, start_vib, n_ac, n_vib))

        # Partner-pool index keyed by (dataset_idx, n_mics, n_vib, n_ac, n_vib,
        # mode_label) — the V2 grouped batch sampler's stack-safety key plus
        # mode_label.  Both modalities use the same partner so the cross-modal
        # pairing is preserved.
        self._partner_pool = {}
        if self._mixup_enabled:
            for ref_i, (si, _sa, _sv, n_ac, n_vib) in enumerate(self._refs):
                seg = self._segments[si]
                key = (
                    int(seg.dataset_idx),
                    int(seg.acoustic_features.shape[0]),
                    int(seg.vibration_features.shape[0]),
                    n_ac, n_vib, seg.mode_label,
                )
                self._partner_pool.setdefault(key, []).append(ref_i)

    def __len__(self) -> int:
        return len(self._refs)

    def _sample_mixup_partner(self, idx: int) -> int | None:
        import random as _r

        si, _sa, _sv, n_ac, n_vib = self._refs[idx]
        seg = self._segments[si]
        key = (
            int(seg.dataset_idx),
            int(seg.acoustic_features.shape[0]),
            int(seg.vibration_features.shape[0]),
            n_ac, n_vib, seg.mode_label,
        )
        candidates = self._partner_pool.get(key)
        if not candidates or len(candidates) < 2:
            return None
        anchor_rec = seg.recording_id
        for _ in range(6):
            cand = candidates[_r.randrange(len(candidates))]
            if cand == idx:
                continue
            cand_seg = self._segments[self._refs[cand][0]]
            if cand_seg.recording_id != anchor_rec:
                return cand
        return None

    def __getitem__(self, idx: int):
        si, sa, sv, n_ac, n_vib = self._refs[idx]
        seg = self._segments[si]
        ac = seg.acoustic_features[..., sa : sa + n_ac]
        vib = seg.vibration_features[..., sv : sv + n_vib]
        ac_t = torch.from_numpy(np.ascontiguousarray(ac))
        vib_t = torch.from_numpy(np.ascontiguousarray(vib))
        if self._mixup_enabled:
            partner = self._sample_mixup_partner(idx)
            if partner is not None:
                import random as _r

                p_si, p_sa, p_sv, _, _ = self._refs[partner]
                p_seg = self._segments[p_si]
                p_ac = p_seg.acoustic_features[..., p_sa : p_sa + n_ac]
                p_vib = p_seg.vibration_features[..., p_sv : p_sv + n_vib]
                lam = _r.betavariate(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
                lam = max(lam, 1.0 - lam)
                ac_t = lam * ac_t + (1.0 - lam) * torch.from_numpy(np.ascontiguousarray(p_ac))
                vib_t = lam * vib_t + (1.0 - lam) * torch.from_numpy(np.ascontiguousarray(p_vib))
        return {
            "ac_feat": ac_t,
            "vib_feat": vib_t,
            "ac_xyz": torch.from_numpy(seg.acoustic_xyz),
            "vib_xyz": torch.from_numpy(seg.vibration_xyz),
            "dataset_idx": int(seg.dataset_idx),
            "mode_label": seg.mode_label,
            "recording_id": seg.recording_id,
        }


class _PairedGroupedBatchSampler(tud.Sampler[list[int]]):
    """Group paired-window indices by (n_mics, n_vib) so each batch is stackable."""

    def __init__(
        self,
        dataset: _PairedWindowedDataset,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        # Bucket by (n_mics, n_vib, frames_ac, frames_vib) — D4 raw vibration
        # (376 Hz) and D1/D2 peak vibration (4 Hz) produce different per-window
        # frame counts even when their channel counts match.
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        for i, (si, _sa, _sv, n_ac, n_vib) in enumerate(dataset._refs):
            seg = dataset._segments[si]
            key = (
                int(seg.acoustic_features.shape[0]),
                int(seg.vibration_features.shape[0]),
                int(n_ac),
                int(n_vib),
            )
            groups.setdefault(key, []).append(i)
        self._groups = groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed if not self.shuffle else None)
        all_batches: list[list[int]] = []
        for _key, idxs in self._groups.items():
            ids = list(idxs)
            if self.shuffle:
                rng.shuffle(ids)
            for i in range(0, len(ids), self.batch_size):
                chunk = ids[i : i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(v) // self.batch_size for v in self._groups.values())
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size for v in self._groups.values()
        )


def _collate(batch: list[dict]) -> dict:
    ac_feat = torch.stack([b["ac_feat"] for b in batch], dim=0).float()
    vib_feat = torch.stack([b["vib_feat"] for b in batch], dim=0).float()
    ac_xyz = torch.stack([b["ac_xyz"] for b in batch], dim=0).float()
    vib_xyz = torch.stack([b["vib_xyz"] for b in batch], dim=0).float()
    dataset_idx = torch.tensor([b["dataset_idx"] for b in batch], dtype=torch.long)
    return {
        "ac_feat": ac_feat,
        "vib_feat": vib_feat,
        "ac_xyz": ac_xyz,
        "vib_xyz": vib_xyz,
        "dataset_idx": dataset_idx,
        "mode_label": [b["mode_label"] for b in batch],
        "recording_id": [b["recording_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Augmentation (paired)
# ---------------------------------------------------------------------------


class _PairedAugmenter:
    """Apply the V1 augmentation set to the paired (acoustic, vibration) batch.

    Each modality is augmented independently — there is no cross-modal
    coupling at the augmentation step.  The two SimCLR views are generated by
    calling this object twice per anchor.
    """

    def __init__(self, cfg: V2SSLConfig, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.gen = generator

    def _gain_and_dropout(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.gain_jitter_db > 0:
            db = (
                torch.rand(*x.shape[:2], 1, generator=self.gen, device=x.device)
                * (2 * cfg.gain_jitter_db)
                - cfg.gain_jitter_db
            )
            scale = 10.0 ** (db / 20.0)
            while scale.ndim < x.ndim:
                scale = scale.unsqueeze(-1)
            x = x * scale
        if cfg.channel_dropout_p > 0:
            drop = (
                torch.rand(*x.shape[:2], generator=self.gen, device=x.device)
                < cfg.channel_dropout_p
            )
            keep = (~drop).float()
            while keep.ndim < x.ndim:
                keep = keep.unsqueeze(-1)
            x = x * keep
        return x

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 2, F, T)
        cfg = self.cfg
        if cfg.spec_augment_freq_mask <= 0 and cfg.spec_augment_time_mask <= 0:
            return x
        B, N, C, Ff, Tt = x.shape
        if cfg.spec_augment_freq_mask > 0 and Ff > cfg.spec_augment_freq_mask:
            start = torch.randint(0, Ff - cfg.spec_augment_freq_mask + 1, (B, N), generator=self.gen, device=x.device)
            width = torch.randint(1, cfg.spec_augment_freq_mask + 1, (B, N), generator=self.gen, device=x.device)
            for b in range(B):
                for n in range(N):
                    s_, w_ = int(start[b, n]), int(width[b, n])
                    x[b, n, :, s_ : s_ + w_, :] = 0.0
        if cfg.spec_augment_time_mask > 0 and Tt > cfg.spec_augment_time_mask:
            start = torch.randint(0, Tt - cfg.spec_augment_time_mask + 1, (B, N), generator=self.gen, device=x.device)
            width = torch.randint(1, cfg.spec_augment_time_mask + 1, (B, N), generator=self.gen, device=x.device)
            for b in range(B):
                for n in range(N):
                    s_, w_ = int(start[b, n]), int(width[b, n])
                    x[b, n, :, :, s_ : s_ + w_] = 0.0
        return x

    def __call__(
        self, ac: torch.Tensor, vib: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ac = self._spec_augment(self._gain_and_dropout(ac.clone()))
        vib = self._gain_and_dropout(vib.clone())
        return ac, vib


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def _lmm_loss(
    fused: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cosine-similarity loss between fused-token output and pre-mask target,
    averaged over masked positions only.  Returns 0.0 when no position is
    masked in the batch (so the gradient simply skips this term).
    """
    if mask is None or not mask.any():
        return torch.zeros((), device=fused.device, dtype=fused.dtype)
    p = F.normalize(fused[mask], dim=-1)
    t = F.normalize(target[mask], dim=-1)
    return (1.0 - (p * t).sum(-1)).mean()


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


@dataclass
class V2Result:
    encoder: V2FusionEncoder
    projection: _ProjectionHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_simclr_history: list[float]
    train_lmm_history: list[float]
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    rq1: dict
    drop_vibration: bool
    early_stopped_epoch: int | None = None
    best_val_loss: float = float("nan")


def _time_split_paired_segment(
    seg: _PairedSegment, val_ratio: float
) -> tuple[_PairedSegment | None, _PairedSegment | None]:
    """Paired-segment analogue of `v1_ssl._time_split_segment`.

    Splits a single `_PairedSegment` along its time axes (acoustic and
    vibration sliced consistently in wall-clock seconds) so that
    single-recording mode-label groups still produce both a train and a
    val pseudo-segment.  See the v1 docstring for the methodological
    rationale; the temporal-leakage caveat applies equally here.
    """
    T_ac = int(seg.acoustic_features.shape[-1])
    T_vib = int(seg.vibration_features.shape[-1])
    if T_ac < 4 or T_vib < 4:
        return None, None

    # Anchor the split in wall-clock time so the acoustic and vibration
    # halves cover the same physical interval of the recording.
    t_total_s = T_ac / max(seg.acoustic_fs, 1e-9)
    t_val_s = t_total_s * float(val_ratio)
    n_val_ac = max(2, int(round(t_val_s * seg.acoustic_fs)))
    n_val_vib = max(2, int(round(t_val_s * seg.vibration_fs)))
    n_val_ac = min(n_val_ac, T_ac - 2)
    n_val_vib = min(n_val_vib, T_vib - 2)
    if n_val_ac < 2 or n_val_vib < 2 or T_ac - n_val_ac < 2 or T_vib - n_val_vib < 2:
        return None, None

    train_seg = _PairedSegment(
        acoustic_features=seg.acoustic_features[..., : T_ac - n_val_ac].copy(),
        acoustic_xyz=seg.acoustic_xyz,
        acoustic_fs=seg.acoustic_fs,
        vibration_features=seg.vibration_features[..., : T_vib - n_val_vib].copy(),
        vibration_xyz=seg.vibration_xyz,
        vibration_fs=seg.vibration_fs,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__train_half",
        source_dir=seg.source_dir,
    )
    val_seg = _PairedSegment(
        acoustic_features=seg.acoustic_features[..., T_ac - n_val_ac :].copy(),
        acoustic_xyz=seg.acoustic_xyz,
        acoustic_fs=seg.acoustic_fs,
        vibration_features=seg.vibration_features[..., T_vib - n_val_vib :].copy(),
        vibration_xyz=seg.vibration_xyz,
        vibration_fs=seg.vibration_fs,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__val_half",
        source_dir=seg.source_dir,
    )
    return train_seg, val_seg


def _split_segments_by_recording(
    segments: list[_PairedSegment], val_ratio: float, seed: int
) -> tuple[list[_PairedSegment], list[_PairedSegment]]:
    """Stratified-by-mode-label held-out split at the recording level.

    Mirrors `v1_ssl._split_segments_by_recording`: modes with ≥ 2
    recordings use the standard recording-level stratified split; modes
    with exactly 1 recording are **time-split** so they still appear in
    val (see `_time_split_paired_segment`).  Unlabeled D3 / D4
    speed-bucket recordings (mode_label = None) follow the
    ≥ 2 / == 1 branching like any other label.
    """
    rng = np.random.default_rng(seed)
    rec_to_label: dict[tuple, str | None] = {}
    rec_to_seg: dict[tuple, _PairedSegment] = {}
    for s in segments:
        key = (s.dataset_idx, s.recording_id, s.source_dir)
        if key not in rec_to_label:
            rec_to_label[key] = s.mode_label
            rec_to_seg[key] = s

    by_label: dict[str | None, list] = {}
    for key, lbl in rec_to_label.items():
        by_label.setdefault(lbl, []).append(key)

    train_keys: set = set()
    val_keys: set = set()
    time_split_train: list[_PairedSegment] = []
    time_split_val: list[_PairedSegment] = []
    for recs in by_label.values():
        recs_shuffled = list(recs)
        rng.shuffle(recs_shuffled)
        if len(recs_shuffled) == 1:
            only_key = recs_shuffled[0]
            tr, va = _time_split_paired_segment(rec_to_seg[only_key], val_ratio)
            if tr is not None and va is not None:
                time_split_train.append(tr)
                time_split_val.append(va)
            else:
                train_keys.update(recs_shuffled)
            continue
        n_val_for_label = max(1, int(round(len(recs_shuffled) * val_ratio)))
        n_val_for_label = min(n_val_for_label, len(recs_shuffled) - 1)
        val_keys.update(recs_shuffled[:n_val_for_label])
        train_keys.update(recs_shuffled[n_val_for_label:])

    if not train_keys and not time_split_train:
        train_keys = {val_keys.pop()}
    train = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in train_keys]
    val = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in val_keys]
    train.extend(time_split_train)
    val.extend(time_split_val)
    return train, val


def _gather_paired_segments(
    loaders: Iterable[TestDatasetLoader], cfg: V2SSLConfig
) -> list[_PairedSegment]:
    """Collect every healthy (non-anomaly) paired segment across all loaders.

    Healthy = `s.is_anomaly is False`.  Includes D3 / D4 speed-bucket
    recordings whose mode is unknown — they contribute SSL training signal
    even without a mode label.  The label-leakage invariant is preserved
    because SSL never reads `mode_label`.
    """
    out: list[_PairedSegment] = []
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            pre = _precompute_paired(s, cfg)
            if pre is not None:
                out.append(pre)
    return out


def _gather_labeled_segments(
    loaders: Iterable[TestDatasetLoader], cfg: V2SSLConfig
) -> list[_PairedSegment]:
    """Collect non-anomaly segments with an explicit mode_label — used by
    the K=3 RQ1 cluster-purity evaluation."""
    out: list[_PairedSegment] = []
    healthy = set(cfg.healthy_modes)
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            if s.mode_label is None or s.mode_label not in healthy:
                continue
            pre = _precompute_paired(s, cfg)
            if pre is not None:
                out.append(pre)
    return out


def train_v2_fusion(
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    cfg: V2SSLConfig | None = None,
    *,
    v1_acoustic_state_dict: dict | None = None,
    v1_vibration_state_dict: dict | None = None,
) -> V2Result:
    """Train V2 multimodal SSL on healthy paired windows.

    ``v1_*_state_dict`` are optional `PerModalityEncoder.state_dict()` payloads
    from V1 SSL.  When supplied they initialise the V2 per-modality encoders;
    otherwise the encoders start from random init.  V2 itself is fully
    self-supervised regardless — V1 was already SSL, so the chained system is
    label-free end-to-end.
    """
    cfg = cfg or V2SSLConfig()
    if hasattr(loaders, "list_segments"):
        loaders = [loaders]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"V2: device={describe_device(device)}")

    segments = _gather_paired_segments(loaders, cfg)
    if not segments:
        raise RuntimeError("V2 SSL: no healthy paired segments found")

    train_segs, val_segs = _split_segments_by_recording(segments, cfg.val_ratio, cfg.seed)
    train_ds = _PairedWindowedDataset(train_segs, cfg, enable_mixup=True)
    val_ds = _PairedWindowedDataset(val_segs, cfg, enable_mixup=False)
    if len(train_ds) == 0:
        raise RuntimeError("V2 SSL: zero training windows after splitting; lower window_seconds")

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_PairedGroupedBatchSampler(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_loader = tud.DataLoader(
        val_ds,
        batch_sampler=_PairedGroupedBatchSampler(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )

    encoder = V2FusionEncoder(
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        context_mode=cfg.context_mode,
        num_context_seeds=cfg.num_context_seeds,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    ).to(device)
    encoder.load_v1_weights(v1_acoustic_state_dict, v1_vibration_state_dict, strict=True)

    projection = _ProjectionHead(cfg.embed_dim, cfg.proj_dim).to(device)
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    aug_gen = torch.Generator(device=device)
    aug_gen.manual_seed(cfg.seed)
    augmenter = _PairedAugmenter(cfg, aug_gen)

    train_history: list[float] = []
    train_simclr: list[float] = []
    train_lmm: list[float] = []
    val_history: list[float] = []

    def _maybe_drop_vib(vib: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(vib) if cfg.drop_vibration else vib

    def _modality_dropout(
        ac: torch.Tensor, vib: torch.Tensor, gen: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent per-modality dropout.

        - When `cfg.acoustic_dropout_p > 0` or `cfg.vibration_dropout_p > 0`,
          each modality is dropped (zeroed) with its own probability per
          batch.  This is the publication-default regime and lets the
          asymmetry favour the strong acoustic mode-discriminator.
        - Otherwise falls back to the legacy symmetric 50/50 coin-flip
          gated by `cfg.modality_dropout_p` (kept for back-compatibility).
        """
        # Generator and target device must agree — `gen` lives on the same
        # device as `aug_gen` (the training device).  Sample scalars there.
        gd = gen.device
        ap = float(cfg.acoustic_dropout_p)
        vp = float(cfg.vibration_dropout_p)
        if ap > 0.0 or vp > 0.0:
            if ap > 0.0 and float(torch.rand((), generator=gen, device=gd)) < ap:
                ac = torch.zeros_like(ac)
            if vp > 0.0 and float(torch.rand((), generator=gen, device=gd)) < vp:
                vib = torch.zeros_like(vib)
            return ac, vib
        if cfg.modality_dropout_p <= 0.0:
            return ac, vib
        u = float(torch.rand((), generator=gen, device=gd))
        if u >= cfg.modality_dropout_p:
            return ac, vib
        if float(torch.rand((), generator=gen, device=gd)) < 0.5:
            return torch.zeros_like(ac), vib
        return ac, torch.zeros_like(vib)

    # Early-stop bookkeeping — tensor-clone snapshot, NOT copy.deepcopy.
    # See V1 trainer for the RAM-fragmentation rationale.
    def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    best_val_loss = float("inf")
    best_encoder_state = _snapshot(encoder)
    epochs_without_improvement = 0
    early_stopped_epoch: int | None = None

    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V2 fusion (cma={cfg.cma_weight}, ctx={cfg.context_mode})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        encoder.train()
        projection.train()
        loss_sum = simclr_sum = lmm_sum = 0.0
        n = 0
        for batch in train_loader:
            ac = batch["ac_feat"].to(device)
            vib = _maybe_drop_vib(batch["vib_feat"].to(device))
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)

            ac1, vib1 = augmenter(ac, vib)
            ac2, vib2 = augmenter(ac, vib)
            vib1 = _maybe_drop_vib(vib1)
            vib2 = _maybe_drop_vib(vib2)
            # Apply modality dropout *after* SimCLR view augmentation but
            # *before* the encoder.  Each view independently risks losing a
            # modality — the contrastive loss then pulls together the two
            # views' c_t even when one of them is single-modality.
            ac1, vib1 = _modality_dropout(ac1, vib1, aug_gen)
            ac2, vib2 = _modality_dropout(ac2, vib2, aug_gen)

            out1 = encoder(ac1, ac_xyz, vib1, vib_xyz, ds_idx, mask_p=cfg.lmm_mask_p)
            out2 = encoder(ac2, ac_xyz, vib2, vib_xyz, ds_idx, mask_p=0.0)

            z1 = projection(out1["context"])
            z2 = projection(out2["context"])
            if z1.shape[0] < 2:
                continue

            simclr = _nt_xent(z1, z2, cfg.temperature)
            lmm_a = _lmm_loss(out1["a_fused"], out1["a_target"], out1["mask_a"])
            lmm_v = _lmm_loss(out1["v_fused"], out1["v_target"], out1["mask_v"])
            lmm = lmm_a + lmm_v
            cma = (
                _nt_xent(out1["a_summary"], out1["v_summary"], cfg.cma_temperature)
                if cfg.cma_weight > 0.0 and out1["a_summary"].shape[0] >= 2
                else torch.zeros((), device=z1.device, dtype=z1.dtype)
            )
            loss = simclr + cfg.lmm_weight * lmm + cfg.cma_weight * cma

            optim.zero_grad()
            loss.backward()
            optim.step()

            B = z1.shape[0]
            loss_sum += float(loss.item()) * B
            simclr_sum += float(simclr.item()) * B
            lmm_sum += float(lmm.item()) * B
            n += B

        denom = max(1, n)
        train_history.append(loss_sum / denom)
        train_simclr.append(simclr_sum / denom)
        train_lmm.append(lmm_sum / denom)

        encoder.eval()
        projection.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ac = batch["ac_feat"].to(device)
                vib = _maybe_drop_vib(batch["vib_feat"].to(device))
                ac_xyz = batch["ac_xyz"].to(device)
                vib_xyz = batch["vib_xyz"].to(device)
                ds_idx = batch["dataset_idx"].to(device)

                ac1, vib1 = augmenter(ac, vib)
                ac2, vib2 = augmenter(ac, vib)
                vib1 = _maybe_drop_vib(vib1)
                vib2 = _maybe_drop_vib(vib2)

                out1 = encoder(ac1, ac_xyz, vib1, vib_xyz, ds_idx, mask_p=cfg.lmm_mask_p)
                out2 = encoder(ac2, ac_xyz, vib2, vib_xyz, ds_idx, mask_p=0.0)
                z1 = projection(out1["context"])
                z2 = projection(out2["context"])
                if z1.shape[0] < 2:
                    continue
                simclr = _nt_xent(z1, z2, cfg.temperature)
                lmm = _lmm_loss(out1["a_fused"], out1["a_target"], out1["mask_a"]) + \
                      _lmm_loss(out1["v_fused"], out1["v_target"], out1["mask_v"])
                cma = (
                    _nt_xent(out1["a_summary"], out1["v_summary"], cfg.cma_temperature)
                    if cfg.cma_weight > 0.0 and out1["a_summary"].shape[0] >= 2
                    else torch.zeros((), device=z1.device, dtype=z1.dtype)
                )
                B = z1.shape[0]
                v_loss += float((simclr + cfg.lmm_weight * lmm + cfg.cma_weight * cma).item()) * B
                v_n += B
        val_history.append(v_loss / max(1, v_n))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.3f}",
            val=f"{val_history[-1]:.3f}",
            lmm=f"{train_lmm[-1]:.3f}",
        )

        # Early stop on val total-loss.  See V1 trainer for the pattern.
        cur_val = val_history[-1]
        if cur_val < best_val_loss - cfg.early_stop_min_delta:
            best_val_loss = cur_val
            best_encoder_state = _snapshot(encoder)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                early_stopped_epoch = _epoch + 1
                break

    if cfg.restore_best:
        encoder.load_state_dict(best_encoder_state)
    del best_encoder_state

    # RQ1 cluster purity is computed on labeled-only val segments — D3/D4
    # speed-bucket recordings have no mode_label so including them in the
    # K=3 K-means against the three known modes would be uninterpretable.
    labeled_modes = set(cfg.healthy_modes)
    val_labeled = [s for s in val_segs if s.mode_label in labeled_modes]
    rq1 = evaluate_rq1_purity(encoder, val_labeled, cfg)

    def _qualify(seg: _PairedSegment) -> str:
        from pathlib import Path as _P

        return f"{_P(seg.source_dir).name}/{seg.recording_id}"

    return V2Result(
        encoder=encoder,
        projection=projection,
        train_loss_history=train_history,
        val_loss_history=val_history,
        train_simclr_history=train_simclr,
        train_lmm_history=train_lmm,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_segs}),
        rq1=rq1,
        drop_vibration=cfg.drop_vibration,
        early_stopped_epoch=early_stopped_epoch,
        best_val_loss=best_val_loss,
    )


def evaluate_rq1_purity(
    encoder: V2FusionEncoder,
    segments: list[_PairedSegment],
    cfg: V2SSLConfig,
    *,
    n_clusters: int = 3,
) -> dict:
    """Compute K-means cluster purity on `c_t` for the RQ1 headline number."""
    if not segments:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": 0, "label_set": tuple()}
    ds = _PairedWindowedDataset(segments, cfg)
    if len(ds) < 2:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": int(len(ds)), "label_set": tuple()}
    loader = tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
    )

    device = next(encoder.parameters()).device
    encoder.eval()
    contexts: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            if cfg.drop_vibration:
                vib = torch.zeros_like(vib)
            out = encoder(
                ac,
                batch["ac_xyz"].to(device),
                vib,
                batch["vib_xyz"].to(device),
                batch["dataset_idx"].to(device),
                mask_p=0.0,
            )
            contexts.append(out["context"].cpu().numpy())
            labels.extend(batch["mode_label"])
    embeddings = np.concatenate(contexts, axis=0)
    metric = cluster_purity_and_nmi(embeddings, labels, n_clusters=n_clusters, seed=cfg.seed)
    metric["n_windows"] = int(embeddings.shape[0])
    return metric


__all__ = [
    "V2Result",
    "V2SSLConfig",
    "evaluate_rq1_purity",
    "train_v2_fusion",
]
