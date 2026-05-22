"""V1 per-modality SSL warmup — SimCLR-style contrastive on healthy windows.

Trains the acoustic OR vibration encoder independently on healthy windows of
D1+D2.  No labels touch the training loop.  Mode labels appear only at the
sanity-gate evaluation step (cluster purity, computed by `cluster_metric`).

Design choices (per the plan's smart-decisions table):
  - Per-modality SimCLR contrastive (acoustic and vibration trained
    separately).  No cross-modal anything yet — that arrives in V2.
  - NT-Xent loss between two augmented views of each anchor window.
  - Augmentations applied in feature space: SpecAugment + channel dropout +
    gain jitter.  We deliberately skip pre-feature time-domain augmentations
    so feature extraction can be amortised with a one-time precomputation.
  - Per-recording train/val split (recordings disjoint between splits).

`train_v1_per_modality` is the single entry point.  It returns a `V1Result`
dataclass containing the trained encoder, projection head, sanity-gate
metrics, and the train/val recording IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal

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


def _registry_window_scales() -> dict[str, tuple[float, ...]]:
    """Per-dataset multi-scale window cadence sourced from the registry."""
    return {m.id: m.window_scales_seconds for m in REGISTRY}
from ...features.audio_spectral import compute_encoder_input_stack
from ...features.vibration_temporal import compute_vibration_input_stack
from ...ingestion.test_dataset_loader import (
    TestDatasetLoader,
    TestDatasetSegment,
)
from ..encoders import PerModalityEncoder
from .cluster_metric import cluster_purity_and_nmi


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V1SSLConfig:
    """Hyperparameters for V1 per-modality SSL warmup.

    Every field default below is sourced from
    :mod:`src.config.architecture` — that file is the single source of
    truth for thesis-relevant numerical choices.  Do not change defaults
    here; edit `architecture.py` and let the change propagate.
    """

    # Window cadence (legacy single-scale knobs; preserved for back-compat).
    # When `window_scales_seconds == ()` the dataset falls back to the
    # legacy single scale `(window_seconds,)` with the explicit
    # `window_stride_seconds` — byte-equivalent to pre-2026-05-19 behaviour.
    window_seconds: float = WINDOWING.window_seconds
    window_stride_seconds: float = WINDOWING.window_stride_seconds

    # Multi-scale window cadence (2026-05-19, full justification in
    # `docs/chapters/chapter_3_field_data_and_preprocessing.md` §3.4.4):
    #
    #   * `window_scales_seconds`: a tuple of per-segment scales applied to
    #     EVERY dataset.  Empty tuple means "use the legacy single scale".
    #   * `window_scales_seconds_per_dataset`: a dict
    #     `{dataset_id: (scale1, scale2, ...)}` that OVERRIDES the global
    #     tuple per dataset.  This is the publication path because the
    #     slowest-vibration constraint differs across datasets:
    #         - D1, D2 at 4 Hz vibration cannot resolve windows below
    #           5 / 4 = 1.25 s (one Vibration1DCNN kernel of length 5);
    #         - D3 at 16 Hz vibration can go down to 0.31 s;
    #         - D4 at 376 Hz raw vibration is effectively unconstrained.
    #
    # When `window_scale_strategy="uniform"`, each training batch is
    # single-dataset × single-scale (enforced by `_GroupedBatchSampler`'s
    # `(channel_count, n_frames)` bucket key); the scale within a segment's
    # valid set is sampled uniformly across all its windows by virtue of
    # the dataset emitting one (start, n_frames) tuple per (segment, scale).
    # Stride per scale = `scale * window_stride_ratio` (50 % overlap by
    # default — every frame appears in exactly two windows, the
    # audio-SSL convention).
    window_scales_seconds: tuple[float, ...] = ()
    window_scales_seconds_per_dataset: dict[str, tuple[float, ...]] = field(
        default_factory=_registry_window_scales
    )
    window_scale_strategy: Literal["fixed", "uniform"] = WINDOWING.window_scale_strategy
    window_stride_ratio: float = WINDOWING.window_stride_ratio

    # Encoder dims — sourced from `ENCODER` in architecture.py.
    feature_dim: int = ENCODER.feature_dim
    embed_dim: int = ENCODER.embed_dim
    n_heads: int = ENCODER.n_heads
    proj_dim: int = ENCODER.proj_dim

    # Training schedule — tuned per experiment, NOT centralised in
    # architecture.py.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1
    val_ratio: float = 0.3

    # Acoustic feature parameters — sourced from `ACOUSTIC_FEATURES`.
    n_mels: int = ACOUSTIC_FEATURES.n_mels
    n_fft: int = ACOUSTIC_FEATURES.n_fft
    hop_length: int = ACOUSTIC_FEATURES.hop_length
    cwt_n_scales: int = ACOUSTIC_CWT.n_scales
    use_cwt: bool = True  # smoke-test override: skip CWT to speed up
    # F4 toggle — per-channel z-score of the log-mel + CWT stack before the
    # CNN.  Default False: the 2026-05-14 audit found F4 is NOT load-bearing
    # — the V1 acoustic encoder trains fine without it once BatchNorm is the
    # encoder norm (the real collapse cause was F7/GroupNorm, not F4).  Kept
    # as a knob because the log-mel-vs-CWT channel-scale mismatch is a real
    # concern worth revisiting if the SSL objective changes.
    standardize_acoustic: bool = False

    # Vibration feature parameters — see `compute_vibration_input_stack` for
    # the full justification.  Channel-2 statistic (kurtosis vs crest factor)
    # is selected per-dataset from the segment's accel_sample_rate so a
    # single physical-time knob works across D1/D2 (4 Hz), D3 (16 Hz), and
    # D4 raw (~376 Hz).
    vib_kurtosis_window_seconds: float = VIBRATION_FEATURES.kurtosis_window_seconds
    vib_min_kurtosis_samples: int = VIBRATION_FEATURES.min_kurtosis_samples
    vib_crest_factor_window_seconds: float = VIBRATION_FEATURES.crest_factor_window_seconds
    vib_min_crest_factor_samples: int = VIBRATION_FEATURES.min_crest_factor_samples
    # Vibration amplitude + envelope z-score toggle (pre-existing behaviour;
    # default True).  Channel 2 (impulsiveness) is dimensionless and never
    # re-standardised — the F5 experiment that z-scored it showed no benefit
    # and was reverted.
    standardize_vibration: bool = True

    # R1a — Acoustic2DCNN channel-width multiplier.  Default 1 reproduces
    # the published 32/64/128 backbone; set to 2 for the wider 64/128/256
    # variant.  V1 and V2 must use the same value so that V2 can load
    # V1 acoustic weights without shape mismatch.  Vibration backbone is
    # NOT scaled — the R1 experiment changes acoustic only.
    acoustic_cnn_width_mult: int = ENCODER.acoustic_cnn_width_mult

    # Augmentations (applied in feature space) — per-experiment knobs;
    # the orchestrator's `_v1_cfg(quick)` overrides to publication values.
    gain_jitter_db: float = 6.0
    channel_dropout_p: float = 0.2
    spec_augment_freq_mask: int = 6
    spec_augment_time_mask: int = 8

    # Labels — healthy operating modes only.  RandomFault is anomaly data and
    # must not enter SSL training or the cluster-purity eval (which uses K=3
    # against {Pump, Standstill, Turbine}).  D3's `Healthy` token covers its
    # speed-bucket recordings.
    healthy_modes: tuple[str, ...] = (
        "Pump",
        "Standstill",
        "Turbine",
        "Healthy",
    )

    # Early stopping on val NT-Xent loss.  Patience counts consecutive epochs
    # without > `early_stop_min_delta` absolute improvement; when reached, the
    # loop breaks and (if `restore_best`) the encoder weights are reset to the
    # epoch with the best (lowest) val loss.  Best-state snapshot uses a
    # tensor-clone dict comprehension rather than `copy.deepcopy` to avoid
    # RAM fragmentation and PyTorch-autograd edge cases on long runs.
    patience: int = 3
    restore_best: bool = True
    early_stop_min_delta: float = 1e-4

    # Cross-recording mixup.  When > 0, each training window is linearly
    # blended with a partner window from a *different recording, same mode,
    # same dataset, same shape*: `feat = λ * feat_anchor + (1-λ) * feat_partner`
    # with `λ ~ Beta(α, α)`.  Inflates the effective recording cohort from
    # O(N_rec) to O(N_rec²) — the highest-ROI data-side regularizer for the
    # 6-10-recording cohort.  Default 0.0 disables (byte-equivalent to pre-fix
    # behaviour); recommended ablation values: {0.0, 0.2, 0.4}.  Only applied
    # to TRAIN windows — val windows are passed through untouched.
    mixup_alpha: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"

    extra: dict = field(default_factory=dict)


def _dataset_idx(dataset_id: str) -> int:
    """Registry-driven dataset index for embedding lookups.

    The integer index comes from ``DatasetRegistry`` (alphabetical-sorted by
    canonical id) and is stable across runs.  Aliases (e.g. ``illwerke`` ->
    ``illwerke_raw``) resolve to the same canonical index.
    """
    return REGISTRY.index_of(dataset_id)


# ---------------------------------------------------------------------------
# Feature precomputation
# ---------------------------------------------------------------------------


@dataclass
class _PrecomputedSegment:
    """One segment's full-duration feature stack + per-channel xyz + label."""

    features: np.ndarray  # acoustic: (n_mics, 2, F, T_frames) ; vibration: (n_vib, 3, T)
    xyz: np.ndarray  # (N, 3)
    dataset_idx: int
    dataset_id: str  # canonical string id used to resolve per-dataset window scales
    mode_label: str
    recording_id: str
    source_dir: str
    feature_fs: float  # frames per second along the last feature axis


def _precompute_segment(
    s: TestDatasetSegment,
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> _PrecomputedSegment | None:
    if modality == "acoustic":
        if cfg.use_cwt:
            features = compute_encoder_input_stack(
                s.segment.mic_data,
                fs=int(s.segment.mic_sample_rate),
                n_mels=cfg.n_mels,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                cwt_n_scales=cfg.cwt_n_scales,
                standardize=cfg.standardize_acoustic,
            )
        else:
            # Fast path for smoke tests — log-mel only, duplicated as channel 1.
            from ...features.audio_spectral import compute_log_mel_spectrogram

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
            features = np.stack(mels, axis=0)
        xyz = s.mic_positions.astype(np.float32)
        feature_fs = float(s.segment.mic_sample_rate) / float(cfg.hop_length)
    elif modality == "vibration":
        features = compute_vibration_input_stack(
            s.segment.accel_data,
            sample_rate=float(s.segment.accel_sample_rate),
            kurtosis_window_seconds=cfg.vib_kurtosis_window_seconds,
            min_kurtosis_samples=cfg.vib_min_kurtosis_samples,
            crest_factor_window_seconds=cfg.vib_crest_factor_window_seconds,
            min_crest_factor_samples=cfg.vib_min_crest_factor_samples,
            standardize=cfg.standardize_vibration,
        )
        xyz = s.vib_positions.astype(np.float32)
        feature_fs = float(s.segment.accel_sample_rate)
    else:
        raise ValueError(modality)

    if features.shape[-1] < 2:
        return None  # too short for any window

    return _PrecomputedSegment(
        features=features.astype(np.float32),
        xyz=xyz,
        dataset_idx=_dataset_idx(s.dataset_id),
        dataset_id=str(s.dataset_id),
        mode_label=s.mode_label or "Unknown",
        recording_id=s.recording_id,
        source_dir=str(s.source_dir),
        feature_fs=feature_fs,
    )


# ---------------------------------------------------------------------------
# Dataset + augmentation
# ---------------------------------------------------------------------------


def _resolve_segment_scales(
    cfg: V1SSLConfig, dataset_id: str
) -> tuple[tuple[float, ...], float]:
    """Return the (scales, stride_ratio) for one segment.

    Priority:
      1. Per-dataset override in ``cfg.window_scales_seconds_per_dataset``.
      2. Global ``cfg.window_scales_seconds`` tuple if non-empty.
      3. Legacy fallback ``(cfg.window_seconds,)`` with stride derived
         from ``cfg.window_stride_seconds / cfg.window_seconds`` — this
         is the byte-equivalent path for pre-2026-05-19 configs.
    """
    per_ds = cfg.window_scales_seconds_per_dataset or {}
    if dataset_id in per_ds and per_ds[dataset_id]:
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


class _WindowedFeatureDataset(tud.Dataset):
    """Yields (feature_window, xyz, dataset_idx, mode_label) per index.

    Window size in frames is computed *per segment* from its `feature_fs`
    AND per scale from `cfg.window_scales_seconds[_per_dataset]`, so a
    2-second window over D4 raw vibration (~376 Hz) keeps all 752 samples
    and a 2-second window over D1 peak vibration (4 Hz) keeps all 8.

    When the config defines multiple scales, every (segment, scale) pair
    materialises its own window list; the grouped batch sampler then
    buckets by ``(channel_count, n_frames)`` so every batch is
    single-dataset × single-scale and ``torch.stack`` never trips.
    """

    def __init__(
        self,
        segments: list[_PrecomputedSegment],
        cfg: V1SSLConfig,
        *,
        enable_mixup: bool = False,
    ) -> None:
        self.cfg = cfg
        self._segments = segments
        # Mixup is opt-in per dataset instance; the trainer enables it on the
        # TRAIN dataset only, never on val.  Reading `cfg.mixup_alpha > 0`
        # alone would silently blend val windows too.
        self._mixup_enabled = bool(enable_mixup and cfg.mixup_alpha > 0.0)

        if not segments:
            self._refs: list[tuple[int, int, int]] = []
            self._partner_pool: dict[tuple, list[int]] = {}
            return

        self._refs = []
        for si, seg in enumerate(segments):
            scales, stride_ratio = _resolve_segment_scales(cfg, seg.dataset_id)
            T = int(seg.features.shape[-1])
            for scale_s in scales:
                n_frames = max(2, int(round(scale_s * seg.feature_fs)))
                stride = max(1, int(round(scale_s * stride_ratio * seg.feature_fs)))
                if T < n_frames:
                    continue
                for start in range(0, T - n_frames + 1, stride):
                    self._refs.append((si, start, n_frames))

        # Partner-pool index for cross-recording mixup.  Bucket key is
        # `(dataset_idx, n_ch, n_frames, mode_label)` — same as the grouped
        # batch sampler's stack-safety key plus mode_label so we only mix
        # within-mode (preserves the contrastive task's semantic content).
        # Sampling at __getitem__ filters this bucket to refs from a
        # different recording.
        self._partner_pool = {}
        if self._mixup_enabled:
            for ref_i, (si, _start, n_frames) in enumerate(self._refs):
                seg = self._segments[si]
                n_ch = int(seg.features.shape[0])
                key = (int(seg.dataset_idx), n_ch, n_frames, seg.mode_label)
                self._partner_pool.setdefault(key, []).append(ref_i)

    def __len__(self) -> int:
        return len(self._refs)

    def _sample_mixup_partner(self, idx: int) -> int | None:
        """Pick a ref index from a different recording, same bucket.

        Returns None if no valid partner exists — caller then skips mixup
        for this anchor (rather than mixing same-recording windows, which
        would defeat the "cross-recording" purpose).
        """
        import random as _r

        si, _start, n_frames = self._refs[idx]
        seg = self._segments[si]
        n_ch = int(seg.features.shape[0])
        key = (int(seg.dataset_idx), n_ch, n_frames, seg.mode_label)
        candidates = self._partner_pool.get(key)
        if not candidates or len(candidates) < 2:
            return None
        anchor_rec = seg.recording_id
        # Cheap rejection sampling — small bucket → fast; large bucket → expect
        # to hit a different recording in ≤ 3 tries.
        for _ in range(6):
            cand = candidates[_r.randrange(len(candidates))]
            if cand == idx:
                continue
            cand_seg = self._segments[self._refs[cand][0]]
            if cand_seg.recording_id != anchor_rec:
                return cand
        return None

    def __getitem__(self, idx: int):
        si, start, n_frames = self._refs[idx]
        seg = self._segments[si]
        feat = seg.features[..., start : start + n_frames]
        feat_t = torch.from_numpy(np.ascontiguousarray(feat))
        if self._mixup_enabled:
            partner = self._sample_mixup_partner(idx)
            if partner is not None:
                import random as _r

                p_si, p_start, _ = self._refs[partner]
                p_seg = self._segments[p_si]
                p_feat = p_seg.features[..., p_start : p_start + n_frames]
                lam = _r.betavariate(self.cfg.mixup_alpha, self.cfg.mixup_alpha)
                # Symmetric beta keeps mixup variance moderate; max(lam, 1-lam)
                # ensures the anchor is always the dominant component, which
                # preserves the "anchor's labels apply" invariant the
                # contrastive loss assumes.
                lam = max(lam, 1.0 - lam)
                feat_t = lam * feat_t + (1.0 - lam) * torch.from_numpy(
                    np.ascontiguousarray(p_feat)
                )
        return {
            "feat": feat_t,
            "xyz": torch.from_numpy(seg.xyz),
            "dataset_idx": int(seg.dataset_idx),
            "mode_label": seg.mode_label,
            "recording_id": seg.recording_id,
        }


def _collate(batch: list[dict]) -> dict:
    feats = torch.stack([b["feat"] for b in batch], dim=0).float()
    xyz = torch.stack([b["xyz"] for b in batch], dim=0).float()
    dataset_idx = torch.tensor([b["dataset_idx"] for b in batch], dtype=torch.long)
    mode_labels = [b["mode_label"] for b in batch]
    rec_ids = [b["recording_id"] for b in batch]
    return {
        "feat": feats,
        "xyz": xyz,
        "dataset_idx": dataset_idx,
        "mode_label": mode_labels,
        "recording_id": rec_ids,
    }


class _GroupedBatchSampler(tud.Sampler[list[int]]):
    """Yields batches whose samples share the same per-segment channel count.

    Without this, a `D1` (4-mic) sample and a `D2` (5-mic) sample collide in
    `torch.stack`.  The Set-Transformer pool itself is channel-agnostic at
    forward time; the constraint is only at the batch-tensor boundary.
    """

    def __init__(
        self,
        dataset: "_WindowedFeatureDataset",
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
        # Bucket by (channels, frame count) — D1/D2 (4 Hz vib) and D4 (376 Hz)
        # have different per-window frame counts even when channel counts match.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (si, _start, n_frames) in enumerate(dataset._refs):
            n_ch = int(dataset._segments[si].features.shape[0])
            groups.setdefault((n_ch, n_frames), []).append(i)
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


class _Augmenter:
    """Apply gain jitter + channel dropout + SpecAugment in feature space.

    All augmentations operate per-sample, per-channel, and produce a single
    augmented copy.  The contrastive loop calls this twice per anchor to
    build the two SimCLR views.
    """

    def __init__(self, modality: str, cfg: V1SSLConfig, generator: torch.Generator) -> None:
        self.modality = modality
        self.cfg = cfg
        self.gen = generator

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: acoustic (B, N, 2, F, T) or vibration (B, N, 3, T)
        cfg = self.cfg
        x = x.clone()

        # Gain jitter — multiplicative scalar per (sample, channel)
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

        # Channel dropout — zero out a fraction of channels per sample
        if cfg.channel_dropout_p > 0:
            drop = (
                torch.rand(*x.shape[:2], generator=self.gen, device=x.device)
                < cfg.channel_dropout_p
            )
            keep = (~drop).float()
            while keep.ndim < x.ndim:
                keep = keep.unsqueeze(-1)
            x = x * keep

        # SpecAugment-style masks (acoustic only, on F and T)
        if self.modality == "acoustic" and (cfg.spec_augment_freq_mask > 0 or cfg.spec_augment_time_mask > 0):
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


# ---------------------------------------------------------------------------
# Projection head + NT-Xent
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
    """Symmetric SimCLR NT-Xent loss between two batches of normalised embeddings."""
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


@dataclass
class V1Result:
    encoder: PerModalityEncoder
    projection: _ProjectionHead
    train_loss_history: list[float]
    val_loss_history: list[float]
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    sanity_gate: dict
    modality: str
    # None means the loop ran to `cfg.epochs`; an int N means the loop broke
    # after epoch N because val loss did not improve for `cfg.patience` epochs.
    # `best_val_loss` is the min val loss observed (which `restore_best`
    # restored the encoder to if enabled).
    early_stopped_epoch: int | None = None
    best_val_loss: float = float("nan")


def _time_split_segment(
    seg: _PrecomputedSegment, val_ratio: float
) -> tuple[_PrecomputedSegment | None, _PrecomputedSegment | None]:
    """Split a single segment in time: first ``(1 − val_ratio)`` of frames →
    train pseudo-segment, last ``val_ratio`` of frames → val pseudo-segment.

    Used by `_split_segments_by_recording` when a `mode_label` group
    has only one recording — without this, that mode never appears in
    val and the K = 3 sanity-gate cannot evaluate it.  Time-splitting
    the same recording trades a small temporal-correlation leak (no
    different from any within-recording train/val split in the
    machine-condition-monitoring literature) for full coverage of the
    K = 3 mode hypothesis at evaluation time.  The pseudo-segments
    share `mode_label`, `xyz`, `feature_fs`, `dataset_idx`, and
    `source_dir`, and differ only in `features` (sliced along the last
    axis) and `recording_id` (suffixed `__train_half` / `__val_half`).
    Returns ``(None, None)`` if the segment is too short to split.
    """
    T = int(seg.features.shape[-1])
    if T < 4:
        return None, None
    n_val = max(2, int(round(T * val_ratio)))
    n_val = min(n_val, T - 2)
    if n_val < 2 or T - n_val < 2:
        return None, None
    train_feats = seg.features[..., : T - n_val]
    val_feats = seg.features[..., T - n_val :]
    train_seg = _PrecomputedSegment(
        features=train_feats,
        xyz=seg.xyz,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__train_half",
        source_dir=seg.source_dir,
        feature_fs=seg.feature_fs,
    )
    val_seg = _PrecomputedSegment(
        features=val_feats,
        xyz=seg.xyz,
        dataset_idx=seg.dataset_idx,
        dataset_id=seg.dataset_id,
        mode_label=seg.mode_label,
        recording_id=f"{seg.recording_id}__val_half",
        source_dir=seg.source_dir,
        feature_fs=seg.feature_fs,
    )
    return train_seg, val_seg


def _split_segments_by_recording(
    segments: list[_PrecomputedSegment], val_ratio: float, seed: int
) -> tuple[list[_PrecomputedSegment], list[_PrecomputedSegment]]:
    """Stratified-by-mode-label held-out split at the *recording* level.

    Three regimes per `mode_label` group:

    * ``count ≥ 2`` recordings — standard recording-level stratified split:
      each labeled mode contributes ⌊val_ratio · count⌋ recordings (≥ 1)
      to val with the rest in train.  No within-recording leakage.

    * ``count == 1`` recordings — the recording is **time-split**: first
      ``(1 − val_ratio)`` of its feature frames go into a "train half"
      pseudo-segment, the last ``val_ratio`` into a "val half" pseudo-
      segment.  This guarantees the mode appears in val for the K = 3
      sanity-gate (the alternative — putting the single recording
      entirely in train — was the structural reason Standstill never
      appeared in val before this fix; see REVIEW.md fifth-pass note).
      The temporal-correlation cost is the same as any within-recording
      split in the CBM literature and is documented as a known limit.

    * ``count == 0`` recordings — nothing to do.

    Recordings with `mode_label = None` (D3 / D4 healthy speed-bucket
    recordings — mode unrecorded by protocol) are split at the recording
    level without stratification.
    """
    rng = np.random.default_rng(seed)
    rec_to_label: dict[tuple, str | None] = {}
    rec_to_seg: dict[tuple, _PrecomputedSegment] = {}
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
    time_split_train: list[_PrecomputedSegment] = []
    time_split_val: list[_PrecomputedSegment] = []
    for lbl, recs in by_label.items():
        recs_shuffled = list(recs)
        rng.shuffle(recs_shuffled)
        if len(recs_shuffled) == 1:
            # Time-split the single recording so this mode appears in val.
            only_key = recs_shuffled[0]
            tr, va = _time_split_segment(rec_to_seg[only_key], val_ratio)
            if tr is not None and va is not None:
                time_split_train.append(tr)
                time_split_val.append(va)
            else:
                # Segment too short to split — fall back to train-only.
                train_keys.update(recs_shuffled)
            continue
        n_val_for_label = max(1, int(round(len(recs_shuffled) * val_ratio)))
        # Keep at least one in train per label too.
        n_val_for_label = min(n_val_for_label, len(recs_shuffled) - 1)
        val_keys.update(recs_shuffled[:n_val_for_label])
        train_keys.update(recs_shuffled[n_val_for_label:])

    train = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in train_keys]
    val = [s for s in segments if (s.dataset_idx, s.recording_id, s.source_dir) in val_keys]
    train.extend(time_split_train)
    val.extend(time_split_val)
    return train, val


def _gather_healthy_segments(
    loaders: Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> list[_PrecomputedSegment]:
    """Collect every healthy (non-anomaly) recording across all loaders.

    Healthy = `s.is_anomaly is False`.  This includes:
      - D1 / D2 with explicit mode labels (Pump / Standstill / Turbine)
      - D3 / D4 speed-bucket recordings whose mode label is unknown
        (the campaign protocol did not record which of the three modes
        the unit was in; the encoder discovers it via K-means at inference).
    The label-leakage invariant is preserved: SSL training never reads
    `mode_label`; only the cluster-purity sanity gate does, and only for
    segments where `mode_label is not None`.
    """
    out: list[_PrecomputedSegment] = []
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            pre = _precompute_segment(s, modality, cfg)
            if pre is not None:
                out.append(pre)
    return out


def _gather_labeled_segments(
    loaders: Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> list[_PrecomputedSegment]:
    """Collect only segments that have an explicit `mode_label` *and* are
    not anomaly recordings.  Used by the K=3 cluster-purity sanity gate.

    On the current four-campaign setup this yields D1 + D2 mode folders,
    i.e. recordings whose Pump / Standstill / Turbine label is ground truth.
    D3 / D4 speed buckets and all RandomFault recordings are excluded.
    """
    out: list[_PrecomputedSegment] = []
    healthy = set(cfg.healthy_modes)
    for loader in loaders:
        for s in loader.list_segments():
            if s.is_anomaly:
                continue
            if s.mode_label is None or s.mode_label not in healthy:
                continue
            pre = _precompute_segment(s, modality, cfg)
            if pre is not None:
                out.append(pre)
    return out


def _encode_summary(
    encoder: PerModalityEncoder, batch: dict, device: torch.device
) -> torch.Tensor:
    feat = batch["feat"].to(device)
    xyz = batch["xyz"].to(device)
    dataset_idx = batch["dataset_idx"].to(device)
    _, summary = encoder(feat, xyz, dataset_idx)
    return summary


def train_v1_per_modality(
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig | None = None,
) -> V1Result:
    cfg = cfg or V1SSLConfig()
    # Accept either a single loader or any iterable of loaders.  Duck-typed so
    # test stubs that quack like `TestDatasetLoader` (have `list_segments`) work.
    if hasattr(loaders, "list_segments"):
        loaders = [loaders]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"V1 {modality}: device={describe_device(device)}")

    segments = _gather_healthy_segments(loaders, modality, cfg)
    if not segments:
        raise RuntimeError("V1 SSL: no healthy segments found")

    train_segs, val_segs = _split_segments_by_recording(segments, cfg.val_ratio, cfg.seed)

    train_ds = _WindowedFeatureDataset(train_segs, cfg, enable_mixup=True)
    val_ds = _WindowedFeatureDataset(val_segs, cfg, enable_mixup=False)
    if len(train_ds) == 0:
        raise RuntimeError("V1 SSL: zero training windows after splitting; lower window_seconds")

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_GroupedBatchSampler(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_loader = tud.DataLoader(
        val_ds,
        batch_sampler=_GroupedBatchSampler(val_ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )

    encoder = PerModalityEncoder(
        modality=modality,
        feature_dim=cfg.feature_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    ).to(device)
    projection = _ProjectionHead(cfg.embed_dim, cfg.proj_dim).to(device)
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Augmenter sees CPU tensors (it runs before the .to(device) move below),
    # so the generator MUST be on CPU — torch requires generator and target
    # tensor device to match.
    aug_gen = torch.Generator(device="cpu")
    aug_gen.manual_seed(cfg.seed)
    augmenter = _Augmenter(modality, cfg, aug_gen)

    train_history: list[float] = []
    val_history: list[float] = []

    # Early-stop bookkeeping.  Snapshot via tensor-clone (NOT copy.deepcopy):
    # deepcopy on a state_dict allocates a fresh CPU tensor per parameter via
    # the pickling path, which fragments system RAM across many "best" updates
    # over hundreds of epochs.  The dict comprehension below is a single tight
    # loop of cloned-tensor allocations and lets the previous best_state's
    # tensors GC the instant we overwrite the binding.
    def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    best_val_loss = float("inf")
    best_encoder_state = _snapshot(encoder)
    epochs_without_improvement = 0
    early_stopped_epoch: int | None = None

    epoch_iter = tqdm(
        range(cfg.epochs),
        desc=f"V1 {modality}",
        unit="epoch",
        leave=False,
    )
    for epoch in epoch_iter:
        encoder.train()
        projection.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            view1 = augmenter(batch["feat"]).to(device)
            view2 = augmenter(batch["feat"]).to(device)
            xyz = batch["xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)

            _, s1 = encoder(view1, xyz, ds_idx)
            _, s2 = encoder(view2, xyz, ds_idx)
            z1 = projection(s1)
            z2 = projection(s2)
            if z1.shape[0] < 2:
                continue
            loss = _nt_xent(z1, z2, cfg.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * z1.shape[0]
            n += z1.shape[0]
        train_history.append(epoch_loss / max(1, n))

        encoder.eval()
        projection.eval()
        epoch_val = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                view1 = augmenter(batch["feat"]).to(device)
                view2 = augmenter(batch["feat"]).to(device)
                xyz = batch["xyz"].to(device)
                ds_idx = batch["dataset_idx"].to(device)
                _, s1 = encoder(view1, xyz, ds_idx)
                _, s2 = encoder(view2, xyz, ds_idx)
                z1 = projection(s1)
                z2 = projection(s2)
                if z1.shape[0] < 2:
                    continue
                loss = _nt_xent(z1, z2, cfg.temperature)
                epoch_val += float(loss.item()) * z1.shape[0]
                n_val += z1.shape[0]
        val_history.append(epoch_val / max(1, n_val))
        epoch_iter.set_postfix(
            train=f"{train_history[-1]:.3f}", val=f"{val_history[-1]:.3f}"
        )

        # Early stop on val NT-Xent loss.  Snapshot the encoder (the projection
        # head is discarded post-training) every time the val loss improves
        # past `early_stop_min_delta`.  Break when patience runs out.
        cur_val = val_history[-1]
        if cur_val < best_val_loss - cfg.early_stop_min_delta:
            best_val_loss = cur_val
            best_encoder_state = _snapshot(encoder)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                early_stopped_epoch = epoch + 1
                break

    if cfg.restore_best:
        encoder.load_state_dict(best_encoder_state)
    del best_encoder_state  # let GC reclaim before sanity-gate eval allocates

    # Sanity gate evaluates on the held-out *labeled* subset only.  D3/D4
    # healthy windows participate in training (they're inside `val_segs`
    # by recording split) but their mode_label is None — including them in
    # the K=3 K-means against the three known modes would give a noisy
    # quality number.  Filter to segments with an explicit mode_label.
    labeled_modes = set(cfg.healthy_modes)
    val_labeled = [
        s for s in val_segs if s.mode_label in labeled_modes
    ]
    sanity = evaluate_sanity_gate(encoder, val_labeled, modality, cfg)

    # Expose recording IDs as fully-qualified `<source_dir_basename>/<recording_id>`
    # so D1's same-`recording_id`-in-different-folders case (e.g., `All/Pump` vs
    # `Pump/Pump`) is disambiguated.  Disjointness of these sets is the
    # held-out-recording invariant.
    def _qualify(seg: _PrecomputedSegment) -> str:
        from pathlib import Path as _P

        return f"{_P(seg.source_dir).name}/{seg.recording_id}"

    return V1Result(
        encoder=encoder,
        projection=projection,
        train_loss_history=train_history,
        val_loss_history=val_history,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_segs}),
        sanity_gate=sanity,
        modality=modality,
        early_stopped_epoch=early_stopped_epoch,
        best_val_loss=best_val_loss,
    )


def evaluate_sanity_gate(
    encoder: PerModalityEncoder,
    segments: list[_PrecomputedSegment],
    modality: Literal["acoustic", "vibration"],
    cfg: V1SSLConfig,
) -> dict:
    """Compute K-means cluster purity vs folder labels on `segments`."""
    if not segments:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": 0, "label_set": tuple()}

    ds = _WindowedFeatureDataset(segments, cfg)
    if len(ds) < 2:
        return {"purity": 0.0, "nmi": 0.0, "n_windows": int(len(ds)), "label_set": tuple()}

    loader = tud.DataLoader(
        ds,
        batch_sampler=_GroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=cfg.seed),
        collate_fn=_collate,
    )

    device = next(encoder.parameters()).device
    encoder.eval()
    summaries: list[np.ndarray] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            summary = _encode_summary(encoder, batch, device)
            summaries.append(summary.cpu().numpy())
            labels.extend(batch["mode_label"])
    embeddings = np.concatenate(summaries, axis=0)
    # K=3 against the three healthy operating modes (Pump, Standstill, Turbine).
    # RandomFault is anomaly data, not a fourth mode — it is filtered out by
    # `healthy_modes` upstream of this evaluation.
    metric = cluster_purity_and_nmi(embeddings, labels, n_clusters=3, seed=cfg.seed)
    metric["n_windows"] = int(embeddings.shape[0])
    return metric


__all__ = [
    "V1SSLConfig",
    "V1Result",
    "train_v1_per_modality",
    "evaluate_sanity_gate",
]
