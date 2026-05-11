"""V3 trainer — frozen V2 encoder → CNF on healthy `(x, c)` pairs → per-cluster thresholds.

`x` = mean-pool of the fused-token sequence (a fixed pool, distinct from PMA).
`c` = `c_t` = PMA pool (the V2 context vector).  The flow learns the conditional
density `p(x | c)` and emits ``-log p(x | c)`` as the anomaly score.

Two ablation knobs:
  - ``cfg.unconditional=True`` → A2 ablation.  Zeros are passed as `c` to the
    flow at both training and inference, so the FiLM modulation degenerates
    to identity and the flow becomes unconditional.
  - The synthetic transition stress-test (`make_transition_segment` +
    `score_segment`) splices two healthy segments with a linear crossfade and
    measures the false-alert rate over the transition windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...ingestion.test_dataset_loader import TestDatasetLoader
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _collate,
    _gather_paired_segments,
    _precompute_paired,
    _split_segments_by_recording,
)
from .cnf_head import ConditionalRealNVP
from .threshold import PerClusterThresholds


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3Config:
    """V3 conditional anomaly head config."""

    # CNF dims
    n_layers: int = 6
    hidden_dim: int = 64
    n_hidden_per_net: int = 2
    # Per-coupling log-scale bound: tanh(scale_net) * scale_max.  2.0 is the
    # standard RealNVP setting and gives each layer a Jacobian factor in
    # [e⁻², e²]; the prior 1.0 default was over-conservative for a 128-D
    # latent with only 6 coupling layers.
    scale_max: float = 2.0

    # Training
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.3
    # Cosine LR schedule with eta_min = 1 % of base lr.  Improves the
    # tail of NLL training relative to the previous fixed-rate AdamW.
    use_cosine_lr: bool = True

    # A2 ablation — zero c at train+infer for unconditional flow.
    unconditional: bool = False

    # Threshold fit — fully unsupervised on healthy data.
    # K = 3 matches the operating-mode hypothesis (Pump / Standstill /
    # Turbine).  A larger K splits real modes into noise sub-clusters whose
    # individual thresholds inherit only a fraction of the per-mode healthy
    # tail and therefore over-trigger; a smaller K conflates modes and
    # over-loosens the threshold.
    n_threshold_clusters: int = 3
    threshold_percentile: int = 95
    # The Youden's-J calibration helper exists in `threshold.py` for
    # post-hoc analysis but is **NOT** wired into the orchestrator.  It
    # would require per-window anomaly labels (or an assumption that all
    # D2 RF / D3 hit windows are anomalous), which the field-collection
    # protocol does not provide.  Threshold quality is instead validated
    # post-hoc via per-cohort alert rates (the orchestrator's
    # `v3_alert_rate_per_cohort` metric).
    calibrate_with_anomalies: bool = False

    # System
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Encoder feature extraction
# ---------------------------------------------------------------------------


def _extract_xc(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run frozen V2 encoder forward; collect mean-pool x, PMA c, mode labels."""
    encoder.eval()
    xs: list[torch.Tensor] = []
    cs: list[torch.Tensor] = []
    labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            xs.append(fused.mean(dim=1).cpu())
            cs.append(out["context"].cpu())
            labels.extend(batch["mode_label"])
    return torch.cat(xs, dim=0), torch.cat(cs, dim=0), labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@dataclass
class V3Result:
    flow: ConditionalRealNVP
    thresholds: PerClusterThresholds
    train_nll: list[float]
    val_nll: list[float]
    train_recording_ids: list[str]
    val_recording_ids: list[str]
    val_scores: np.ndarray
    val_contexts: np.ndarray
    val_labels: list[str]
    unconditional: bool


def train_v3_cnf(
    v2_encoder: V2FusionEncoder,
    loaders: TestDatasetLoader | Iterable[TestDatasetLoader],
    *,
    v2_cfg: V2SSLConfig,
    v3_cfg: V3Config | None = None,
) -> V3Result:
    """Train the conditional CNF on healthy windows; fit per-cluster thresholds."""
    v3_cfg = v3_cfg or V3Config()
    if hasattr(loaders, "list_segments"):
        loaders = [loaders]

    torch.manual_seed(v3_cfg.seed)
    np.random.seed(v3_cfg.seed)
    device = torch.device(v3_cfg.device)
    v2_encoder = v2_encoder.to(device)
    v2_encoder.eval()
    for p in v2_encoder.parameters():
        p.requires_grad_(False)

    segments = _gather_paired_segments(loaders, v2_cfg)
    if not segments:
        raise RuntimeError("V3: no healthy paired segments found")

    train_segs, val_segs = _split_segments_by_recording(
        segments, v3_cfg.val_ratio, v3_cfg.seed
    )
    train_ds = _PairedWindowedDataset(train_segs, v2_cfg)
    val_ds = _PairedWindowedDataset(val_segs, v2_cfg)
    if len(train_ds) == 0:
        raise RuntimeError("V3: zero training windows after splitting")

    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_PairedGroupedBatchSampler(train_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed),
        collate_fn=_collate,
    )
    val_loader = tud.DataLoader(
        val_ds,
        batch_sampler=_PairedGroupedBatchSampler(val_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed),
        collate_fn=_collate,
    )

    # Extract once — encoder is frozen so a single forward pass over the data
    # gives the full training set.
    x_train, c_train, _ = _extract_xc(v2_encoder, train_loader, device)
    x_val, c_val, val_labels = _extract_xc(v2_encoder, val_loader, device)

    if v3_cfg.unconditional:
        c_train_used = torch.zeros_like(c_train)
        c_val_used = torch.zeros_like(c_val)
    else:
        c_train_used = c_train
        c_val_used = c_val

    flow = ConditionalRealNVP(
        dim=int(x_train.shape[1]),
        c_dim=int(c_train.shape[1]),
        n_layers=v3_cfg.n_layers,
        hidden_dim=v3_cfg.hidden_dim,
        n_hidden_per_net=v3_cfg.n_hidden_per_net,
        scale_max=v3_cfg.scale_max,
    ).to(device)
    optim = torch.optim.AdamW(
        flow.parameters(), lr=v3_cfg.lr, weight_decay=v3_cfg.weight_decay
    )
    if v3_cfg.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, int(v3_cfg.epochs)), eta_min=v3_cfg.lr * 0.01
        )
    else:
        scheduler = None

    n_train = int(x_train.shape[0])
    train_nll: list[float] = []
    val_nll: list[float] = []

    suffix = "unconditional" if v3_cfg.unconditional else "conditional"
    epoch_iter = tqdm(
        range(v3_cfg.epochs),
        desc=f"V3 CNF ({suffix})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        flow.train()
        perm = torch.randperm(n_train)
        loss_sum = 0.0
        n = 0
        for i in range(0, n_train, v3_cfg.batch_size):
            idx = perm[i : i + v3_cfg.batch_size]
            xb = x_train[idx].to(device)
            cb = c_train_used[idx].to(device)
            log_p = flow.log_prob(xb, cb)
            loss = -log_p.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        if scheduler is not None:
            scheduler.step()
        train_nll.append(loss_sum / max(1, n))

        flow.eval()
        with torch.no_grad():
            v_log_p = flow.log_prob(x_val.to(device), c_val_used.to(device))
            val_nll.append(float((-v_log_p.mean()).item()))
        epoch_iter.set_postfix(
            train_nll=f"{train_nll[-1]:.3f}", val_nll=f"{val_nll[-1]:.3f}"
        )

    flow.eval()
    with torch.no_grad():
        scores_val = (
            flow.anomaly_score(x_val.to(device), c_val_used.to(device)).cpu().numpy()
        )

    # Threshold fitting always clusters on the *real* `c_t` (label-free);
    # the unconditional flag only affects the flow itself.
    n_clusters = min(v3_cfg.n_threshold_clusters, max(1, c_val.shape[0]))
    thresholds = PerClusterThresholds.fit(
        c_val.numpy(),
        scores_val,
        n_clusters=n_clusters,
        seed=v3_cfg.seed,
    )

    def _qualify(seg: _PairedSegment) -> str:
        return f"{Path(seg.source_dir).name}/{seg.recording_id}"

    return V3Result(
        flow=flow,
        thresholds=thresholds,
        train_nll=train_nll,
        val_nll=val_nll,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_segs}),
        val_scores=scores_val,
        val_contexts=c_val.numpy(),
        val_labels=val_labels,
        unconditional=v3_cfg.unconditional,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def score_segments(
    v2_encoder: V2FusionEncoder,
    flow: ConditionalRealNVP,
    segments: list[_PairedSegment],
    *,
    v2_cfg: V2SSLConfig,
    batch_size: int = 32,
    unconditional: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Score a list of paired segments with the trained flow.

    Returns ``(scores, contexts, mode_labels)`` aligned per window.
    """
    device = torch.device(device)
    v2_encoder = v2_encoder.to(device).eval()
    flow = flow.to(device).eval()

    ds = _PairedWindowedDataset(segments, v2_cfg)
    if len(ds) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, flow.c_dim), dtype=np.float64), []
    loader = tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(ds, batch_size, shuffle=False, seed=0),
        collate_fn=_collate,
    )

    x, c, labels = _extract_xc(v2_encoder, loader, device)
    c_used = torch.zeros_like(c) if unconditional else c
    with torch.no_grad():
        scores = flow.anomaly_score(x.to(device), c_used.to(device)).cpu().numpy()
    return scores, c.numpy(), labels


# ---------------------------------------------------------------------------
# Synthetic transition stress-test
# ---------------------------------------------------------------------------


def precompute_paired(seg, cfg: V2SSLConfig) -> _PairedSegment | None:
    """Public wrapper for the V2 paired-feature precomputation (used by tests)."""
    return _precompute_paired(seg, cfg)


def gate_samples_by_alert(
    samples: list,
    flow: ConditionalRealNVP,
    thresholds: PerClusterThresholds,
    *,
    percentile: int = 99,
    unconditional: bool = False,
    keep_dataset_ids: tuple[str, ...] = (),
    device: torch.device | str = "cpu",
) -> tuple[list, dict]:
    """Filter ``samples`` (any object with `context`, `x_for_v3`, `dataset_id`
    fields — typically `V4Sample`) to only those V3 flags as anomalous.

    `keep_dataset_ids` is a passthrough list — samples whose `dataset_id`
    is in this set are kept regardless of V3's flag.  Use this to keep
    every D2/D3 RandomFault window (continuous-anomaly recordings) while
    gating only D4 (sparse-anomaly).

    Returns ``(kept_samples, stats_dict)`` where ``stats_dict`` records
    per-dataset counts of in / kept / alert-rate for the run log.
    """
    if not samples:
        return [], {"n_in": 0, "n_kept": 0, "by_dataset": {}}

    device = torch.device(device)
    flow = flow.to(device).eval()

    xs = torch.from_numpy(np.stack([s.x_for_v3 for s in samples], axis=0)).to(device)
    cs = torch.from_numpy(np.stack([s.context for s in samples], axis=0)).to(device)
    if unconditional:
        cs = torch.zeros_like(cs)
    with torch.no_grad():
        scores = flow.anomaly_score(xs, cs).cpu().numpy()

    contexts_np = np.stack([s.context for s in samples], axis=0)
    alerts, _clusters = thresholds.alert(contexts_np, scores, percentile=percentile)

    kept = []
    stats: dict[str, dict] = {}
    for s, alert in zip(samples, alerts):
        bucket = stats.setdefault(s.dataset_id, {"in": 0, "kept": 0, "alerts": 0})
        bucket["in"] += 1
        bucket["alerts"] += int(alert)
        if s.dataset_id in keep_dataset_ids or bool(alert):
            kept.append(s)
            bucket["kept"] += 1
    return kept, {"n_in": len(samples), "n_kept": len(kept), "by_dataset": stats}


def make_transition_segment(
    seg_a: _PairedSegment,
    seg_b: _PairedSegment,
    *,
    crossfade_seconds: float = 1.0,
    label: str | None = None,
) -> _PairedSegment:
    """Concatenate two healthy paired segments with a linear acoustic +
    vibration crossfade.  Both segments must share modality counts and feature
    cadences (which they do when drawn from the same dataset).

    The crossfaded region is the last `crossfade_seconds` of A overlapped with
    the first `crossfade_seconds` of B; outputs lengths are
    ``T_a + T_b - crossfade_frames`` per modality.
    """
    if (
        seg_a.acoustic_features.shape[:-1] != seg_b.acoustic_features.shape[:-1]
        or seg_a.vibration_features.shape[:-1] != seg_b.vibration_features.shape[:-1]
    ):
        raise ValueError("transition segments must share sensor counts and feature dims")
    if abs(seg_a.acoustic_fs - seg_b.acoustic_fs) > 1e-6 or abs(
        seg_a.vibration_fs - seg_b.vibration_fs
    ) > 1e-6:
        raise ValueError("transition segments must share feature cadences")

    n_ac = max(1, int(round(crossfade_seconds * seg_a.acoustic_fs)))
    n_vib = max(1, int(round(crossfade_seconds * seg_a.vibration_fs)))
    if seg_a.acoustic_features.shape[-1] < n_ac or seg_b.acoustic_features.shape[-1] < n_ac:
        raise ValueError("acoustic segments too short for the requested crossfade")
    if seg_a.vibration_features.shape[-1] < n_vib or seg_b.vibration_features.shape[-1] < n_vib:
        raise ValueError("vibration segments too short for the requested crossfade")

    fade_in = np.linspace(0.0, 1.0, n_ac, dtype=np.float32)
    fade_out = 1.0 - fade_in
    crossed_ac = (
        seg_a.acoustic_features[..., -n_ac:] * fade_out
        + seg_b.acoustic_features[..., :n_ac] * fade_in
    )
    spliced_ac = np.concatenate(
        [
            seg_a.acoustic_features[..., :-n_ac],
            crossed_ac,
            seg_b.acoustic_features[..., n_ac:],
        ],
        axis=-1,
    )

    fv_in = np.linspace(0.0, 1.0, n_vib, dtype=np.float32)
    fv_out = 1.0 - fv_in
    crossed_v = (
        seg_a.vibration_features[..., -n_vib:] * fv_out
        + seg_b.vibration_features[..., :n_vib] * fv_in
    )
    spliced_v = np.concatenate(
        [
            seg_a.vibration_features[..., :-n_vib],
            crossed_v,
            seg_b.vibration_features[..., n_vib:],
        ],
        axis=-1,
    )

    return _PairedSegment(
        acoustic_features=spliced_ac.astype(np.float32),
        acoustic_xyz=seg_a.acoustic_xyz,
        acoustic_fs=seg_a.acoustic_fs,
        vibration_features=spliced_v.astype(np.float32),
        vibration_xyz=seg_a.vibration_xyz,
        vibration_fs=seg_a.vibration_fs,
        dataset_idx=seg_a.dataset_idx,
        mode_label=label or f"transition[{seg_a.mode_label}->{seg_b.mode_label}]",
        recording_id=f"{seg_a.recording_id}__to__{seg_b.recording_id}",
        source_dir=str(seg_a.source_dir),
    )


def encoder_level_transition_fpr(
    v2_encoder: V2FusionEncoder,
    flow: ConditionalRealNVP,
    thresholds: PerClusterThresholds,
    seg_a: _PairedSegment,
    seg_b: _PairedSegment,
    *,
    v2_cfg: V2SSLConfig,
    n_crossfade_windows: int = 8,
    percentile: int | str = 95,
    unconditional: bool = False,
    device: torch.device | str = "cpu",
) -> dict:
    """Cross-dataset transition stress-test that bypasses sensor-count mismatch.

    `make_transition_segment` requires the two source segments to share
    sensor counts (D1 4 mics ≠ D2 5 mics ≠ D3/D4 9 mics).  This helper
    instead encodes each segment independently into per-window
    `(x, c)` tuples, then linearly crossfades the **encoder outputs** —
    the resulting transition windows are the linear path between two
    segments' c_t representations in latent space, healthy by construction
    at the endpoints.  The FPR over the crossfade region is the same
    diagnostic V3 should pass.
    """
    device = torch.device(device)
    v2_encoder = v2_encoder.to(device).eval()
    flow = flow.to(device).eval()

    # Score each segment to get (x, c) per window.
    a_scores, a_contexts, _ = score_segments(
        v2_encoder, flow, [seg_a], v2_cfg=v2_cfg,
        unconditional=unconditional, device=device,
    )
    b_scores, b_contexts, _ = score_segments(
        v2_encoder, flow, [seg_b], v2_cfg=v2_cfg,
        unconditional=unconditional, device=device,
    )
    if a_contexts.shape[0] == 0 or b_contexts.shape[0] == 0:
        return {"n_windows": 0, "n_alerts": 0, "fpr": 0.0}

    # Build the transition cohort: take the last K windows of A and the
    # first K windows of B, then linearly interpolate between them in
    # latent space to synthesise N_crossfade transition windows.
    K = min(n_crossfade_windows, a_contexts.shape[0], b_contexts.shape[0])
    if K <= 0:
        return {"n_windows": 0, "n_alerts": 0, "fpr": 0.0}
    a_tail_c = a_contexts[-K:]
    b_head_c = b_contexts[:K]

    # Re-extract the matching `x` (mean-pool of fused tokens) for each
    # source segment by running the encoder again.  Cheaper alternative:
    # cache `x` alongside `c` from `score_segments`, but the helper only
    # exposes `c`.  For diagnostic rigour we recompute here.
    a_x = _extract_x_for_segment(v2_encoder, seg_a, v2_cfg, device)[-K:]
    b_x = _extract_x_for_segment(v2_encoder, seg_b, v2_cfg, device)[:K]

    weights = np.linspace(0.0, 1.0, K, dtype=np.float32)
    transition_x = (1.0 - weights[:, None]) * a_x + weights[:, None] * b_x
    transition_c = (1.0 - weights[:, None]) * a_tail_c + weights[:, None] * b_head_c

    if unconditional:
        c_for_flow = np.zeros_like(transition_c)
    else:
        c_for_flow = transition_c
    with torch.no_grad():
        scores = flow.anomaly_score(
            torch.from_numpy(transition_x).float().to(device),
            torch.from_numpy(c_for_flow).float().to(device),
        ).cpu().numpy()
    alerts, _ = thresholds.alert(transition_c, scores, percentile=percentile)
    return {
        "n_windows": int(scores.shape[0]),
        "n_alerts": int(alerts.sum()),
        "fpr": float(alerts.mean()),
        "scores": scores,
    }


def _extract_x_for_segment(
    v2_encoder: V2FusionEncoder,
    seg: _PairedSegment,
    cfg: V2SSLConfig,
    device: torch.device,
) -> np.ndarray:
    """Run V2 forward over every window in `seg`; return mean-pool x array."""
    ds = _PairedWindowedDataset([seg], cfg)
    if len(ds) == 0:
        return np.zeros((0, v2_encoder.embed_dim), dtype=np.float32)
    sampler = _PairedGroupedBatchSampler(ds, cfg.batch_size, shuffle=False, seed=0)
    loader = tud.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)
    xs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = v2_encoder(
                batch["ac_feat"].to(device), batch["ac_xyz"].to(device),
                batch["vib_feat"].to(device), batch["vib_xyz"].to(device),
                batch["dataset_idx"].to(device), mask_p=0.0,
            )
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            xs.append(fused.mean(dim=1).cpu().numpy())
    return np.concatenate(xs, axis=0).astype(np.float32)


def transition_fpr(
    v2_encoder: V2FusionEncoder,
    flow: ConditionalRealNVP,
    thresholds: PerClusterThresholds,
    seg_a: _PairedSegment,
    seg_b: _PairedSegment,
    *,
    v2_cfg: V2SSLConfig,
    crossfade_seconds: float = 1.0,
    percentile: int = 99,
    unconditional: bool = False,
    device: torch.device | str = "cpu",
) -> dict:
    """Splice (A → crossfade → B), score every window, return the FPR.

    The transition is *healthy by construction* (both endpoints are healthy
    segments of different modes), so any alert is a false positive.  This is
    the V3 headline "FPR-on-transitions" metric.
    """
    spliced = make_transition_segment(seg_a, seg_b, crossfade_seconds=crossfade_seconds)
    scores, contexts, _labels = score_segments(
        v2_encoder,
        flow,
        [spliced],
        v2_cfg=v2_cfg,
        unconditional=unconditional,
        device=device,
    )
    if scores.shape[0] == 0:
        return {
            "n_windows": 0,
            "n_alerts": 0,
            "fpr": 0.0,
            "scores": scores,
            "clusters": np.zeros(0, dtype=np.int64),
        }
    alerts, clusters = thresholds.alert(contexts, scores, percentile=percentile)
    return {
        "n_windows": int(scores.shape[0]),
        "n_alerts": int(alerts.sum()),
        "fpr": float(alerts.mean()),
        "scores": scores,
        "clusters": clusters,
    }


__all__ = [
    "V3Config",
    "V3Result",
    "train_v3_cnf",
    "score_segments",
    "make_transition_segment",
    "transition_fpr",
    "precompute_paired",
    "gate_samples_by_alert",
]
