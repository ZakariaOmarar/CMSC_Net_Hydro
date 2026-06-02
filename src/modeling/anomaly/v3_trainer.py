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

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
from tqdm.auto import tqdm

from ...config import describe_device, resolve_device
from ...config.architecture import V3_ANOMALY
from ...ingestion.test_dataset_loader import TestDatasetLoader
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import (
    V2SSLConfig,
    _collate,
    _gather_paired_segments,
    _PairedGroupedBatchSampler,
    _PairedSegment,
    _PairedWindowedDataset,
    _precompute_paired,
    _split_segments_by_recording,
)
from ..encoders.set_transformer import PMA
from .cnf_head import ConditionalRealNVP
from .threshold import PerClusterThresholds

XtPoolKind = Literal["mean", "pma2"]


class _XtPool(nn.Module):
    """Learnable channel-token pool for V3's `x_t` extraction.

    Replaces the legacy ``fused.mean(dim=1)`` (single first-moment average
    over the `(N_a + N_v)` channel-token axis) with two learned attention
    seeds (PMA, Lee et al. ICML 2019), concatenated and projected back to
    ``embed_dim``.

    Motivation (2026-05-19):

      The legacy mean pool was the **second** dilution stage for V3 at
      ``hop_length=43``: a knock signature that survived the per-mic
      ``AdaptiveAvgPool2d`` only contributes ``1/(N_a + N_v)`` to the
      channel-mean.  PMA-2 lets V3 jointly model a "stationary-mode-
      consistent" attention pattern (uniform weights) and a "transient-
      event-localized" attention pattern (mass on whichever mics see the
      knock peak) — two seeds is the minimum count strictly greater than
      V2's existing PMA-1 context pool, so the V3 ``x_t`` carries
      complementary information rather than a rescaled copy of ``c_t``.

      Trained jointly with the conditional flow on the frozen V2 encoder
      output.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pma = PMA(embed_dim, num_seeds=2, num_heads=num_heads)
        self.proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, N_a + N_v, embed_dim)
        pooled = self.pma(fused)                 # (B, 2, embed_dim)
        flat = pooled.flatten(start_dim=1)       # (B, 2 * embed_dim)
        return self.proj(flat)                   # (B, embed_dim)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3Config:
    """V3 conditional anomaly head config."""

    # Per-stage window override (2026-05-19).  When set, the V3 trainer
    # builds a shallow copy of the V2 config with
    # `window_scales_seconds=(override,)` (single-scale) before
    # constructing `_PairedWindowedDataset`.  The frozen V2 encoder is
    # scale-invariant by virtue of ASP pooling (Okabe 2018), so the
    # smaller window is consumed without re-pretraining V2.  May be:
    #
    #   * ``None`` — inherit V2's scale set (legacy behaviour),
    #   * a ``float`` — single override applied to every dataset, or
    #   * a ``dict[dataset_id, float]`` — per-dataset override.  Publication
    #     defaults are 1.0 s on D3/D4 (transient-tight; the smallest sub-2 s
    #     scale where ASP-σ still has ≥ 30 post-MaxPool frames at hop=43)
    #     and 3.0 s on D1/D2 (which cannot go below 1.5 s).
    # Default ``None`` keeps V3 backwards-compatible (inherits v2_cfg
    # window).  The orchestrator (full_run._v3_cfg) sets the publication
    # per-dataset dict from `WINDOWING.v3_window_seconds_override`.
    window_seconds_override: float | dict[str, float] | None = None

    # V3 channel-token pool kind for `x_t`.  Mean-pool dilutes transient
    # signatures on the channel-token axis; PMA-2 (the publication
    # default) learns 2 attention seeds that capture both stationary-mode
    # and transient-event patterns.  See `_XtPool` docstring above for the
    # full justification.
    xt_pool: XtPoolKind = V3_ANOMALY.xt_pool
    xt_pool_num_heads: int = V3_ANOMALY.xt_pool_num_heads

    # CNF dims — sourced from `V3_ANOMALY` in architecture.py.
    n_layers: int = V3_ANOMALY.n_layers
    hidden_dim: int = V3_ANOMALY.hidden_dim
    n_hidden_per_net: int = 2   # CNF coupling MLP depth; not centralised
    scale_max: float = V3_ANOMALY.scale_max

    # Training schedule — per-experiment, not centralised.
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.3
    use_cosine_lr: bool = True

    # A2 ablation — zero c at train+infer for unconditional flow.
    unconditional: bool = False

    # Threshold fit — fully unsupervised on healthy data.
    n_threshold_clusters: int = V3_ANOMALY.n_threshold_clusters
    threshold_percentile: int = V3_ANOMALY.threshold_percentile

    # Nested held-out split inside `val_segs`: a `threshold_fit_val_ratio`
    # fraction of val recordings goes to *fitting* the K-means centroids and
    # per-cluster percentile bar; the remaining fraction is the *reportable*
    # held-out cohort whose NLL and alert rate are quoted in the thesis.
    # Without this split the same windows define the clusters, set the bar,
    # and are scored against the bar — per-cluster p95/p99 are optimistically
    # biased.  Split is by recording (uses `_split_segments_by_recording`),
    # so windows from a single recording cannot span both halves.
    threshold_fit_val_ratio: float = 0.5
    # The Youden's-J calibration helper exists in `threshold.py` for
    # post-hoc analysis but is **NOT** wired into the orchestrator.  It
    # would require per-window anomaly labels (or an assumption that all
    # D2 RF / D3 hit windows are anomalous), which the field-collection
    # protocol does not provide.  Threshold quality is instead validated
    # post-hoc via per-cohort alert rates (the orchestrator's
    # `v3_alert_rate_per_cohort` metric).
    calibrate_with_anomalies: bool = False

    # Early stopping on val mean NLL.  Patience=5 (vs V1/V2's 3) because the
    # CNF's val NLL is noisier (4-recording fit cohort + outlier batches —
    # see val_nll_max audit signal).  Restore-best restores the FLOW state;
    # the V2 encoder is frozen so nothing to restore there.  Tensor-clone
    # snapshot (NOT copy.deepcopy) — see V1 trainer for rationale.
    patience: int = 5
    restore_best: bool = True
    early_stop_min_delta: float = 1e-3  # NLL units; flow loss is O(10-100)

    # CNF coupling MLP dropout — defends against the +56 % train/val NLL gap
    # the audit identified.  Default 0.0 keeps the dataclass byte-equivalent
    # to pre-fix behaviour; the orchestrator `_v3_cfg` builder sets 0.1.
    dropout_p: float = 0.0

    # System
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
# Encoder feature extraction
# ---------------------------------------------------------------------------


def _extract_xc(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    device: torch.device,
    xt_pool: nn.Module | None = None,
    grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run frozen V2 encoder forward; collect x_t, c_t, mode labels.

    When ``xt_pool`` is None (legacy path) or an ``nn.Identity``-like marker,
    ``x_t = fused.mean(dim=1)`` — preserves the original `cfg.xt_pool="mean"`
    behaviour.  When ``xt_pool`` is an :class:`_XtPool`, the V2 encoder is run
    under ``no_grad`` (it is frozen) but the ``xt_pool`` forward is on the
    autograd tape so the V3 trainer can co-optimise it with the flow.

    The ``grad`` flag controls whether ``xt_pool`` is invoked under
    ``torch.no_grad`` (set to True at training time, False for inference /
    feature extraction).
    """
    encoder.eval()
    xs: list[torch.Tensor] = []
    cs: list[torch.Tensor] = []
    labels: list[str] = []
    use_grad = bool(grad and xt_pool is not None)
    for batch in loader:
        ac = batch["ac_feat"].to(device)
        vib = batch["vib_feat"].to(device)
        ac_xyz = batch["ac_xyz"].to(device)
        vib_xyz = batch["vib_xyz"].to(device)
        ds_idx = batch["dataset_idx"].to(device)
        with torch.no_grad():
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1)
            c_t = out["context"]
        if xt_pool is not None:
            if use_grad:
                x_t = xt_pool(fused)
            else:
                with torch.no_grad():
                    x_t = xt_pool(fused)
        else:
            x_t = fused.mean(dim=1)
        xs.append(x_t.detach().cpu())
        cs.append(c_t.detach().cpu())
        labels.extend(batch["mode_label"])
    return torch.cat(xs, dim=0), torch.cat(cs, dim=0), labels


def _resolve_v3_override(
    v3_cfg: V3Config, dataset_ids: Iterable[str]
) -> dict[str, float] | None:
    """Normalise the V3 ``window_seconds_override`` to a per-dataset dict.

    Returns None when no override is set; otherwise a fully-populated
    dict keyed by the dataset ids the trainer is about to consume.
    """
    override = v3_cfg.window_seconds_override
    if override is None:
        return None
    if isinstance(override, dict):
        return {str(k): float(v) for k, v in override.items()}
    return {str(d): float(override) for d in dataset_ids}


def _make_override_v2_cfg(
    v2_cfg: V2SSLConfig,
    override_per_dataset: dict[str, float] | None,
) -> V2SSLConfig:
    """Return a shallow `V2SSLConfig` copy with the V3 override applied.

    The override is materialised as a per-dataset
    ``window_scales_seconds_per_dataset`` dict so the V3
    ``_PairedWindowedDataset`` reuses the same multi-scale plumbing as
    V1 / V2 SSL.  When ``override_per_dataset`` is None the original
    config is returned unchanged.
    """
    if not override_per_dataset:
        return v2_cfg
    per_ds = {ds: (float(v),) for ds, v in override_per_dataset.items()}
    return replace(v2_cfg, window_scales_seconds_per_dataset=per_ds)


def _cache_fused(
    encoder: V2FusionEncoder,
    loader: tud.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Run the frozen V2 encoder once and cache per-batch (fused, c, labels).

    Used by the ``xt_pool="pma2"`` path so the learnable :class:`_XtPool`
    can be co-optimised with the flow over many epochs without re-running
    the encoder.  Each cache entry preserves the grouped-batch contract
    (uniform `N_a + N_v` within a batch) so `_XtPool(fused)` succeeds
    without padding masks.

    Memory cost is proportional to ``num_windows * (N_a + N_v) * embed_dim``;
    at the typical V3 cohort size (~10 k windows, 13 channels, 128 dims)
    this is ~67 MB on CPU — well below RAM budgets.
    """
    encoder.eval()
    cache: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            ac = batch["ac_feat"].to(device)
            vib = batch["vib_feat"].to(device)
            ac_xyz = batch["ac_xyz"].to(device)
            vib_xyz = batch["vib_xyz"].to(device)
            ds_idx = batch["dataset_idx"].to(device)
            out = encoder(ac, ac_xyz, vib, vib_xyz, ds_idx, mask_p=0.0)
            fused = torch.cat([out["a_fused"], out["v_fused"]], dim=1).detach().cpu()
            c = out["context"].detach().cpu()
            cache.append(
                {
                    "fused": fused,
                    "c": c,
                    "labels": list(batch["mode_label"]),
                }
            )
    return cache


def _pool_cached_x(
    cache: list[dict],
    xt_pool: nn.Module | None,
    device: torch.device,
) -> torch.Tensor:
    """Apply ``xt_pool`` (or the legacy mean-pool) over a fused cache.

    Always runs under ``no_grad`` — used for the final extract-once paths
    (val NLL evaluation, threshold fitting, post-training scoring).
    """
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for entry in cache:
            fused = entry["fused"].to(device)
            if xt_pool is None:
                x = fused.mean(dim=1)
            else:
                x = xt_pool(fused)
            chunks.append(x.detach().cpu())
    return torch.cat(chunks, dim=0)


def _stack_c_from_cache(cache: list[dict]) -> torch.Tensor:
    return torch.cat([entry["c"] for entry in cache], dim=0)


def _stack_labels_from_cache(cache: list[dict]) -> list[str]:
    out: list[str] = []
    for entry in cache:
        out.extend(entry["labels"])
    return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


@dataclass
class V3Result:
    flow: ConditionalRealNVP
    thresholds: PerClusterThresholds
    train_nll: list[float]
    val_nll: list[float]
    # F6 — per-epoch outlier-batch tracking (min / max of per-batch
    # train NLL and per-window val NLL).  Spread between max and the
    # mean diagnoses single-batch loss spikes hiding in the average.
    train_nll_min: list[float]
    train_nll_max: list[float]
    val_nll_min: list[float]
    val_nll_max: list[float]
    train_recording_ids: list[str]
    # Reportable held-out cohort — disjoint from train AND from the cohort
    # used to fit per-cluster thresholds.  All quoted val numbers
    # (`val_scores`, `val_contexts`, `val_labels`, per-epoch `val_nll`)
    # come from this cohort.
    val_recording_ids: list[str]
    # Cohort used to fit K-means centroids + per-cluster p95/p99 — disjoint
    # from the reportable val cohort.  Exposed for transparency / auditing.
    threshold_fit_recording_ids: list[str]
    val_scores: np.ndarray
    val_contexts: np.ndarray
    val_labels: list[str]
    unconditional: bool
    # `_XtPool` module trained jointly with the flow.  None when the legacy
    # mean-pool path was used (`xt_pool="mean"`).  Carries learned PMA-2
    # weights that scoring / streaming inference must reuse to obtain
    # comparable scores on new windows.
    xt_pool: nn.Module | None = None
    early_stopped_epoch: int | None = None
    best_val_nll: float = float("nan")
    # Cached x / c arrays for the deep-vs-simple comparison (KDE-on-c_t in
    # `kde_baseline.py`).  These are the same arrays the flow trained on /
    # was evaluated against, so any KDE built from them is apples-to-apples
    # with the CNF's NLL.  Stored on CPU as numpy to keep V3Result picklable.
    train_x: np.ndarray | None = None
    train_contexts: np.ndarray | None = None
    val_x: np.ndarray | None = None


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
    device = resolve_device(v3_cfg.device)
    print(f"V3: device={describe_device(device)}")
    v2_encoder = v2_encoder.to(device)
    v2_encoder.eval()
    for p in v2_encoder.parameters():
        p.requires_grad_(False)

    segments = _gather_paired_segments(loaders, v2_cfg)
    if not segments:
        raise RuntimeError("V3: no healthy paired segments found")

    # Apply the per-stage window override (2026-05-19).  Materialised as a
    # per-dataset `window_scales_seconds_per_dataset` dict so the existing
    # multi-scale dataset plumbing handles it — the override degenerates to
    # "single scale, equal to override" for each dataset present.
    dataset_ids = sorted({s.dataset_id for s in segments})
    override_per_ds = _resolve_v3_override(v3_cfg, dataset_ids)
    effective_v2_cfg = _make_override_v2_cfg(v2_cfg, override_per_ds)

    train_segs, val_segs = _split_segments_by_recording(
        segments, v3_cfg.val_ratio, v3_cfg.seed
    )
    # Nested split inside the held-out val cohort: a `threshold_fit_val_ratio`
    # fraction of val recordings is reserved for K-means + percentile fit;
    # the remaining fraction is the reportable held-out cohort.  Seed offset
    # by +1 to avoid coupling the two permutations.
    val_fit_segs, val_eval_segs = _split_segments_by_recording(
        val_segs, v3_cfg.threshold_fit_val_ratio, v3_cfg.seed + 1
    )
    train_ds = _PairedWindowedDataset(train_segs, effective_v2_cfg)
    val_fit_ds = _PairedWindowedDataset(val_fit_segs, effective_v2_cfg)
    val_eval_ds = _PairedWindowedDataset(val_eval_segs, effective_v2_cfg)
    if len(train_ds) == 0:
        raise RuntimeError("V3: zero training windows after splitting")
    if len(val_fit_ds) == 0 or len(val_eval_ds) == 0:
        raise RuntimeError(
            "V3: threshold-fit nested split produced an empty cohort "
            f"(val_fit={len(val_fit_ds)}, val_eval={len(val_eval_ds)}); "
            "reduce `threshold_fit_val_ratio` or increase `val_ratio`."
        )

    pin = device.type == "cuda"
    train_loader = tud.DataLoader(
        train_ds,
        batch_sampler=_PairedGroupedBatchSampler(train_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_fit_loader = tud.DataLoader(
        val_fit_ds,
        batch_sampler=_PairedGroupedBatchSampler(val_fit_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )
    val_eval_loader = tud.DataLoader(
        val_eval_ds,
        batch_sampler=_PairedGroupedBatchSampler(val_eval_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed),
        collate_fn=_collate,
        pin_memory=pin,
    )

    # ----- xt_pool wiring -----
    # ``pma2`` is the publication default (Lee et al. ICML 2019 PMA, with
    # 2 seeds; see `_XtPool` docstring).  ``mean`` reproduces the legacy
    # mean-pool path one-for-one for ablation / hop=512 reproducibility.
    if v3_cfg.xt_pool == "pma2":
        # Cache the fused tokens once so the joint flow + xt_pool training
        # loop avoids re-running the encoder every epoch (encoder is frozen).
        train_cache = _cache_fused(v2_encoder, train_loader, device)
        val_fit_cache = _cache_fused(v2_encoder, val_fit_loader, device)
        val_eval_cache = _cache_fused(v2_encoder, val_eval_loader, device)
        embed_dim = int(train_cache[0]["fused"].shape[-1])
        xt_pool: nn.Module | None = _XtPool(
            embed_dim=embed_dim, num_heads=int(v3_cfg.xt_pool_num_heads)
        ).to(device)
    elif v3_cfg.xt_pool == "mean":
        train_cache = None
        val_fit_cache = None
        val_eval_cache = None
        xt_pool = None
    else:
        raise ValueError(f"unknown xt_pool {v3_cfg.xt_pool!r}")

    if xt_pool is None:
        # Legacy mean-pool: extract once, then train flow on tensors.
        x_train, c_train, _ = _extract_xc(v2_encoder, train_loader, device)
        x_val_fit, c_val_fit, _val_fit_labels = _extract_xc(v2_encoder, val_fit_loader, device)
        x_val, c_val, val_labels = _extract_xc(v2_encoder, val_eval_loader, device)
    else:
        # PMA-2 path: initial x is the pool's random-init output; the
        # per-epoch loop below re-runs the pool with current weights.  We
        # still pre-compute c (V2's PMA context, frozen) once.
        c_train = _stack_c_from_cache(train_cache)
        c_val_fit = _stack_c_from_cache(val_fit_cache)
        c_val = _stack_c_from_cache(val_eval_cache)
        val_labels = _stack_labels_from_cache(val_eval_cache)
        # Seed x tensors so flow dimensionality can be inferred below.
        # These tensors are NOT used in training; the training loop re-pools
        # each epoch from `train_cache`.  We compute them with the pool's
        # initialisation so flow.__init__ sees the right shape.
        x_train = _pool_cached_x(train_cache, xt_pool, device)
        x_val_fit = _pool_cached_x(val_fit_cache, xt_pool, device)
        x_val = _pool_cached_x(val_eval_cache, xt_pool, device)

    if v3_cfg.unconditional:
        c_train_used = torch.zeros_like(c_train)
        c_val_fit_used = torch.zeros_like(c_val_fit)
        c_val_used = torch.zeros_like(c_val)
    else:
        c_train_used = c_train
        c_val_fit_used = c_val_fit
        c_val_used = c_val

    flow = ConditionalRealNVP(
        dim=int(x_train.shape[1]),
        c_dim=int(c_train.shape[1]),
        n_layers=v3_cfg.n_layers,
        hidden_dim=v3_cfg.hidden_dim,
        n_hidden_per_net=v3_cfg.n_hidden_per_net,
        scale_max=v3_cfg.scale_max,
        dropout_p=v3_cfg.dropout_p,
    ).to(device)
    # Co-optimise the flow with `_XtPool` when present (publication default).
    trainable_params: list[torch.nn.Parameter] = list(flow.parameters())
    if xt_pool is not None:
        trainable_params += list(xt_pool.parameters())
    optim = torch.optim.AdamW(
        trainable_params, lr=v3_cfg.lr, weight_decay=v3_cfg.weight_decay
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
    # F6 — per-epoch outlier-batch tracking.  A coupling layer can occasionally
    # produce a single batch with very large NLL (an unbounded affine
    # transform on an OOD x); we track min/max so the orchestrator can flag
    # epochs where a single batch dominates the mean loss.
    train_nll_min: list[float] = []
    train_nll_max: list[float] = []
    val_nll_min: list[float] = []
    val_nll_max: list[float] = []

    # Early-stop bookkeeping — tensor-clone snapshot, NOT copy.deepcopy.
    # See V1 trainer for the RAM-fragmentation rationale.  Snapshot covers
    # both `flow` and (when present) the learnable `xt_pool` since they are
    # co-optimised; restoring just the flow would leave the pool at a
    # non-best state.
    def _snapshot_combined() -> dict[str, dict[str, torch.Tensor]]:
        out = {"flow": {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}}
        if xt_pool is not None:
            out["xt_pool"] = {
                k: v.detach().cpu().clone() for k, v in xt_pool.state_dict().items()
            }
        return out

    best_val_nll = float("inf")
    best_state = _snapshot_combined()
    epochs_without_improvement = 0
    early_stopped_epoch: int | None = None

    suffix = "unconditional" if v3_cfg.unconditional else "conditional"
    epoch_iter = tqdm(
        range(v3_cfg.epochs),
        desc=f"V3 CNF ({suffix})",
        unit="epoch",
        leave=False,
    )
    for _epoch in epoch_iter:
        flow.train()
        if xt_pool is not None:
            xt_pool.train()
        loss_sum = 0.0
        n = 0
        epoch_min = float("inf")
        epoch_max = float("-inf")
        if xt_pool is None:
            # Legacy path — tensor-only.
            perm = torch.randperm(n_train)
            for i in range(0, n_train, v3_cfg.batch_size):
                idx = perm[i : i + v3_cfg.batch_size]
                xb = x_train[idx].to(device)
                cb = c_train_used[idx].to(device)
                log_p = flow.log_prob(xb, cb)
                loss = -log_p.mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
                optim.step()
                loss_f = float(loss.item())
                loss_sum += loss_f * xb.shape[0]
                n += xb.shape[0]
                if loss_f < epoch_min:
                    epoch_min = loss_f
                if loss_f > epoch_max:
                    epoch_max = loss_f
        else:
            # PMA-2 path — iterate over the cached fused batches in shuffled
            # order; re-pool with the current xt_pool weights every step so
            # `_XtPool` is co-optimised with the flow.
            order = torch.randperm(len(train_cache)).tolist()
            cum_idx = 0
            for batch_pos in order:
                entry = train_cache[batch_pos]
                fused = entry["fused"].to(device)
                B = int(fused.shape[0])
                # Slice the matching c rows from c_train_used (which is a
                # CPU tensor concatenated in cache order — see `_cache_fused`).
                cb = c_train_used[cum_idx : cum_idx + B].to(device)
                cum_idx += B
                xb = xt_pool(fused)
                log_p = flow.log_prob(xb, cb)
                loss = -log_p.mean()
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optim.step()
                loss_f = float(loss.item())
                loss_sum += loss_f * B
                n += B
                if loss_f < epoch_min:
                    epoch_min = loss_f
                if loss_f > epoch_max:
                    epoch_max = loss_f
        if scheduler is not None:
            scheduler.step()
        train_nll.append(loss_sum / max(1, n))
        train_nll_min.append(epoch_min if epoch_min != float("inf") else float("nan"))
        train_nll_max.append(epoch_max if epoch_max != float("-inf") else float("nan"))

        flow.eval()
        if xt_pool is not None:
            xt_pool.eval()
        with torch.no_grad():
            if xt_pool is not None:
                # Re-pool the cached val_eval with current weights.
                x_val_epoch = _pool_cached_x(val_eval_cache, xt_pool, device)
            else:
                x_val_epoch = x_val
            v_log_p = flow.log_prob(x_val_epoch.to(device), c_val_used.to(device))
            v_nll_per_window = (-v_log_p).cpu().numpy()
            val_nll.append(float(v_nll_per_window.mean()))
            val_nll_min.append(float(v_nll_per_window.min()) if v_nll_per_window.size else float("nan"))
            val_nll_max.append(float(v_nll_per_window.max()) if v_nll_per_window.size else float("nan"))
        epoch_iter.set_postfix(
            train_nll=f"{train_nll[-1]:.3f}", val_nll=f"{val_nll[-1]:.3f}"
        )

        # Early stop on val mean NLL.  Snapshot flow + xt_pool jointly.
        cur_val = val_nll[-1]
        if cur_val < best_val_nll - v3_cfg.early_stop_min_delta:
            best_val_nll = cur_val
            best_state = _snapshot_combined()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= v3_cfg.patience:
                early_stopped_epoch = _epoch + 1
                break

    if v3_cfg.restore_best:
        flow.load_state_dict(best_state["flow"])
        if xt_pool is not None and "xt_pool" in best_state:
            xt_pool.load_state_dict(best_state["xt_pool"])
    del best_state

    flow.eval()
    if xt_pool is not None:
        xt_pool.eval()
        # Re-pool with final pool weights so threshold-fit and held-out scores
        # both reflect the converged `_XtPool` state.
        x_val_fit_final = _pool_cached_x(val_fit_cache, xt_pool, device)
        x_val_final = _pool_cached_x(val_eval_cache, xt_pool, device)
    else:
        x_val_fit_final = x_val_fit
        x_val_final = x_val
    with torch.no_grad():
        # Threshold-fit cohort scores (used only to set the percentile bar).
        scores_val_fit = (
            flow.anomaly_score(x_val_fit_final.to(device), c_val_fit_used.to(device))
            .cpu()
            .numpy()
        )
        # Reportable held-out cohort scores (all downstream metrics).
        scores_val = (
            flow.anomaly_score(x_val_final.to(device), c_val_used.to(device)).cpu().numpy()
        )

    # Threshold fitting always clusters on the *real* `c_t` (label-free);
    # the unconditional flag only affects the flow itself.  Critically, the
    # cohort feeding `.fit()` is `val_fit`, which is disjoint from the
    # `val_eval` cohort whose scores are returned for reporting — this is
    # what closes the F1 leakage path.
    n_clusters = min(v3_cfg.n_threshold_clusters, max(1, c_val_fit.shape[0]))
    thresholds = PerClusterThresholds.fit(
        c_val_fit.numpy(),
        scores_val_fit,
        n_clusters=n_clusters,
        seed=v3_cfg.seed,
    )

    def _qualify(seg: _PairedSegment) -> str:
        return f"{Path(seg.source_dir).name}/{seg.recording_id}"

    # Cache x / c arrays for the deep-vs-simple KDE comparison.  Re-pool the
    # train cohort with the FINAL xt_pool weights so the cached x_train is
    # consistent with x_val_final the flow was just evaluated on.
    if xt_pool is not None:
        with torch.no_grad():
            x_train_final = _pool_cached_x(train_cache, xt_pool, device)
    else:
        x_train_final = x_train

    return V3Result(
        flow=flow,
        thresholds=thresholds,
        train_nll=train_nll,
        val_nll=val_nll,
        train_nll_min=train_nll_min,
        train_nll_max=train_nll_max,
        val_nll_min=val_nll_min,
        val_nll_max=val_nll_max,
        train_recording_ids=sorted({_qualify(s) for s in train_segs}),
        val_recording_ids=sorted({_qualify(s) for s in val_eval_segs}),
        threshold_fit_recording_ids=sorted({_qualify(s) for s in val_fit_segs}),
        val_scores=scores_val,
        val_contexts=c_val.numpy(),
        val_labels=val_labels,
        unconditional=v3_cfg.unconditional,
        xt_pool=xt_pool,
        early_stopped_epoch=early_stopped_epoch,
        best_val_nll=best_val_nll,
        train_x=x_train_final.detach().cpu().numpy(),
        train_contexts=c_train.detach().cpu().numpy(),
        val_x=x_val_final.detach().cpu().numpy(),
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
    device: torch.device | str = "auto",
    xt_pool: nn.Module | None = None,
    window_seconds_override: float | dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Score a list of paired segments with the trained flow.

    Returns ``(scores, contexts, mode_labels)`` aligned per window.

    ``xt_pool`` is the learned ``_XtPool`` from
    :class:`V3Result.xt_pool` — pass it through so inference reuses the
    same per-window summary the flow was trained on.  When None, the
    legacy mean-pool path is used (only valid if the flow was trained
    with ``xt_pool="mean"``).

    ``window_seconds_override`` mirrors :attr:`V3Config.window_seconds_override`
    and is applied to ``v2_cfg`` before the dataset is constructed.
    """
    device = resolve_device(device)
    v2_encoder = v2_encoder.to(device).eval()
    flow = flow.to(device).eval()
    if xt_pool is not None:
        xt_pool = xt_pool.to(device).eval()

    # Materialise the override as a per-dataset dict before constructing the
    # dataset, so the V2 paired dataset's wall-clock pairing is preserved.
    if window_seconds_override is not None:
        dataset_ids = sorted({s.dataset_id for s in segments})
        if isinstance(window_seconds_override, dict):
            override_per_ds = {str(k): float(v) for k, v in window_seconds_override.items()}
        else:
            override_per_ds = {str(d): float(window_seconds_override) for d in dataset_ids}
        effective_cfg = _make_override_v2_cfg(v2_cfg, override_per_ds)
    else:
        effective_cfg = v2_cfg

    ds = _PairedWindowedDataset(segments, effective_cfg)
    if len(ds) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, flow.c_dim), dtype=np.float64), []
    loader = tud.DataLoader(
        ds,
        batch_sampler=_PairedGroupedBatchSampler(ds, batch_size, shuffle=False, seed=0),
        collate_fn=_collate,
    )

    x, c, labels = _extract_xc(v2_encoder, loader, device, xt_pool=xt_pool, grad=False)
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
    device: torch.device | str = "auto",
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

    device = resolve_device(device)
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
        dataset_id=getattr(seg_a, "dataset_id", "synthetic"),
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
    device: torch.device | str = "auto",
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
    device = resolve_device(device)
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
    device: torch.device | str = "auto",
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
    "gate_samples_by_alert",
    "make_transition_segment",
    "precompute_paired",
    "score_segments",
    "train_v3_cnf",
    "transition_fpr",
]
