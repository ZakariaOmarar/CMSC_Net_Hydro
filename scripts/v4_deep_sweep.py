"""Deep V4 localization sweep — Phase 2 of the V3-first deep campaign.

Runs AFTER the Phase-1 V3 winner is chosen, because V4 is gated by V3.
Trains the V4 head individually against a FROZEN V2 encoder (samples cached
once; ~10 min/cell) and evaluates on the **held-out positions** (localise-an-
unseen-position), **gated by the Phase-1 V3** (deployment-faithful: V4 only
fires on V3-flagged windows), and compares against V0 multilateration on the
same held-out positions.

Selection objective (gap is a guardrail, not the target): minimize the
**V3-gated holdout MAE**, **subject to** ``|val_mae − train_mae| ≤ guardrail``.
Report Δ-vs-V0; if no cell beats V0 even with expanded data + gating, that is
the rigorously-documented N-limited ceiling.

Axes (superset of v4_aug_sweep):
  Regularization: head_dropout_p {0.0,0.1,0.2,0.3} × weight_decay {1e-4,5e-4,1e-3}
    v4_hd{0,1,2,3}_w{4,5,3}
  Capacity (cnn_feature_dim, hidden_dim):
    v4_cap_small (32,32)  v4_cap_base (64,64)  v4_cap_big (128,128)
  Residual half-range:
    v4_rs10 (0.10)  v4_rs20 (0.20)  v4_rs30 (0.30)
  Augmentation (target_pos_noise × srp_volume_noise):
    v4_pos{1,5,10}_srp{02,10,20}

Run::

    python -m scripts.v4_deep_sweep --encoder-run <dir> --v3-run <phase1_v3_winner_dir>
    python -m scripts.v4_deep_sweep --encoder-run <dir> --v3-run <dir> --cell v4_hd2_w5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly.cnf_head import ConditionalRealNVP
from src.modeling.anomaly.threshold import PerClusterThresholds
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.localization import (
    GridSpec,
    precompute_v4_samples,
    split_samples_by_position,
    train_v4_localization,
)
from src.modeling.orchestration.full_run import (
    REPO_ROOT,
    V4_HOLDOUT_POSITIONS_M,
    _d3_spatial_overrides,
    _resolved_loader,
    _v2_cfg,
    _v3_cfg,
    _v4_cfg,
    _v3_event_intervals_for_recordings,
)


_DROPOUT_LEVELS = {"hd0": 0.0, "hd1": 0.1, "hd2": 0.2, "hd3": 0.3}
_WD_LEVELS = {"w4": 1e-4, "w5": 5e-4, "w3": 1e-3}
_CAP_LEVELS = {"small": (32, 32), "base": (64, 64), "big": (128, 128)}
_RS_LEVELS = {"rs10": 0.10, "rs20": 0.20, "rs30": 0.30}
_POS_LEVELS = {"pos1": 0.002, "pos5": 0.010, "pos10": 0.020}
_SRP_LEVELS = {"srp02": 0.02, "srp10": 0.10, "srp20": 0.20}


def _all_cells() -> list[str]:
    reg = [f"v4_{d}_{w}" for d in _DROPOUT_LEVELS for w in _WD_LEVELS]
    cap = [f"v4_cap_{c}" for c in _CAP_LEVELS]
    rs = [f"v4_{r}" for r in _RS_LEVELS]
    aug = [f"v4_{p}_{s}" for p in _POS_LEVELS for s in _SRP_LEVELS]
    return reg + cap + rs + aug


def _apply_cell(cell_id: str, v4_cfg):
    if cell_id.startswith("v4_cap_"):
        key = cell_id[len("v4_cap_"):]
        cnn, hidden = _CAP_LEVELS[key]
        return replace(v4_cfg, cnn_feature_dim=cnn, hidden_dim=hidden)
    parts = cell_id.split("_")
    if len(parts) == 2 and parts[1] in _RS_LEVELS:  # v4_rs10
        return replace(v4_cfg, residual_scale_m=_RS_LEVELS[parts[1]])
    if len(parts) == 3 and parts[1] in _DROPOUT_LEVELS and parts[2] in _WD_LEVELS:
        return replace(v4_cfg, head_dropout_p=_DROPOUT_LEVELS[parts[1]],
                       weight_decay=_WD_LEVELS[parts[2]])
    if len(parts) == 3 and parts[1] in _POS_LEVELS and parts[2] in _SRP_LEVELS:
        return replace(v4_cfg, target_pos_noise_m=_POS_LEVELS[parts[1]],
                       srp_volume_noise_std=_SRP_LEVELS[parts[2]])
    raise ValueError(f"unknown v4 cell id {cell_id!r}")


def _load_v3(v3_run: Path, c_dim: int):
    """Load the Phase-1 V3 winner's flow + thresholds + xt_pool for gating."""
    th = np.load(v3_run / "thresholds.npz")
    thresholds = PerClusterThresholds(
        centroids=th["centroids"], p95=th["p95"], p99=th["p99"],
        n_per_cluster=th["n_per_cluster"],
    )
    # Infer flow dims from the saved state_dict + thresholds centroid dim.
    state = torch.load(v3_run / "flow.pt", map_location="cpu")
    cfg = json.loads((v3_run / "cell_config.json").read_text())["v3_cfg"]
    flow = ConditionalRealNVP(
        dim=int(thresholds.centroids.shape[1]) if thresholds.centroids.ndim == 2 else c_dim,
        c_dim=c_dim, n_layers=int(cfg["n_layers"]), hidden_dim=int(cfg["hidden_dim"]),
        n_hidden_per_net=int(cfg["n_hidden_per_net"]), scale_max=float(cfg["scale_max"]),
        dropout_p=float(cfg.get("dropout_p", 0.0)),
    )
    flow.load_state_dict(state)
    flow.eval()
    # Reconstruct the learned xt_pool (PMA-2) so gating-time inference pooling
    # matches training pooling — the calibration fix.  Absent file => the V3
    # cell used the legacy mean-pool, so xt_pool stays None.
    xt_pool = None
    xtp_path = v3_run / "xt_pool.pt"
    if str(cfg.get("xt_pool", "pma2")) == "pma2" and xtp_path.exists():
        from src.modeling.anomaly.v3_trainer import _XtPool
        xt_pool = _XtPool(c_dim, num_heads=int(cfg.get("xt_pool_num_heads", 4)))
        xt_pool.load_state_dict(torch.load(xtp_path, map_location="cpu"))
        xt_pool.eval()
    return flow, thresholds, xt_pool, cfg


class _V3Holder:
    """Minimal duck-typed stand-in for a V3Result.  ``.flow``, ``.thresholds``
    and ``.xt_pool`` are read by `_v3_event_intervals_for_recordings`."""
    def __init__(self, flow, thresholds, xt_pool=None):
        self.flow = flow
        self.thresholds = thresholds
        self.xt_pool = xt_pool


def _log(msg: str, log_path: Path) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, help="Run dir with v2/encoder.pt")
    p.add_argument("--v3-run", default=None,
                   help="Phase-1 V3 winner dir (flow.pt + thresholds.npz) for gating. "
                        "Omit to report ungated holdout MAE only.")
    p.add_argument("--cell", default=None, help=f"Single cell; omit to run all. {len(_all_cells())} cells")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--samples-cache", default=None,
                   help="Path to a pickle of precomputed V4 samples.  If it exists, "
                        "load it (skips the expensive SRP-PHAT precompute); else "
                        "precompute and write it.  The campaign driver passes one "
                        "shared path so all cells precompute exactly ONCE.")
    p.add_argument("--all-channel-modes", action="store_true",
                   help="Train the cell at all 4 channel modes (acoustic SRP / "
                        "tdoa / vibration-only-learned / fusion) for the per-modality "
                        "localization breakdown.  Default: fusion ('both') only.")
    args = p.parse_args()

    encoder_run = Path(args.encoder_run)
    if not (encoder_run / "v2" / "encoder.pt").exists():
        raise SystemExit(f"v2/encoder.pt not found under {encoder_run}")

    cells = [args.cell] if args.cell else _all_cells()
    v2_cfg = _v2_cfg(args.quick)
    v3_cfg = _v3_cfg(args.quick)
    base_v4 = _v4_cfg(args.quick)
    for cid in cells:  # fail fast
        _apply_cell(cid, base_v4)

    import pickle

    t0 = time.time()
    encoder = V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim, embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads, context_mode=v2_cfg.context_mode,
        num_context_seeds=v2_cfg.num_context_seeds,
        acoustic_cnn_width_mult=v2_cfg.acoustic_cnn_width_mult,
    )
    encoder.load_state_dict(torch.load(encoder_run / "v2" / "encoder.pt", map_location="cpu"))
    encoder.eval()
    grid = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))

    cache_path = Path(args.samples_cache) if args.samples_cache else None
    loaders = None  # built lazily — needed for V3 gating and/or precompute
    v4_samples = None
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as fh:
            v4_samples = pickle.load(fh)
        print(f"Loaded {len(v4_samples)} cached V4 samples from {cache_path} "
              f"(skipped precompute)")
    if v4_samples is None:
        print("Loading frozen V2 + D2/D3/D4/D5 loaders, precomputing V4 samples ...")
        loaders = {d: _resolved_loader(f"{d}.yaml") for d in ("d2", "d3", "d4", "d5")}
        d2_labeled = [s for s in loaders["d2"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None]
        d3_segs = loaders["d3"].list_segments()
        overrides = _d3_spatial_overrides(d3_segs)
        d3_labeled = [s for s in d3_segs if s.recording_id in overrides]
        d4_labeled = [s for s in loaders["d4"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        d5_labeled = [s for s in loaders["d5"].list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        v4_samples = precompute_v4_samples(
            encoder, d2_labeled + d3_labeled + d4_labeled + d5_labeled,
            v2_cfg=v2_cfg, grid=grid, spatial_label_overrides=overrides,
            burst_aware_srp=True, burst_seconds=0.10,
        )
        print(f"Precomputed {len(v4_samples)} V4 samples in {time.time()-t0:.0f}s")
        if cache_path is not None:
            # Atomic write so a killed cell never leaves a half-written cache
            # that a later cell would load.
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(v4_samples, fh)
            tmp.replace(cache_path)
            print(f"Cached V4 samples to {cache_path}")

    train_pos, holdout_pos = split_samples_by_position(v4_samples, V4_HOLDOUT_POSITIONS_M)
    if len(train_pos) < 4 or len(holdout_pos) < 1:
        raise SystemExit(f"insufficient spatial split: {len(train_pos)} train / {len(holdout_pos)} holdout")
    print(f"Spatial split: {len(train_pos)} train / {len(holdout_pos)} holdout samples")

    # Load Phase-1 V3 for gating (optional).
    v3_holder = None
    if args.v3_run:
        c_dim = int(v4_samples[0].context.shape[0])
        flow, thresholds, xt_pool, _ = _load_v3(Path(args.v3_run), c_dim)
        v3_holder = _V3Holder(flow, thresholds, xt_pool)
        # Gating re-runs V3 on the holdout recordings, so it needs the loaders
        # even when samples came from cache (precompute was skipped).
        if loaders is None:
            loaders = {d: _resolved_loader(f"{d}.yaml") for d in ("d2", "d3", "d4", "d5")}

    loaders_by_id = loaders or {}  # _v3_event_intervals_for_recordings keys by dataset_id

    # Precompute V3-gated keep-mask ONCE per holdout cohort (mode-independent —
    # gating depends on V3 + window times, not on the V4 channel mode).
    gate_keep = None
    if v3_holder is not None:
        try:
            from src.modeling.anomaly.weak_labels import window_overlaps_any
            intervals = _v3_event_intervals_for_recordings(
                holdout_pos, loaders_by_id, encoder, v3_holder, v2_cfg, v3_cfg)
            win_s = float(v2_cfg.window_seconds)
            gate_keep = np.array([
                bool(intervals.get(s.recording_id))
                and window_overlaps_any(s.window_start_s, s.window_start_s + win_s,
                                        intervals.get(s.recording_id, []))
                for s in holdout_pos], dtype=bool)
        except Exception as e:
            print(f"  V3 gating precompute skipped: {type(e).__name__}: {e}")

    def _train_eval(cfg, log_path) -> dict:
        res = train_v4_localization(v4_samples, cfg=cfg, grid=grid,
                                    explicit_split=(train_pos, holdout_pos))
        d: dict = {
            "channel_mode": cfg.channel_mode,
            "holdout_mae_ungated_m": float(res.val_mae_3d),
            "holdout_p95_ungated_m": float(res.val_p95_3d),
            "train_val_gap_m": float(abs(
                (res.val_loss_history[-1] if res.val_loss_history else float("nan"))
                - (res.train_loss_history[-1] if res.train_loss_history else float("nan")))),
            "early_stopped_epoch": res.early_stopped_epoch,
            "n_holdout": int(res.val_predictions.shape[0]),
        }
        if gate_keep is not None and gate_keep.shape[0] == res.val_predictions.shape[0] and gate_keep.any():
            err = np.linalg.norm(
                res.val_predictions[gate_keep] - res.val_targets[gate_keep], axis=-1)
            d["holdout_mae_v3gated_m"] = float(np.mean(err))
            d["n_holdout_gated"] = int(gate_keep.sum())
        else:
            d["holdout_mae_v3gated_m"] = None
            d["n_holdout_gated"] = int(gate_keep.sum()) if gate_keep is not None else 0
        return d, res

    modes = (["srp_only", "tdoa_only", "vibration_only_learned", "both"]
             if args.all_channel_modes else ["both"])

    for cell_id in cells:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / "runs" / f"{ts}__v4deep_{cell_id}_s{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run_log.txt"
        base_cfg = _apply_cell(cell_id, base_v4)
        (out_dir / "cell_config.json").write_text(json.dumps({
            "cell": cell_id, "seed": args.seed, "encoder_run": str(encoder_run),
            "v3_run": args.v3_run, "v4_cfg": asdict(replace(base_cfg, seed=args.seed)),
        }, indent=2, default=str))

        m: dict = {"cell": cell_id, "seed": args.seed}
        per_mode: dict = {}
        for mode in modes:
            cfg = replace(base_cfg, seed=args.seed, channel_mode=mode)
            t0 = time.time()
            try:
                d, res = _train_eval(cfg, log_path)
            except Exception as e:
                _log(f"  V4[{mode}] FAILED: {type(e).__name__}: {e}", log_path)
                per_mode[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            per_mode[mode] = d
            gated = d.get("holdout_mae_v3gated_m")
            _log(f"  V4[{mode}] {time.time()-t0:.0f}s — holdout MAE "
                 f"ungated={d['holdout_mae_ungated_m']:.4f}m "
                 f"gated={gated if gated is None else round(gated,4)}m "
                 f"gap={d['train_val_gap_m']:.4f} es={d['early_stopped_epoch']}", log_path)
            if mode == "both":
                torch.save(res.head.state_dict(), out_dir / "head.pt")
        # Hoist the fusion ('both') metrics to the top level so the campaign's
        # selection helper (which reads holdout_mae_v3gated_m) works unchanged.
        if "both" in per_mode and "error" not in per_mode["both"]:
            m.update({k: v for k, v in per_mode["both"].items() if k != "channel_mode"})
        if args.all_channel_modes:
            m["channel_modes"] = per_mode
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
        print(f"Wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
