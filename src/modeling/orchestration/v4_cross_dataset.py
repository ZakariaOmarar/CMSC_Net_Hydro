"""V4 cross-dataset transfer driver.

Train V4 on one set of datasets (e.g. D1+D2+D3+D4), evaluate on a disjoint
set (e.g. D5 only).  Confirms that the V4 head generalizes across recording
sessions / rig states — not just across positions within the same session.

For each direction (`train_ids → test_ids`), the driver:
  1. Pulls the precomputed V4Sample list shared with the campaign.
  2. Splits via `split_samples_by_dataset(samples, holdout=test_ids)`.
  3. Trains V4 from scratch on the train split; evaluates on test split.
  4. Optionally repeats per channel mode (`--all-channel-modes`).

Output:  `<out_dir>/summary.json` (per-direction × per-modality MAE).

Usage:

    python -m src.modeling.orchestration.v4_cross_dataset \\
        --encoder-run <dir> [--samples-cache <path>] [--all-channel-modes] \\
        [--out-dir <dir>] [--seed 42]

Phase 5 of `scripts/run_deep_v3v4_campaign.py` invokes this on the V4 winner
only.  Reuses the same V2 encoder + V4 sample cache as Phase 2 / Phase 4
(no re-precompute).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..context.v2_fusion import V2FusionEncoder
from ..eval import percentile_bootstrap_ci
from ..localization import (
    GridSpec,
    precompute_v4_samples,
    split_samples_by_dataset,
    train_v4_localization,
)
from .full_run import (
    REPO_ROOT,
    _d3_spatial_overrides,
    _resolved_loader,
    _v2_cfg,
    _v4_cfg,
)


_CHANNEL_MODES: tuple[Literal["both", "srp_only", "tdoa_only", "vibration_only_learned"], ...] = (
    "both", "srp_only", "tdoa_only", "vibration_only_learned",
)


# Direction = (label, train_dataset_ids, test_dataset_ids).
# Primary: train on the four older sessions, test on the newer D5 session.
# Secondary: reverse for symmetry sanity (much smaller train cohort).
_DEFAULT_DIRECTIONS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("d1to4_to_d5", ("d2", "d3", "d4"), ("d5",)),
    ("d5_to_d4",    ("d5",),             ("d4",)),
)


def run_cross_dataset(
    *,
    encoder_run: Path,
    samples_cache: Path | None = None,
    all_channel_modes: bool = False,
    directions: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = _DEFAULT_DIRECTIONS,
    out_dir: Path | None = None,
    seed: int = 42,
    quick: bool = False,
    burst_aware_srp: bool = True,
) -> dict:
    out_dir = out_dir or (REPO_ROOT / "results" / "cross_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_cfg = _v2_cfg(quick)
    v4_cfg = _v4_cfg(quick)

    print(f"V4 cross-dataset: loading V2 encoder from {encoder_run}/v2/encoder.pt")
    encoder = V2FusionEncoder(
        feature_dim=v2_cfg.feature_dim, embed_dim=v2_cfg.embed_dim,
        n_heads=v2_cfg.n_heads, context_mode=v2_cfg.context_mode,
        num_context_seeds=v2_cfg.num_context_seeds,
        acoustic_cnn_width_mult=v2_cfg.acoustic_cnn_width_mult,
    )
    encoder.load_state_dict(torch.load(encoder_run / "v2" / "encoder.pt", map_location="cpu"))
    encoder.eval()

    samples: list | None = None
    if samples_cache is not None and Path(samples_cache).exists():
        import pickle
        with Path(samples_cache).open("rb") as fh:
            samples = pickle.load(fh)
        print(f"V4 cross-dataset: loaded {len(samples)} cached V4 samples "
              f"from {samples_cache}")

    if samples is None:
        print("V4 cross-dataset: gathering labeled segments + precomputing V4 samples ...")
        D2 = _resolved_loader("d2.yaml")
        D3 = _resolved_loader("d3.yaml")
        D4 = _resolved_loader("d4.yaml")
        D5 = _resolved_loader("d5.yaml")
        d2_labeled = [
            s for s in D2.list_segments()
            if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
        ]
        d3_segs = D3.list_segments()
        overrides = _d3_spatial_overrides(d3_segs)
        d3_labeled = [s for s in d3_segs if s.recording_id in overrides]
        d4_labeled = [s for s in D4.list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        d5_labeled = [s for s in D5.list_segments()
                      if s.is_anomaly and s.spatial_label is not None]
        all_labeled = d2_labeled + d3_labeled + d4_labeled + d5_labeled
        grid = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))
        t0 = time.time()
        samples = precompute_v4_samples(
            encoder, all_labeled,
            v2_cfg=v2_cfg, grid=grid,
            spatial_label_overrides=overrides,
            burst_aware_srp=burst_aware_srp, burst_seconds=0.10,
            restrict_to_knock_intervals=True,
        )
        print(f"  precomputed {len(samples)} V4 samples in {time.time() - t0:.1f}s")
        if samples_cache is not None:
            import pickle
            with Path(samples_cache).open("wb") as fh:
                pickle.dump(samples, fh)

    grid = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))
    modes = list(_CHANNEL_MODES) if all_channel_modes else ["both"]

    results: dict[str, dict] = {}
    for label, train_ids, test_ids in directions:
        # Drop samples whose dataset_id is in neither set (e.g. d2 when
        # train=d5, test=d4).  The split helper would otherwise put them
        # in the train half; here we want disjoint partitions only.
        keep_ids = set(train_ids) | set(test_ids)
        sub_samples = [s for s in samples if s.dataset_id in keep_ids]
        tr, te = split_samples_by_dataset(sub_samples, set(test_ids))
        if not tr or not te:
            print(f"  [{label}] SKIPPED (train={len(tr)}, test={len(te)})")
            results[label] = {"error": "empty train or test split",
                              "n_train": len(tr), "n_test": len(te)}
            continue
        n_train_positions = len({tuple(s.target_xyz) for s in tr})
        n_test_positions = len({tuple(s.target_xyz) for s in te})
        per_mode: dict[str, dict] = {}
        for mode in modes:
            cfg = replace(v4_cfg, seed=seed, channel_mode=mode)
            t0 = time.time()
            try:
                res = train_v4_localization(
                    sub_samples, cfg=cfg, grid=grid, explicit_split=(tr, te)
                )
            except Exception as e:
                per_mode[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            errs = np.linalg.norm(res.val_predictions - res.val_targets, axis=-1)
            ci_low, ci_high = float("nan"), float("nan")
            if errs.size >= 2:
                ci = percentile_bootstrap_ci(errs, n_boot=1000, seed=seed)
                ci_low, ci_high = ci.ci_low, ci.ci_high
            per_mode[mode] = {
                "val_mae_3d_m": float(res.val_mae_3d),
                "val_p95_3d_m": float(res.val_p95_3d),
                "train_mae_3d_m": float(res.train_mae_3d),
                "ci95_low_m": ci_low,
                "ci95_high_m": ci_high,
                "elapsed_seconds": float(time.time() - t0),
            }
            print(f"  [{label}/{mode}] MAE={res.val_mae_3d:.3f}m "
                  f"(train MAE={res.train_mae_3d:.3f}m) "
                  f"n_train_pos={n_train_positions} n_test_pos={n_test_positions} "
                  f"in {time.time()-t0:.0f}s")
        results[label] = {
            "train_dataset_ids": list(train_ids),
            "test_dataset_ids": list(test_ids),
            "n_train_windows": len(tr),
            "n_test_windows": len(te),
            "n_train_positions": n_train_positions,
            "n_test_positions": n_test_positions,
            "per_channel_mode": per_mode,
        }

    summary = {
        "encoder_run": str(encoder_run),
        "seed": int(seed),
        "channel_modes": modes,
        "directions": results,
        "v4_cfg": asdict(replace(v4_cfg, seed=seed)),
        "method": "cross_dataset_transfer",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"V4 cross-dataset: summary written to {out_dir / 'summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, type=Path,
                   help="Run dir containing v2/encoder.pt")
    p.add_argument("--samples-cache", default=None, type=Path,
                   help="Shared V4Sample pickle (re-used; precomputed once)")
    p.add_argument("--all-channel-modes", action="store_true",
                   help="Run all 4 channel modes per direction (4× wall time)")
    p.add_argument("--out-dir", default=None, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    run_cross_dataset(
        encoder_run=args.encoder_run,
        samples_cache=args.samples_cache,
        all_channel_modes=args.all_channel_modes,
        out_dir=args.out_dir, seed=args.seed, quick=args.quick,
    )


if __name__ == "__main__":
    main()
