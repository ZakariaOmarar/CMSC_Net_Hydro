"""Run the full V1→V5 pipeline for multiple seeds and aggregate the
headline numbers (mean ± std) for the publication tables.

Usage:
    python -m src.modeling.orchestration.multi_seed \
        --seeds 42 1337 2718 \
        --quick   # optional, halves epoch counts

Each seed runs to completion and is archived under `results/runs/`.
After every seed finishes the script prints the running mean ± std for
the headline metrics so progress is visible during the (possibly hours-
long) sweep.

Final output: `results/runs/multi_seed_summary.json` with mean / std
per metric across all seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from .archive import ARCHIVE_ROOT
from .full_run import (
    main as full_run_main,
)

HEADLINE_KEYS = (
    "v1_acoustic_purity",
    "v1_vibration_purity",
    "v2_purity",
    "v2_nmi",
    "v2_a1_purity",
    "v3_val_nll",
    "v3_a2_val_nll",
    "v4_mae_3d",
    "v4_p95_3d",
    "v4_a3_mae_3d",
    "v5_1_mae_3d",
)


def _override_seed(seed: int) -> dict:
    """Hot-patch the orchestrator's `_v*_cfg` helpers to use this seed."""
    import src.modeling.orchestration.full_run as fr

    original_v1 = fr._v1_cfg
    original_v2 = fr._v2_cfg
    original_v3 = fr._v3_cfg

    def v1(quick: bool):
        cfg = original_v1(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    def v2(quick: bool):
        cfg = original_v2(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    def v3(quick: bool):
        cfg = original_v3(quick)
        return cfg.__class__(**{**asdict(cfg), "seed": seed})

    fr._v1_cfg = v1
    fr._v2_cfg = v2
    fr._v3_cfg = v3
    return {"v1": original_v1, "v2": original_v2, "v3": original_v3}


def _restore(originals: dict) -> None:
    import src.modeling.orchestration.full_run as fr

    fr._v1_cfg = originals["v1"]
    fr._v2_cfg = originals["v2"]
    fr._v3_cfg = originals["v3"]


def main(seeds: list[int], quick: bool = False) -> dict:
    print(f"Multi-seed sweep: seeds={seeds}, quick={quick}")
    per_seed_headlines: list[dict] = []
    per_seed_metrics: list[dict] = []
    sweep_start = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n=== Seed {seed} ({i + 1}/{len(seeds)}) ===")
        t0 = time.time()
        originals = _override_seed(seed)
        try:
            metrics = full_run_main(quick=quick)
        finally:
            _restore(originals)
        elapsed_min = (time.time() - t0) / 60.0
        print(f"Seed {seed} finished in {elapsed_min:.1f} min")

        per_seed_metrics.append(metrics)
        # Read the just-archived index entry for the headline.
        from .archive import list_runs

        runs = list_runs()
        if runs:
            per_seed_headlines.append({"seed": seed, **runs[0]["headline"]})

        # Running aggregate.
        if len(per_seed_headlines) > 1:
            print("Running aggregate (mean ± std):")
            for k in HEADLINE_KEYS:
                vals = [h[k] for h in per_seed_headlines if h.get(k) is not None]
                if not vals:
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                print(f"  {k:<24} = {m:.4f} ± {s:.4f}  (n={len(vals)})")

    total_elapsed = (time.time() - sweep_start) / 60.0
    print(f"\nMulti-seed sweep done in {total_elapsed:.1f} min total.")

    # Final aggregate.
    summary: dict = {
        "seeds": seeds,
        "quick": quick,
        "total_elapsed_minutes": total_elapsed,
        "per_seed_headlines": per_seed_headlines,
        "aggregate": {},
    }
    for k in HEADLINE_KEYS:
        vals = [h[k] for h in per_seed_headlines if h.get(k) is not None]
        if not vals:
            continue
        summary["aggregate"][k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "values": [float(v) for v in vals],
        }

    out_path = ARCHIVE_ROOT / "multi_seed_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Aggregate written to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 1337, 2718],
        help="Seeds to run sequentially.  Default: 42 1337 2718",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use the --quick (halved-epoch) profile in each run.",
    )
    args = parser.parse_args()
    main(seeds=args.seeds, quick=args.quick)
