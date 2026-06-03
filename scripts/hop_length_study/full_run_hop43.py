"""End-to-end full_run.py at acoustic ``hop_length=43`` (n_fft=1024).

Purpose
-------
The default full_run.py uses hop=512.  This wrapper re-executes the SAME
orchestration with hop=43 (the "match D4 vibration rate" condition from the
hop-length ablation), so every downstream metric (V0, V1, V2, V3, V3 cohort
validation, V4 localization, V5.1, modality probe, etc.) can be compared
head-to-head against the existing ``results/full_run/metrics.json`` baseline.

Controlled variables
--------------------
* All V1/V2/V3 epoch counts, batch sizes, LRs, augmentation strengths kept
  at their canonical values via ``v1_config(quick=False)`` / ``v2_config`` / etc.
* SpecAugment time-mask width is rescaled from 16 frames at hop=512 to
  ~191 frames at hop=43 so the masked physical duration stays at ~512 ms.
  Without this, hop=43 would mask only ~43 ms — a confound that changes
  augmentation strength.
* Same seed (42), same data, same dataset list (D1+D2+D3+D4).

Outputs
-------
* The baseline at ``results/full_run/`` is copied to
  ``results/full_run_hop512_baseline/`` first (only if that dir doesn't
  already exist; idempotent).
* full_run.main() writes its usual artefacts to ``results/full_run/``.
* On completion, ``results/full_run/`` is renamed to
  ``results/full_run_hop43/`` so subsequent comparisons see both dirs side
  by side.

Wall-clock estimate: ~7-12 hours (baseline ~6h at hop=512; hop=43 V1+V2
~1.8x slower per the pilot ablation, V3+V4+V5 ~unchanged).

Usage::

    python scripts/hop_length_study/full_run_hop43.py            # full pipeline
    python scripts/hop_length_study/full_run_hop43.py --quick    # smoke test of the wrapper

Memory note
-----------
At hop=43 the in-memory acoustic feature cache is ~12x larger than at
hop=512.  The full pipeline trains on D1+D2+D3+D4 with V1 precomputation;
peak RAM may exceed 30 GB.  If the process OOMs, the cleanest fallback is
``python scripts/hop_length_study/full_run_hop43.py --only-datasets d4`` (matches the
``--only-datasets`` semantics of ``full_run.py``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.orchestration import full_run as fr

HOP_NEW = 43
N_FFT_NEW = 1024
ANCHOR_HOP = 512  # the hop the canonical time_mask=16 was tuned for


_orig_v1_cfg = fr.v1_config
_orig_v2_cfg = fr.v2_config


def _patched_v1_cfg(quick: bool):
    cfg = _orig_v1_cfg(quick)
    new_time_mask = max(1, int(round(cfg.spec_augment_time_mask * (ANCHOR_HOP / HOP_NEW))))
    return replace(
        cfg,
        hop_length=HOP_NEW,
        n_fft=N_FFT_NEW,
        spec_augment_time_mask=new_time_mask,
    )


def _patched_v2_cfg(quick: bool):
    cfg = _orig_v2_cfg(quick)
    new_time_mask = max(1, int(round(cfg.spec_augment_time_mask * (ANCHOR_HOP / HOP_NEW))))
    return replace(
        cfg,
        hop_length=HOP_NEW,
        n_fft=N_FFT_NEW,
        spec_augment_time_mask=new_time_mask,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Halve epoch counts for a smoke-level real-data run")
    ap.add_argument("--only-datasets", type=str, nargs="+", default=None,
                    choices=["d1", "d2", "d3", "d4"],
                    help="Restrict V1/V2/V3 SSL training to these dataset IDs (memory escape hatch).")
    args = ap.parse_args()

    # ---- 1. Back up the existing baseline (idempotent) ----
    run_dir = REPO / "results" / "full_run"
    baseline_dir = REPO / "results" / "full_run_hop512_baseline"
    hop43_dir = REPO / "results" / "full_run_hop43"

    if run_dir.exists() and not baseline_dir.exists():
        print(f"[hop43-wrapper] Backing up baseline {run_dir} -> {baseline_dir}", flush=True)
        shutil.copytree(run_dir, baseline_dir)
    elif baseline_dir.exists():
        print(f"[hop43-wrapper] Baseline backup already exists at {baseline_dir} — leaving it untouched", flush=True)
    else:
        print(f"[hop43-wrapper] No existing {run_dir} to back up — will write a fresh hop=43 run", flush=True)

    # If a previous hop43 run was already renamed in place, abort to avoid clobber.
    if hop43_dir.exists():
        raise SystemExit(
            f"[hop43-wrapper] {hop43_dir} already exists; delete or rename it before re-running."
        )

    # Wipe results/full_run/ so this run starts cleanly without mixing artefacts.
    if run_dir.exists():
        print(f"[hop43-wrapper] Removing {run_dir} (already backed up) so new run writes cleanly", flush=True)
        shutil.rmtree(run_dir)

    # ---- 2. Monkey-patch the config builders ----
    print(f"[hop43-wrapper] Patching v1_config / v2_config: hop_length={HOP_NEW}, n_fft={N_FFT_NEW}", flush=True)
    fr.v1_config = _patched_v1_cfg
    fr.v2_config = _patched_v2_cfg
    sample_v1 = _patched_v1_cfg(args.quick)
    print(
        f"[hop43-wrapper] V1 cfg: hop={sample_v1.hop_length}, n_fft={sample_v1.n_fft}, "
        f"spec_augment_time_mask={sample_v1.spec_augment_time_mask}f "
        f"(~{1000.0 * sample_v1.spec_augment_time_mask * sample_v1.hop_length / 16000:.0f} ms)",
        flush=True,
    )

    # ---- 3. Run the full pipeline ----
    print(f"[hop43-wrapper] Launching full_run.main(quick={args.quick}, only_datasets={args.only_datasets}) ...", flush=True)
    fr.main(
        quick=args.quick,
        dataset_ids=tuple(args.only_datasets) if args.only_datasets else None,
    )

    # ---- 4. Rename the output dir so it doesn't clobber the baseline on re-runs ----
    if run_dir.exists():
        print(f"[hop43-wrapper] Renaming {run_dir} -> {hop43_dir}", flush=True)
        run_dir.rename(hop43_dir)
        print(f"[hop43-wrapper] DONE. Compare against {baseline_dir}.", flush=True)
    else:
        print(f"[hop43-wrapper] WARNING: {run_dir} missing after full_run; nothing to rename", flush=True)


if __name__ == "__main__":
    main()
