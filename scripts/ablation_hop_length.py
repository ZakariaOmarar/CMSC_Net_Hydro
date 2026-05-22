"""Ablation: effect of acoustic ``hop_length`` (and matching ``n_fft``) on
V3-fusion anomaly detection.

PROTOCOL (pre-registered; do not edit after results are in)
-----------------------------------------------------------
Hypothesis:
  H0: V3-fusion anomaly-detection AUROC is invariant to the acoustic
      hop_length within the tested range, holding all other pipeline
      hyperparameters constant.
  H1: AUROC differs by more than the noise floor estimated from
      seed-to-seed variation.

Conditions (3):
  C1 default       hop=512, n_fft=1024  (current baseline)
  C2 naive_match   hop=43,  n_fft=1024  (literal "match D4 376 Hz vibration"
                                         — heavily redundant STFT overlap)
  C3 coherent_fast hop=64,  n_fft=256   (genuinely finer time-frequency;
                                         16 ms analysis window, 4 ms hop)

Controlled variables (held constant across all runs):
  * window_seconds = 2.0, window_stride_seconds = 1.0
  * V1/V2/V3 epochs, batch size, LR, loss config
  * Same recording-level train/val/test split (seeded)
  * SpecAugment time-mask width auto-rescaled to keep the *physical
    duration* of masked time constant (otherwise hop changes the
    augmentation strength implicitly — a confound).

Dependent variables (recorded per run):
  * Primary: synthetic-anomaly ROC-AUC at SNR=0 dB on the V3-fusion flow
  * Secondary: full AUC vs SNR curve (-10, -5, 0, +5, +10 dB) with 95% CI
  * Control: V3 val_NLL final, V1 acoustic cluster purity (sanity gate)
  * Cost: wall-clock seconds per run

Pre-registered decision rule (applied by analyze_hop_ablation.py):
  Reject H0 only if BOTH
    (a) mean-condition-difference in AUC@0dB > 2 * pooled within-condition
        std across seeds, AND
    (b) Wilcoxon signed-rank p < 0.05 across seed-paired differences.
  Else accept H0 and report effect size (Cohen's d) with 95% bootstrap CI.

Usage
-----
Pilot (single seed, fast):
  python scripts/ablation_hop_length.py --seeds 0 --quick --datasets d4 \\
      --conditions default,naive_match,coherent_fast

Full (3 seeds):
  python scripts/ablation_hop_length.py --seeds 0,1,2 --datasets d4 \\
      --conditions default,naive_match,coherent_fast

Output: results/ablation_hop/<timestamp>/{metrics.csv, run_log.txt, per_run/*.json}
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly.synthetic_eval import evaluate_synthetic_anomaly_auc
from src.modeling.anomaly.v3_trainer import _extract_xc, train_v3_cnf
from src.modeling.context.v1_ssl import train_v1_per_modality
from src.modeling.context.v2_ssl import (
    _PairedGroupedBatchSampler,
    _PairedWindowedDataset,
    _collate,
    _gather_paired_segments,
    _split_segments_by_recording,
    train_v2_fusion,
)
from src.modeling.orchestration.full_run import (
    _resolved_loader,
    _v1_cfg,
    _v2_cfg,
    _v3_cfg,
)

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# Pre-registered conditions
# ----------------------------------------------------------------------------

# Each condition is (hop_length, n_fft). Time-mask is auto-rescaled to keep
# physical duration constant (anchored to the C1 default).
CONDITIONS: dict[str, dict[str, int]] = {
    "default":       {"hop_length": 512, "n_fft": 1024},
    "naive_match":   {"hop_length": 43,  "n_fft": 1024},
    "coherent_fast": {"hop_length": 64,  "n_fft": 256},
}

# Anchor for SpecAugment time-mask rescaling — at hop=512 the default is
# spec_augment_time_mask=16 frames = 16 * (512/16000) s = 512 ms. We keep
# that 512 ms duration constant across conditions by scaling the integer
# frame count proportionally to (anchor_hop / new_hop).
SPEC_AUG_TIME_MASK_ANCHOR_HOP = 512


def _override_v1_cfg(base, hop_length: int, n_fft: int, seed: int):
    """Apply per-condition + per-seed overrides to the canonical V1 config."""
    base_time_mask = base.spec_augment_time_mask
    scaled_time_mask = max(1, int(round(base_time_mask * (SPEC_AUG_TIME_MASK_ANCHOR_HOP / hop_length))))
    return replace(
        base,
        hop_length=hop_length,
        n_fft=n_fft,
        spec_augment_time_mask=scaled_time_mask,
        seed=seed,
    )


def _override_v2_cfg(base, hop_length: int, n_fft: int, seed: int):
    base_time_mask = base.spec_augment_time_mask
    scaled_time_mask = max(1, int(round(base_time_mask * (SPEC_AUG_TIME_MASK_ANCHOR_HOP / hop_length))))
    return replace(
        base,
        hop_length=hop_length,
        n_fft=n_fft,
        spec_augment_time_mask=scaled_time_mask,
        seed=seed,
    )


def _override_v3_cfg(base, seed: int):
    return replace(base, seed=seed)


# ----------------------------------------------------------------------------
# Single-run worker
# ----------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _run_one(
    *,
    condition: str,
    seed: int,
    dataset_yamls: list[str],
    quick: bool,
    out_per_run: Path,
    log,
) -> dict:
    """Execute one (condition, seed) cell of the ablation; return metrics dict."""
    t_start = time.time()
    _seed_everything(seed)
    cond_spec = CONDITIONS[condition]
    hop, nfft = cond_spec["hop_length"], cond_spec["n_fft"]

    base_v1 = _v1_cfg(quick)
    base_v2 = _v2_cfg(quick)
    base_v3 = _v3_cfg(quick)
    v1_cfg = _override_v1_cfg(base_v1, hop, nfft, seed)
    v2_cfg = _override_v2_cfg(base_v2, hop, nfft, seed)
    v3_cfg = _override_v3_cfg(base_v3, seed)

    log(
        f"[{condition} seed={seed}] V1/V2 hop={hop} n_fft={nfft} "
        f"time_mask={v1_cfg.spec_augment_time_mask}f (~{1000.0 * v1_cfg.spec_augment_time_mask * hop / 16000:.0f}ms) "
        f"epochs V1={v1_cfg.epochs} V2={v2_cfg.epochs} V3={v3_cfg.epochs}"
    )

    loaders = [_resolved_loader(y) for y in dataset_yamls]
    log(f"[{condition} seed={seed}] datasets: {[L.spec.id for L in loaders]}")

    # ---------- V1 acoustic ----------
    t0 = time.time()
    log(f"[{condition} seed={seed}] training V1 acoustic ...")
    v1_a = train_v1_per_modality(loaders, modality="acoustic", cfg=v1_cfg)
    t_v1a = time.time() - t0
    purity_a = float(v1_a.sanity_gate.get("purity", float("nan")))
    nmi_a = float(v1_a.sanity_gate.get("nmi", float("nan")))
    log(f"[{condition} seed={seed}] V1 acoustic done in {t_v1a:.0f}s (purity={purity_a:.3f}, NMI={nmi_a:.3f})")

    # ---------- V1 vibration ----------
    t0 = time.time()
    log(f"[{condition} seed={seed}] training V1 vibration ...")
    v1_v = train_v1_per_modality(loaders, modality="vibration", cfg=v1_cfg)
    t_v1v = time.time() - t0
    log(f"[{condition} seed={seed}] V1 vibration done in {t_v1v:.0f}s")

    # ---------- V2 fusion ----------
    t0 = time.time()
    log(f"[{condition} seed={seed}] training V2 fusion ...")
    v2 = train_v2_fusion(
        loaders,
        cfg=v2_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    t_v2 = time.time() - t0
    log(f"[{condition} seed={seed}] V2 done in {t_v2:.0f}s")

    # ---------- V3 fusion ----------
    t0 = time.time()
    log(f"[{condition} seed={seed}] training V3 fusion ...")
    v3 = train_v3_cnf(v2.encoder, loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
    t_v3 = time.time() - t0
    val_nll_final = float(v3.val_nll[-1])
    log(f"[{condition} seed={seed}] V3 done in {t_v3:.0f}s (val_NLL={val_nll_final:.3f})")

    # ---------- Synthetic-anomaly ROC-AUC ----------
    t0 = time.time()
    log(f"[{condition} seed={seed}] computing synthetic-anomaly ROC-AUC ...")
    auc_at = {snr: float("nan") for snr in (-10.0, -5.0, 0.0, 5.0, 10.0)}
    ci_low = {snr: float("nan") for snr in auc_at}
    ci_high = {snr: float("nan") for snr in auc_at}
    n_clean = 0
    try:
        segs_all = _gather_paired_segments(loaders, v2_cfg)
        _train_segs, _val_segs_full = _split_segments_by_recording(
            segs_all, v3_cfg.val_ratio, v3_cfg.seed
        )
        _val_fit_segs, val_segs_for_auc = _split_segments_by_recording(
            _val_segs_full, v3_cfg.threshold_fit_val_ratio, v3_cfg.seed + 1
        )
        val_ds = _PairedWindowedDataset(val_segs_for_auc, v2_cfg)
        if len(val_ds) > 0:
            import torch.utils.data as _tud
            val_loader = _tud.DataLoader(
                val_ds,
                batch_sampler=_PairedGroupedBatchSampler(
                    val_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed
                ),
                collate_fn=_collate,
            )
            x_val, c_val, _ = _extract_xc(v2.encoder, val_loader, torch.device("cpu"))
            auc_result = evaluate_synthetic_anomaly_auc(
                v3.flow, x_val.numpy(), c_val.numpy(),
                snr_db_list=tuple(auc_at.keys()),
                n_boot=500,
                seed=v3_cfg.seed,
            )
            auc_at = dict(auc_result.snr_db_to_auc)
            ci_low = dict(auc_result.snr_db_to_auc_ci_low)
            ci_high = dict(auc_result.snr_db_to_auc_ci_high)
            n_clean = int(auc_result.snr_db_to_n_clean[0.0])
    except Exception as e:
        log(f"[{condition} seed={seed}] synthetic AUC FAILED: {type(e).__name__}: {e}")
    t_auc = time.time() - t0
    log(
        f"[{condition} seed={seed}] AUC computed in {t_auc:.0f}s — "
        f"AUC@0dB={auc_at.get(0.0, float('nan')):.3f}  (n_clean={n_clean})"
    )

    total_seconds = time.time() - t_start
    row = {
        "condition": condition,
        "seed": seed,
        "hop_length": hop,
        "n_fft": nfft,
        "n_frames_2s_window": int(round(2.0 * (16000.0 / hop))),
        "datasets": ",".join(dataset_yamls),
        "quick": quick,
        "v1_purity_acoustic": purity_a,
        "v1_nmi_acoustic": nmi_a,
        "v3_val_nll_final": val_nll_final,
        "n_val_eval_windows": int(v3.val_scores.shape[0]),
        "n_synth_clean": n_clean,
        "auc_neg10db": auc_at.get(-10.0, float("nan")),
        "auc_neg5db":  auc_at.get(-5.0, float("nan")),
        "auc_0db":     auc_at.get(0.0, float("nan")),  # primary headline
        "auc_pos5db":  auc_at.get(5.0, float("nan")),
        "auc_pos10db": auc_at.get(10.0, float("nan")),
        "auc_0db_ci_low":  ci_low.get(0.0, float("nan")),
        "auc_0db_ci_high": ci_high.get(0.0, float("nan")),
        "wall_v1a_s": round(t_v1a, 1),
        "wall_v1v_s": round(t_v1v, 1),
        "wall_v2_s":  round(t_v2, 1),
        "wall_v3_s":  round(t_v3, 1),
        "wall_auc_s": round(t_auc, 1),
        "wall_total_s": round(total_seconds, 1),
    }
    # Per-run dump for full reproducibility (configs included)
    per_run_path = out_per_run / f"{condition}_seed{seed}.json"
    per_run_path.write_text(json.dumps({
        "row": row,
        "v1_cfg": {k: v for k, v in asdict(v1_cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "v2_cfg": {k: v for k, v in asdict(v2_cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "v3_cfg": {k: v for k, v in asdict(v3_cfg).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "auc_full": {f"snr_{int(s)}db": {"auc": auc_at[s], "ci_low": ci_low[s], "ci_high": ci_high[s]} for s in auc_at},
    }, indent=2))
    log(f"[{condition} seed={seed}] DONE in {total_seconds/60:.1f} min — wrote {per_run_path.name}")
    return row


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

CSV_FIELDS = [
    "condition", "seed", "hop_length", "n_fft", "n_frames_2s_window",
    "datasets", "quick",
    "v1_purity_acoustic", "v1_nmi_acoustic",
    "v3_val_nll_final", "n_val_eval_windows", "n_synth_clean",
    "auc_neg10db", "auc_neg5db", "auc_0db", "auc_pos5db", "auc_pos10db",
    "auc_0db_ci_low", "auc_0db_ci_high",
    "wall_v1a_s", "wall_v1v_s", "wall_v2_s", "wall_v3_s", "wall_auc_s", "wall_total_s",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="default,naive_match,coherent_fast",
                    help=f"Comma-separated condition names. Available: {sorted(CONDITIONS)}")
    ap.add_argument("--seeds", default="0",
                    help="Comma-separated integer seeds, e.g. 0,1,2")
    ap.add_argument("--datasets", default="d4",
                    help="Comma-separated dataset ids (yaml names without .yaml), e.g. d4 or d1,d2,d3,d4")
    ap.add_argument("--quick", action="store_true",
                    help="Use 3-epoch V1/V2 + 8-epoch V3 (pilot mode); else full 12/12/15")
    ap.add_argument("--out-dir", default=None,
                    help="Override output dir; default results/ablation_hop/<timestamp>")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dataset_yamls = [f"{d.strip()}.yaml" for d in args.datasets.split(",") if d.strip()]

    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}; available: {sorted(CONDITIONS)}")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.out_dir else (REPO / "results" / "ablation_hop" / timestamp)
    out_root.mkdir(parents=True, exist_ok=True)
    out_per_run = out_root / "per_run"
    out_per_run.mkdir(exist_ok=True)
    csv_path = out_root / "metrics.csv"
    log_path = out_root / "run_log.txt"

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"Ablation: hop_length on V3-fusion AUROC")
    log(f"out_dir   = {out_root}")
    log(f"conditions= {conditions}")
    log(f"seeds     = {seeds}")
    log(f"datasets  = {dataset_yamls}")
    log(f"quick     = {args.quick}")

    # Initialise CSV with header (if file doesn't exist) — append rows per run
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            fh.flush()

    n_runs = len(conditions) * len(seeds)
    log(f"Total runs: {n_runs}")
    run_idx = 0
    for seed in seeds:
        for cond in conditions:
            run_idx += 1
            log(f"\n=== Run {run_idx}/{n_runs}: condition={cond}, seed={seed} ===")
            try:
                row = _run_one(
                    condition=cond,
                    seed=seed,
                    dataset_yamls=dataset_yamls,
                    quick=args.quick,
                    out_per_run=out_per_run,
                    log=log,
                )
                with csv_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                    writer.writerow(row)
                    fh.flush()
            except Exception as e:
                log(f"!!! Run FAILED: {type(e).__name__}: {e}")
                log(traceback.format_exc())

    log(f"\nAll runs complete. CSV: {csv_path}")
    log("Next: python scripts/analyze_hop_ablation.py --csv " + str(csv_path))


if __name__ == "__main__":
    main()
