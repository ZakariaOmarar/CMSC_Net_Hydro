"""R3.4 — train FOUR V4 localisation pipelines and emit comparable per-paradigm
predictions for the RQ3 paradigm-comparison eval.

Paradigms (per the approved plan):

  * **V0 SRP-PHAT** (acoustic classical, non-learned) — existing
    `evaluate_srp_phat` reused unchanged.
  * **V0 accel multilateration** (vibration classical, non-learned) —
    R3.2 standalone solver from
    :mod:`src.modeling.localization.multilateration`.
  * **V4-acoustic** = ``channel_mode="srp_only"`` (TDOA tokens zeroed,
    learned head + acoustic-SRP soft-argmax init + V2 c_t FiLM).
  * **V4-vibration** = ``channel_mode="vibration_only_learned"`` (SRP
    volume zeroed, learned head with multilateration init + V2 c_t FiLM).
  * **V4-fusion** = ``channel_mode="both"`` (full architecture; the
    chained-system "Intermediate Fusion" paradigm).

All four learned trainings share the same precomputed V4 samples
(`precompute_v4_samples` is called ONCE — multilateration is included in
the sample object per R3.3, so paradigm switching is just a config swap).
This keeps per-window predictions directly comparable across paradigms.

Writes under ``results/runs/<timestamp>__v4_three_paradigms/``:

  v4_<paradigm>/head.pt            — saved trained head
  v4_<paradigm>/val_predictions.npz — (n_val, 3) predictions + targets
                                       + recording_ids
  classical/v0_srp_phat.json       — per-dataset SRP-PHAT MAE
  classical/v0_multilat.json       — per-recording multilateration positions
  metrics.json                     — headline numbers side-by-side

Run::

    python -m scripts.run_v4_three_paradigms \\
        --source-run results/runs/<v1+v2-run>
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

from src.modeling.anomaly_baselines.srp_phat_baseline import (
    SRPConfig,
    evaluate_srp_phat,
    summarise,
)
from src.modeling.context.v2_fusion import V2FusionEncoder
from src.modeling.context.v2_ssl import V2SSLConfig
from src.modeling.localization.multilateration import accel_tdoa_multilateration_v0
from src.modeling.localization.v4_features import GridSpec
from src.modeling.localization.v4_trainer import (
    V4Config,
    V4Sample,
    precompute_v4_samples,
    train_v4_localization,
)
from src.modeling.orchestration.full_run import (
    _d3_spatial_overrides,
    _resolved_loader,
    _v2_cfg,
    _v4_cfg,
)


REPO = Path(__file__).resolve().parents[1]


def _build_v2(cfg: V2SSLConfig) -> V2FusionEncoder:
    return V2FusionEncoder(
        feature_dim=cfg.feature_dim, embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
        context_mode=cfg.context_mode, num_context_seeds=cfg.num_context_seeds,
        acoustic_cnn_width_mult=cfg.acoustic_cnn_width_mult,
    )


def _save_predictions(
    pipeline_dir: Path,
    result,
    samples: list[V4Sample],
) -> None:
    """Save per-window predictions + recording metadata for later late-fusion."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.head.state_dict(), pipeline_dir / "head.pt")
    # `result.val_predictions` / `val_targets` are already per-window arrays.
    # We additionally need a stable key per window so late-fusion can pair
    # them across paradigms — recording_id is stable across paradigm runs
    # when the same sample list and split seed are used.
    val_keys: list[str] = []
    # Mirror the val cohort the trainer would have produced from samples.
    # `result.val_recording_ids` is sorted; we have to recover the per-
    # window order, which lives in `result.val_targets` order.  Since
    # `_split_samples_by_recording` keeps insertion order within val_keys,
    # we just iterate val samples in the order they appear in `samples`
    # AFTER filtering to val_recording_ids.
    val_set = set(result.val_recording_ids)
    for s in samples:
        key = f"{Path(s.source_dir).name}/{s.recording_id}"
        if key in val_set:
            val_keys.append(key)
    np.savez(
        pipeline_dir / "val_predictions.npz",
        predictions=result.val_predictions,
        targets=result.val_targets,
        init_xyz=result.val_init_xyz,
        residuals=result.val_residuals,
        recording_keys=np.asarray(val_keys, dtype="U64"),
    )


def _train_one_v4(
    name: str,
    channel_mode: str,
    samples: list[V4Sample],
    grid: GridSpec,
    base_cfg: V4Config,
    out_dir: Path,
    log,
) -> dict:
    log(f"\n=== V4-{name} (channel_mode={channel_mode}) ===")
    cfg = replace(base_cfg, channel_mode=channel_mode)
    t0 = time.time()
    try:
        result = train_v4_localization(samples, cfg=cfg, grid=grid)
        dt = time.time() - t0
        log(f"V4-{name} done in {dt:.0f}s — val MAE = {result.val_mae_3d:.4f} m, p95 = {result.val_p95_3d:.4f} m")
        _save_predictions(out_dir / f"v4_{name}", result, samples)
        return {
            "channel_mode": channel_mode,
            "val_mae_3d": float(result.val_mae_3d),
            "val_p95_3d": float(result.val_p95_3d),
            "n_val": int(result.val_predictions.shape[0]),
            "n_train_recordings": len(result.train_recording_ids),
            "n_val_recordings": len(result.val_recording_ids),
            "train_loss_final": float(result.train_loss_history[-1])
            if result.train_loss_history else float("nan"),
            "val_loss_final": float(result.val_loss_history[-1])
            if result.val_loss_history else float("nan"),
        }
    except Exception as e:
        log(f"V4-{name} FAILED: {type(e).__name__}: {e}")
        return {"channel_mode": channel_mode, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True,
                    help="Run dir containing v1/{acoustic,vibration}.pt + v2/encoder.pt")
    ap.add_argument("--quick", action="store_true", help="15-epoch V4 smoke instead of 30")
    args = ap.parse_args()

    src_run = Path(args.source_run).resolve()
    if not src_run.exists():
        raise SystemExit(f"--source-run {src_run} does not exist")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "results" / "runs" / f"{timestamp}__v4_three_paradigms"
    (out_dir / "classical").mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with (out_dir / "run_log.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"src_run = {src_run}")
    log(f"out_dir = {out_dir}")

    v2_cfg = _v2_cfg(args.quick)
    v4_cfg = _v4_cfg(args.quick)

    log("Loading loaders ...")
    D2 = _resolved_loader("d2.yaml")
    D3 = _resolved_loader("d3.yaml"); D4 = _resolved_loader("d4.yaml")

    log("Loading V2 encoder ...")
    v2_encoder = _build_v2(v2_cfg)
    sd = torch.load(src_run / "v2" / "encoder.pt", map_location="cpu")
    v2_encoder.load_state_dict(sd, strict=False)
    v2_encoder.eval()

    # Gather labeled segments (mirror full_run.py:935-950).
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    log(f"  D2 labeled: {len(d2_labeled)} | D3 labeled: {len(d3_labeled)} | D4 labeled: {len(d4_labeled)}")

    grid = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))

    log("Precomputing V4 samples (one pass; multilat included per R3.3) ...")
    t0 = time.time()
    samples = precompute_v4_samples(
        v2_encoder,
        d2_labeled + d3_labeled + d4_labeled,
        v2_cfg=v2_cfg,
        grid=grid,
        spatial_label_overrides=overrides,
        burst_aware_srp=True,
        burst_seconds=0.10,
    )
    log(f"  {len(samples)} V4 samples in {time.time() - t0:.1f}s")
    n_with_multilat = sum(1 for s in samples if s.multilat_xyz is not None)
    log(f"  multilat init available on {n_with_multilat}/{len(samples)} samples")

    # Train four V4 instances (acoustic, vibration, fusion + tdoa_only legacy
    # for reference — that one's the existing "weak vibration" baseline).
    pipelines = [
        ("acoustic", "srp_only"),
        ("vibration", "vibration_only_learned"),
        ("vibration_tdoa_only_legacy", "tdoa_only"),
        ("fusion", "both"),
    ]
    metrics: dict = {
        "source_run": str(src_run.relative_to(REPO)),
        "n_samples_total": len(samples),
        "n_samples_with_multilat": n_with_multilat,
        "v4_cfg": {k: v for k, v in asdict(v4_cfg).items() if not isinstance(v, (list, tuple, dict))},
        "pipelines": {},
    }
    for name, mode in pipelines:
        if mode == "vibration_only_learned" and n_with_multilat < len(samples):
            log(f"NOTE: filtering to {n_with_multilat} samples with multilat for {name}")
            mode_samples = [s for s in samples if s.multilat_xyz is not None]
        else:
            mode_samples = samples
        metrics["pipelines"][name] = _train_one_v4(
            name, mode, mode_samples, grid, v4_cfg, out_dir, log,
        )

    # ── V0 classical baselines per dataset / recording ─────────────────
    log("\nV0 SRP-PHAT (acoustic classical) per dataset ...")
    v0_srp_results: dict = {}
    for L, name in [(D2, "d2"), (D3, "d3"), (D4, "d4")]:
        try:
            recs = evaluate_srp_phat(L, SRPConfig())
            s = summarise(recs)
            v0_srp_results[name] = s
            log(f"  {name}: {s.get('n_recordings', 0)} recordings, mean MAE = "
                f"{s.get('mean_error_m', float('nan')):.3f} m")
        except Exception as e:
            v0_srp_results[name] = {"error": str(e)}
            log(f"  {name}: skipped ({e})")
    with (out_dir / "classical" / "v0_srp_phat.json").open("w", encoding="utf-8") as fh:
        json.dump(v0_srp_results, fh, indent=2)

    log("\nV0 accel multilateration (vibration classical) per recording ...")
    v0_multilat: dict = {}
    for L, name in [(D2, "d2"), (D3, "d3"), (D4, "d4")]:
        per_rec: list[dict] = []
        for s in L.list_segments():
            if not s.is_anomaly or s.spatial_label is None:
                continue
            try:
                if s.segment.accel_data.shape[0] < 4:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "n_accel < 4"})
                    continue
                pos, residual = accel_tdoa_multilateration_v0(
                    s.segment.accel_data, s.vib_positions,
                    fs=float(s.segment.accel_sample_rate),
                )
                if name == "d3":
                    target = overrides.get(s.recording_id)
                else:
                    target = s.spatial_label
                if target is None:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "no spatial label"})
                    continue
                err = float(np.linalg.norm(pos - np.asarray(target, dtype=np.float64)))
                per_rec.append({
                    "recording_id": s.recording_id,
                    "target": list(map(float, target)),
                    "pred": list(map(float, pos)),
                    "residual": float(residual),
                    "error_m": err,
                })
            except Exception as e:
                per_rec.append({"recording_id": s.recording_id, "error": str(e)})
        errs = [r["error_m"] for r in per_rec if "error_m" in r]
        v0_multilat[name] = {
            "n_recordings": len(per_rec),
            "n_successful": len(errs),
            "mean_error_m": float(np.mean(errs)) if errs else float("nan"),
            "median_error_m": float(np.median(errs)) if errs else float("nan"),
            "p95_error_m": float(np.percentile(errs, 95)) if errs else float("nan"),
            "per_recording": per_rec,
        }
        log(f"  {name}: {len(errs)}/{len(per_rec)} resolved, "
            f"mean MAE = {v0_multilat[name]['mean_error_m']:.3f} m")
    with (out_dir / "classical" / "v0_multilat.json").open("w", encoding="utf-8") as fh:
        json.dump(v0_multilat, fh, indent=2)
    metrics["classical"] = {
        "v0_srp_phat": v0_srp_results,
        "v0_multilat_summary": {k: {
            kk: vv for kk, vv in v.items() if kk != "per_recording"
        } for k, v in v0_multilat.items()},
    }

    # Headline side-by-side summary.
    log("\n=== V4 four-paradigm headline ===")
    log(f"{'paradigm':<32} | {'MAE (m)':>9} | {'p95 (m)':>9} | n_val")
    for name, _ in pipelines:
        m = metrics["pipelines"][name]
        if "error" in m:
            log(f"V4-{name:<28} | ERROR: {m['error']}")
            continue
        log(f"V4-{name:<28} | {m['val_mae_3d']:>9.4f} | {m['val_p95_3d']:>9.4f} | {m['n_val']}")
    log("\nClassical:")
    for ds, s in v0_srp_results.items():
        if isinstance(s, dict) and "mean_error_m" in s:
            log(f"V0 SRP-PHAT ({ds}): mean MAE = {s['mean_error_m']:.3f} m")
    for ds, s in v0_multilat.items():
        log(f"V0 multilat ({ds}): mean MAE = {s['mean_error_m']:.3f} m  "
            f"({s['n_successful']}/{s['n_recordings']})")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"\nWrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
