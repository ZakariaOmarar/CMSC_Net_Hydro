"""Probe: does a clean healthy/anomaly separating threshold exist per modality?

Loads ONE valid (xt_pool-persisted) canonical run, scores the healthy hold-out
and the D2/D3/D4 anomaly cohorts under each modality, and reports:

  * AUC (anomaly-vs-healthy) per modality+cohort  -- the threshold-free ceiling.
  * Healthy FPR + per-cohort anomaly alert rate (TPR proxy) under three
    threshold strategies, all fit on the EVALUATION healthy hold-out so the
    operating point is honest and seed-stable:
      - saved      : the thresholds.npz p95 (current behaviour)
      - fresh_pc   : per-cluster p95 re-fit on the eval healthy hold-out
      - fresh_glob : a single global p95 on the eval healthy hold-out

Run:  python -m scripts.diagnostics.probe_v3_separation [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.threshold import PerClusterThresholds  # noqa: E402
from src.modeling.eval.rq2_three_paradigm_eval import (  # noqa: E402
    _build_loader,
    _build_v1,
    _build_v2,
    _load_state,
    _load_v3,
    _score_cohort_three_paradigms,
    _segments_for,
    _loader,
    _PipelineState,
    REPO as EVAL_REPO,
)
from src.modeling.anomaly.v3_per_modality import (  # noqa: E402
    V3AcousticOnlyAdapter,
    V3VibrationOnlyAdapter,
)
from src.modeling.orchestration.full_run import v1_config, v2_config  # noqa: E402

DEFAULT_RUN = REPO / "results" / "runs" / "20260616_022513__full_pipeline_b5_cma"


def _auc(healthy: np.ndarray, anom: np.ndarray) -> float:
    """Rank AUC: P(anom score > healthy score).  0.5 = no separation."""
    if healthy.size == 0 or anom.size == 0:
        return float("nan")
    allv = np.concatenate([healthy, anom])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    r_anom = ranks[healthy.size:].sum()
    n_h, n_a = healthy.size, anom.size
    return float((r_anom - n_a * (n_a + 1) / 2.0) / (n_h * n_a))


def main() -> int:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    src = run  # full_pipeline runs hold v1/v2 in the same dir
    print(f"run = {run.name}")

    v1_cfg = v1_config(False)
    v2_cfg = v2_config(False)
    embed = int(v1_cfg.embed_dim)

    v1_a = _build_v1("acoustic", v1_cfg); _load_state(src / "v1" / "acoustic.pt", v1_a); v1_a.eval()
    v1_v = _build_v1("vibration", v1_cfg); _load_state(src / "v1" / "vibration.pt", v1_v); v1_v.eval()
    v2 = _build_v2(v2_cfg); _load_state(src / "v2" / "encoder.pt", v2); v2.eval()

    flow_a, th_a, xt_a = _load_v3(run / "v3_acoustic", x_dim=embed, c_dim=embed)
    flow_v, th_v, xt_v = _load_v3(run / "v3_vibration", x_dim=embed, c_dim=embed)
    flow_f, th_f, xt_f = _load_v3(run / "v3_fusion", x_dim=embed, c_dim=embed)
    pipelines = [
        _PipelineState("acoustic", V3AcousticOnlyAdapter(v1_a), flow_a, th_a, xt_a),
        _PipelineState("vibration", V3VibrationOnlyAdapter(v1_v), flow_v, th_v, xt_v),
        _PipelineState("fusion", v2, flow_f, th_f, xt_f),
    ]

    loaders = [_loader(d) for d in ("d1", "d2", "d3", "d4")]
    print("gathering cohorts ...")
    healthy = _segments_for(loaders[:2], v2_cfg, healthy=True)
    cohorts = {"healthy": healthy}
    for key, idx in (("d2_anom", 1), ("d3_anom", 2), ("d4_anom", 3)):
        segs = _segments_for([loaders[idx]], v2_cfg, healthy=False)
        if segs:
            cohorts[key] = segs
    scored = {}
    for k, v in cohorts.items():
        print(f"scoring {k} ({len(v)} segs) ...", flush=True)
        scored[k] = _score_cohort_three_paradigms(pipelines, _build_loader(v, v2_cfg))

    for mod in ("acoustic", "vibration", "fusion"):
        h_s = scored["healthy"][mod]["scores"]
        h_c = scored["healthy"][mod]["contexts"]
        print(f"\n========== {mod} ==========")
        print(f"healthy NLL: p50={np.percentile(h_s,50):.1f} p95={np.percentile(h_s,95):.1f} "
              f"max={h_s.max():.1f}  (n={h_s.size})")

        # Three threshold strategies, all fit on the EVAL healthy hold-out.
        th_saved = pipelines[("acoustic", "vibration", "fusion").index(mod)].thresholds
        th_pc = PerClusterThresholds.fit(h_c, h_s, n_clusters=3, seed=42)
        glob_thr = float(np.percentile(h_s, 95))

        def rates(scores, ctx):
            a_saved = (th_saved.alert(ctx, scores, percentile=95)[0]).mean()
            a_pc = (th_pc.alert(ctx, scores, percentile=95)[0]).mean()
            a_glob = (scores > glob_thr).mean()
            return a_saved, a_pc, a_glob

        print(f"{'cohort':<10} {'AUC':>6} | {'saved':>7} {'fresh_pc':>9} {'fresh_glob':>11}")
        for ck in cohorts:
            s = scored[ck][mod]["scores"]; c = scored[ck][mod]["contexts"]
            auc = _auc(h_s, s) if ck != "healthy" else float("nan")
            a_saved, a_pc, a_glob = rates(s, c)
            tag = "(FPR)" if ck == "healthy" else ""
            print(f"{ck:<10} {auc:>6.3f} | {a_saved:>7.3f} {a_pc:>9.3f} {a_glob:>11.3f} {tag}")

    # --- combined rules at the dynamic 5%-FPR per-modality operating point ---
    # Each modality thresholded at the global p95 of its OWN eval-healthy scores
    # (FPR pinned to 5% by construction), then AND / OR over the per-window masks.
    print("\n========== combined (per-modality threshold = eval-healthy p95) ==========")
    thr_a = float(np.percentile(scored["healthy"]["acoustic"]["scores"], 95))
    thr_v = float(np.percentile(scored["healthy"]["vibration"]["scores"], 95))
    print(f"{'cohort':<10} {'AND':>6} {'OR':>6} {'acoustic':>9} {'vibration':>10}")
    for ck in cohorts:
        am = scored[ck]["acoustic"]["scores"] > thr_a
        vm = scored[ck]["vibration"]["scores"] > thr_v
        tag = "(FPR)" if ck == "healthy" else ""
        print(f"{ck:<10} {(am & vm).mean():>6.3f} {(am | vm).mean():>6.3f} "
              f"{am.mean():>9.3f} {vm.mean():>10.3f} {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
