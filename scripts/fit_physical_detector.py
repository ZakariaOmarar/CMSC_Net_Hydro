"""Fit + evaluate the physical-feature anomaly detector on a trained run.

Closed-form, label-free, NO retraining: uses the run's existing V2 encoder only
to produce the context vector `c` for regime-normalization.  Fits on pooled
healthy, evaluates per campaign (AUC vs pooled healthy + detection at the global
threshold), and persists the fitted detector to ``<run>/physical_detector.pkl``.

Run:
    python -m scripts.fit_physical_detector --run results/runs/<id>
    python -m scripts.fit_physical_detector --run results/runs/<id> --normalizer ridge
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.physical_detector import (  # noqa: E402
    PhysicalFeatureDetector, physical_features,
)
from src.modeling.eval.rq2_three_paradigm_eval import (  # noqa: E402
    _build_loader, _build_v2, _load_state, _loader, _segments_for,
)
from src.modeling.orchestration.full_run import v2_config  # noqa: E402

DS = ("d1", "d2", "d3", "d4")


def _auc(h: np.ndarray, a: np.ndarray) -> float:
    if h.size == 0 or a.size == 0:
        return float("nan")
    allv = np.concatenate([h, a]); r = allv.argsort().argsort() + 1.0
    return float((r[h.size:].sum() - a.size * (a.size + 1) / 2) / (h.size * a.size))


def _collect(segs, v2, cfg):
    """Return phys={'ac','vib': (N,5)}, ctx (N, c_dim) over all windows in segs."""
    ac, vib, ctx = [], [], []
    with torch.no_grad():
        for b in _build_loader(segs, cfg):
            d = v2(b["ac_feat"], b["ac_xyz"], b["vib_feat"], b["vib_xyz"],
                   b["dataset_idx"], mask_p=0.0)
            ctx.append(d["context"].cpu().numpy())
            ac.append(physical_features(b["ac_feat"].numpy()))
            vib.append(physical_features(b["vib_feat"].numpy()))
    return ({"ac": np.concatenate(ac), "vib": np.concatenate(vib)},
            np.concatenate(ctx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with v2/encoder.pt")
    ap.add_argument("--normalizer", choices=("knn", "ridge"), default="knn")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    run = Path(args.run).resolve()

    cfg = v2_config(False)
    v2 = _build_v2(cfg); _load_state(run / "v2" / "encoder.pt", v2); v2.eval()
    loaders = [_loader(d) for d in DS]

    print("collecting healthy + anomaly windows ...", flush=True)
    H_phys, H_ctx, A = {}, {}, {}
    for i, dn in enumerate(DS):
        h = _segments_for([loaders[i]], cfg, healthy=True)
        a = _segments_for([loaders[i]], cfg, healthy=False)
        if h:
            print(f"  {dn} healthy ({len(h)} segs)", flush=True)
            H_phys[dn], H_ctx[dn] = _collect(h, v2, cfg)
        if a:
            print(f"  {dn} anomaly ({len(a)} segs)", flush=True)
            A[dn] = _collect(a, v2, cfg)

    phys_h = {m: np.concatenate([H_phys[dn][m] for dn in H_phys]) for m in ("ac", "vib")}
    ctx_h = np.concatenate([H_ctx[dn] for dn in H_ctx])

    det = PhysicalFeatureDetector(
        normalizer=args.normalizer, k=args.k, target_fpr=args.target_fpr,
    ).fit(phys_h, ctx_h)

    sh = det.fused_score(phys_h, ctx_h)
    mh = det.modality_scores(phys_h, ctx_h)
    print(f"\n=== physical detector ({args.normalizer}, target_fpr={args.target_fpr:.0%}) ===")
    print(f"pooled-healthy fused FPR @ threshold = {(sh > det.threshold).mean():.3f}")
    print(f"{'ds':<6} {'ac_AUC':>7} {'vib_AUC':>8} {'SUM_AUC':>8} {'detect@thr':>11}")
    for dn in A:
        ph, ct = A[dn]
        ms = det.modality_scores(ph, ct)
        fs = det.fused_score(ph, ct)
        print(f"{dn:<6} {_auc(mh['ac'], ms['ac']):>7.3f} {_auc(mh['vib'], ms['vib']):>8.3f} "
              f"{_auc(sh, fs):>8.3f} {(fs > det.threshold).mean():>11.3f}")

    out = run / "physical_detector.pkl"
    with out.open("wb") as fh:
        pickle.dump(det, fh)
    print(f"\nsaved -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
