"""Train + evaluate the deep impulse-aware anomaly flow (learned raw front-end).

One-class: a 1-D CNN reads raw windows and is trained END-TO-END with a
conditional flow on the HEALTHY NLL, with the hand-crafted impulse+spectral
anchor concatenated (recall guarantee + anti-collapse).  Per modality (mic,
accel); anomaly score = sum of z-scored per-modality NLLs.

Fit on FIT datasets' healthy; D5 (or any non-fit) is held out to verify the
theory on a new campaign.  Reports recording-level ROC/PR and window-level
(healthy-referenced) detection, plus a healthy-calibrated operating point.

Run:
    python -m scripts.train_deep_impulse_flow                 # GPU auto
    python -m scripts.train_deep_impulse_flow --epochs 40 --fit-ds d2 d3 d4 --test-ds d2 d3 d4 d5
    python -m scripts.train_deep_impulse_flow --smoke         # tiny self-test, no data
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.deep_impulse_flow import DeepImpulseFlow  # noqa: E402
from src.modeling.anomaly.raw_impulse_detector import window_features  # noqa: E402
from src.modeling.anomaly.weak_labels import derive_knock_events, window_overlaps_any  # noqa: E402

L_MIC, L_ACC, N_ANCHOR = 8192, 512, 16
WIN_S, STRIDE_S = 1.0, 0.5


def _resample(w: np.ndarray, L: int) -> np.ndarray:
    if w.size == 0:
        return np.zeros(L)
    if w.size == L:
        return w.astype(np.float64)
    from scipy.signal import resample
    return resample(w.astype(np.float64), L)


def _collect(loader, is_anom: bool):
    mraw, araw, manc, aanc, env, peak = [], [], [], [], [], []
    for s in loader.list_segments():
        if s.is_anomaly != is_anom:
            continue
        ds = s.segment
        mic = getattr(ds, "mic_data", None)
        if mic is None or not mic.size:
            continue
        fs_m = float(ds.mic_sample_rate)
        acc = getattr(ds, "accel_data", None)
        fs_a = float(getattr(ds, "accel_sample_rate", 0) or 1.0)
        intervals = derive_knock_events(s, max_events=300, noise_floor_mult=3.0) if is_anom else []
        wm = np.sqrt(np.mean(mic.astype(np.float64) ** 2, axis=0))
        wa = (np.sqrt(np.mean(acc.astype(np.float64) ** 2, axis=0))
              if acc is not None and acc.size else None)
        T = wm.size; step = max(1, int(STRIDE_S * fs_m)); wlen = int(WIN_S * fs_m)
        for i0 in range(0, max(1, T - wlen + 1), step):
            t0, t1 = i0 / fs_m, (i0 + wlen) / fs_m
            seg = wm[i0:i0 + wlen]
            mraw.append(_resample(seg, L_MIC)); manc.append(window_features(seg, fs_m))
            peak.append(float(np.abs(seg).max()) if seg.size else 0.0)
            sa = wa[int(t0 * fs_a):int(t1 * fs_a)] if wa is not None else np.zeros(0)
            araw.append(_resample(sa, L_ACC)); aanc.append(window_features(sa, fs_a))
            env.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return {"mraw": np.asarray(mraw), "araw": np.asarray(araw),
            "manc": np.asarray(manc), "aanc": np.asarray(aanc),
            "env": np.asarray(env), "peak": np.asarray(peak)}


def _train_one(model, raw, anc, dev, epochs, bs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    raw_t = torch.tensor(raw, dtype=torch.float32)
    anc_t = torch.tensor(anc, dtype=torch.float32)
    n = raw_t.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            rb = raw_t[idx].to(dev); ab = anc_t[idx].to(dev)
            loss = -model.log_prob(rb, ab).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()) * len(idx)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}  NLL={tot/n:.3f}", flush=True)
    return model


def _scores(model, raw, anc, dev) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        rt = torch.tensor(raw, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
        for i in range(0, rt.shape[0], 512):
            out.append(model.anomaly_score(rt[i:i+512].to(dev), at[i:i+512].to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _zfit(s):
    return float(s.mean()), float(s.std() + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ds", nargs="*", default=["d2", "d3", "d4"])
    ap.add_argument("--test-ds", nargs="*", default=["d2", "d3", "d4", "d5"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--out", default="results/deep_impulse_flow.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic self-test, no data")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")

    if args.smoke:
        mic = DeepImpulseFlow(L_MIC, N_ANCHOR).to(dev)
        raw = np.random.randn(128, L_MIC); anc = np.random.randn(128, N_ANCHOR)
        _train_one(mic, raw, anc, dev, epochs=2, bs=32, lr=1e-3)
        print("smoke scores:", _scores(mic, raw[:8], anc[:8], dev).shape)
        return 0

    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.modeling.eval.rq2_three_paradigm_eval import _loader
    all_ds = sorted(set(args.fit_ds) | set(args.test_ds))
    H, A = {}, {}
    print("collecting raw windows + anchors ...", flush=True)
    for dn in all_ds:
        L = _loader(dn)
        H[dn] = _collect(L, False); A[dn] = _collect(L, True)
        floor = float(np.percentile(H[dn]["peak"], 99.5)) if H[dn]["peak"].size else float("inf")
        A[dn]["ref"] = (A[dn]["peak"] > floor).astype(int)
        print(f"  {dn}: healthy={H[dn]['mraw'].shape[0]} anom={A[dn]['mraw'].shape[0]}", flush=True)

    # anchor standardization on FIT healthy
    fm = np.concatenate([H[d]["manc"] for d in args.fit_ds])
    fa = np.concatenate([H[d]["aanc"] for d in args.fit_ds])
    mmu, msd = fm.mean(0), fm.std(0) + 1e-8
    amu, asd = fa.mean(0), fa.std(0) + 1e-8
    def stdm(x): return (x - mmu) / msd
    def stda(x): return (x - amu) / asd

    mic = DeepImpulseFlow(L_MIC, N_ANCHOR).to(dev)
    acc = DeepImpulseFlow(L_ACC, N_ANCHOR).to(dev)
    print("training mic model ...", flush=True)
    _train_one(mic, np.concatenate([H[d]["mraw"] for d in args.fit_ds]), stdm(fm), dev,
               args.epochs, args.batch_size, args.lr)
    print("training accel model ...", flush=True)
    _train_one(acc, np.concatenate([H[d]["araw"] for d in args.fit_ds]), stda(fa), dev,
               args.epochs, args.batch_size, args.lr)

    # healthy z-norm + fused threshold on FIT healthy
    hzm = _scores(mic, np.concatenate([H[d]["mraw"] for d in args.fit_ds]), stdm(fm), dev)
    hza = _scores(acc, np.concatenate([H[d]["araw"] for d in args.fit_ds]), stda(fa), dev)
    mz, ma = _zfit(hzm), _zfit(hza)
    def fused(d):
        zm = (_scores(mic, d["mraw"], stdm(d["manc"]), dev) - mz[0]) / mz[1]
        za = (_scores(acc, d["araw"], stda(d["aanc"]), dev) - ma[0]) / ma[1]
        return zm + za
    hf = (hzm - mz[0]) / mz[1] + (hza - ma[0]) / ma[1]
    thr = float(np.percentile(hf, 100 * (1 - args.target_fpr)))

    res = {"fit_ds": args.fit_ds, "target_fpr": args.target_fpr,
           "healthy_fpr": float((hf > thr).mean()), "per_dataset": {}}
    print(f"\nhealthy FPR @ threshold = {res['healthy_fpr']:.3f}\n")
    print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'href_ROC':>9} {'flag@thr':>9} {'heldout':>8}")
    for dn in args.test_ds:
        zh = fused(H[dn]); za = fused(A[dn])
        y = np.concatenate([np.zeros(zh.size), np.ones(za.size)]); s = np.concatenate([zh, za])
        rr, rp = roc_auc_score(y, s), average_precision_score(y, s)
        yr = A[dn]["ref"]
        hr = roc_auc_score(yr, za) if 0 < yr.sum() < yr.size else float("nan")
        flag = float((za > thr).mean())
        res["per_dataset"][dn] = {"reclvl_roc": float(rr), "reclvl_pr": float(rp),
                                  "href_roc": float(hr), "flag_at_thr": flag,
                                  "held_out": dn not in args.fit_ds}
        print(f"{dn:<5} {rr:>10.3f} {rp:>10.3f} {hr if not np.isnan(hr) else 0:>9.3f} "
              f"{flag:>9.3f} {'YES' if dn not in args.fit_ds else '':>8}")

    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mic": mic.state_dict(), "acc": acc.state_dict(),
                "anchor_stats": {"mmu": mmu, "msd": msd, "amu": amu, "asd": asd},
                "score_norm": {"mic": mz, "acc": ma}, "threshold": thr,
                "cfg": vars(args)}, out)
    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nsaved -> {out.relative_to(REPO)}\nsaved -> {out.with_suffix('.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
