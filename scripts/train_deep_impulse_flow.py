"""Train + evaluate the deep impulse-aware anomaly flow (learned front-end).

One-class: a CNN front-end (spectrogram by default; preserves spectral AND
transient cues) is trained END-TO-END with a conditional flow on the HEALTHY
NLL, with the hand-crafted impulse+spectral anchor concatenated (recall
guarantee + anti-collapse).  Per modality (mic, accel); anomaly score = sum of
z-scored per-modality NLLs.  Light time-shift/noise augmentation for small-data
generalization.

Fit on FIT datasets' healthy; D5 (or any non-fit) is held out to verify the
theory on a new campaign.  Reports recording-level ROC/PR and window-level
(healthy-referenced) detection at a healthy-calibrated operating point.

Run:
    python -m scripts.train_deep_impulse_flow --epochs 40
    python -m scripts.train_deep_impulse_flow --smoke
"""
from __future__ import annotations

import argparse
import json
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

N_MELS, N_T, N_ANCHOR = 64, 64, 16
WIN_S, STRIDE_S = 1.0, 0.5


def _spectrogram(w: np.ndarray, fs: float) -> np.ndarray:
    """log-STFT magnitude, freq-binned to N_MELS, time-resized to N_T."""
    if w.size < 32 or not np.any(w):
        return np.zeros((N_MELS, N_T))
    from scipy.signal import stft
    nper = min(512, max(32, w.size // 16))
    _, _, Z = stft(w.astype(np.float64), fs=fs, nperseg=nper, noverlap=nper // 2)
    mag = np.log1p(np.abs(Z))                                   # (F, Tf)
    F, Tf = mag.shape
    if F >= N_MELS:                                             # downscale: average-bin
        mel = np.stack([b.mean(0) for b in np.array_split(mag, N_MELS, axis=0)])
    else:                                                       # upscale: interpolate freq
        fi = np.linspace(0, F - 1, N_MELS)
        mel = np.stack([np.interp(fi, np.arange(F), mag[:, j]) for j in range(Tf)], axis=1)
    if mel.shape[1] != N_T:                                     # resize time
        xi = np.linspace(0, mel.shape[1] - 1, N_T)
        mel = np.stack([np.interp(xi, np.arange(mel.shape[1]), mel[i]) for i in range(N_MELS)])
    return mel


def _collect(loader, is_anom: bool):
    msp, asp, manc, aanc, env, peak = [], [], [], [], [], []
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
            msp.append(_spectrogram(seg, fs_m)); manc.append(window_features(seg, fs_m))
            peak.append(float(np.abs(seg).max()) if seg.size else 0.0)
            sa = wa[int(t0 * fs_a):int(t1 * fs_a)] if wa is not None else np.zeros(0)
            asp.append(_spectrogram(sa, fs_a)); aanc.append(window_features(sa, fs_a))
            env.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return {"msp": np.asarray(msp), "asp": np.asarray(asp),
            "manc": np.asarray(manc), "aanc": np.asarray(aanc),
            "env": np.asarray(env), "peak": np.asarray(peak)}


def _augment(spec: torch.Tensor) -> torch.Tensor:
    """Time-roll + small gaussian noise (small-data regularization)."""
    shift = int(torch.randint(-N_T // 8, N_T // 8 + 1, (1,)))
    return torch.roll(spec, shifts=shift, dims=-1) + 0.02 * torch.randn_like(spec)


def _train_one(model, spec, anc, dev, epochs, bs, lr, augment):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    st = torch.tensor(spec, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
    n = st.shape[0]; model.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            sb = st[idx].to(dev); ab = at[idx].to(dev)
            if augment:
                sb = _augment(sb)
            loss = -model.log_prob(sb, ab).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()) * len(idx)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}  NLL={tot/n:.3f}", flush=True)
    return model


def _scores(model, spec, anc, dev) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        st = torch.tensor(spec, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
        for i in range(0, st.shape[0], 512):
            out.append(model.anomaly_score(st[i:i+512].to(dev), at[i:i+512].to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-ds", nargs="*", default=["d2", "d3", "d4"])
    ap.add_argument("--test-ds", nargs="*", default=["d2", "d3", "d4", "d5"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default="results/deep_impulse_flow.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {dev}")
    augment = not args.no_augment

    if args.smoke:
        m = DeepImpulseFlow(N_ANCHOR, front="spectro").to(dev)
        sp = np.random.randn(96, N_MELS, N_T); an = np.random.randn(96, N_ANCHOR)
        _train_one(m, sp, an, dev, 2, 32, 1e-3, augment)
        print("smoke scores:", _scores(m, sp[:8], an[:8], dev).shape)
        return 0

    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.modeling.eval.rq2_three_paradigm_eval import _loader
    all_ds = sorted(set(args.fit_ds) | set(args.test_ds))
    H, A = {}, {}
    print("collecting spectrograms + anchors ...", flush=True)
    for dn in all_ds:
        L = _loader(dn)
        H[dn] = _collect(L, False); A[dn] = _collect(L, True)
        floor = float(np.percentile(H[dn]["peak"], 99.5)) if H[dn]["peak"].size else float("inf")
        A[dn]["ref"] = (A[dn]["peak"] > floor).astype(int)
        print(f"  {dn}: healthy={H[dn]['msp'].shape[0]} anom={A[dn]['msp'].shape[0]}", flush=True)

    fm = np.concatenate([H[d]["manc"] for d in args.fit_ds])
    fa = np.concatenate([H[d]["aanc"] for d in args.fit_ds])
    mmu, msd = fm.mean(0), fm.std(0) + 1e-8
    amu, asd = fa.mean(0), fa.std(0) + 1e-8

    mic = DeepImpulseFlow(N_ANCHOR, front="spectro").to(dev)
    acc = DeepImpulseFlow(N_ANCHOR, front="spectro").to(dev)
    print("training mic model ...", flush=True)
    _train_one(mic, np.concatenate([H[d]["msp"] for d in args.fit_ds]), (fm - mmu) / msd,
               dev, args.epochs, args.batch_size, args.lr, augment)
    print("training accel model ...", flush=True)
    _train_one(acc, np.concatenate([H[d]["asp"] for d in args.fit_ds]), (fa - amu) / asd,
               dev, args.epochs, args.batch_size, args.lr, augment)

    hzm = _scores(mic, np.concatenate([H[d]["msp"] for d in args.fit_ds]), (fm - mmu) / msd, dev)
    hza = _scores(acc, np.concatenate([H[d]["asp"] for d in args.fit_ds]), (fa - amu) / asd, dev)
    mz = (hzm.mean(), hzm.std() + 1e-8); ma = (hza.mean(), hza.std() + 1e-8)

    def fused(d):
        zm = (_scores(mic, d["msp"], (d["manc"] - mmu) / msd, dev) - mz[0]) / mz[1]
        za = (_scores(acc, d["asp"], (d["aanc"] - amu) / asd, dev) - ma[0]) / ma[1]
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
                "score_norm": {"mic": mz, "acc": ma}, "threshold": thr, "cfg": vars(args)}, out)
    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nsaved -> {out.relative_to(REPO)}\nsaved -> {out.with_suffix('.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
