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
    # Per-window instance normalization: removes the regime-level offset so the
    # CNN front-end sees a regime-INVARIANT shape (the absolute energy/impulse
    # signal is kept by the hand-crafted anchor).  This is what lets the learned
    # features transfer zero-shot to an unseen campaign, and it bounds the input
    # so the flow's val NLL stops blowing up.
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
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
    """Strong regime-simulating augmentation for DOMAIN GENERALIZATION.

    The deep model fails on a new campaign because it overfits the training
    regimes' absolute spectra.  Simulating regime shift during training -
    random gain, frequency shift, time shift, SpecAugment freq/time masking,
    and noise - forces the front-end to learn regime-INVARIANT healthy
    structure, so an unseen campaign falls inside the trained "healthy"
    distribution.  Applied to healthy training windows only; the hand-crafted
    anchor (separate input) still carries the absolute impulse signal.
    """
    x = spec.clone()
    B, F, T = x.shape
    dev = x.device
    # random per-sample gain (level invariance)
    x = x * (1.0 + 0.4 * (torch.rand(B, 1, 1, device=dev) - 0.5))
    # frequency shift (spectral-regime invariance) + time shift
    x = torch.roll(x, shifts=int(torch.randint(-F // 6, F // 6 + 1, (1,))), dims=1)
    x = torch.roll(x, shifts=int(torch.randint(-T // 6, T // 6 + 1, (1,))), dims=2)
    # SpecAugment: zero a random frequency band and a random time span
    fm = int(torch.randint(0, F // 4 + 1, (1,))); f0 = int(torch.randint(0, max(1, F - fm), (1,)))
    x[:, f0:f0 + fm, :] = 0.0
    tm = int(torch.randint(0, T // 4 + 1, (1,))); t0 = int(torch.randint(0, max(1, T - tm), (1,)))
    x[:, :, t0:t0 + tm] = 0.0
    return x + 0.05 * torch.randn_like(x)


def _train_one(model, spec, anc, dev, epochs, bs, lr, augment,
               patience=8, val_frac=0.15):
    """One-class training with held-out-healthy early stopping (restore-best).

    A val_frac slice of the healthy windows is held out; we monitor val NLL and
    stop when it stops improving for `patience` epochs, restoring the best-val
    weights — the standard guard against the CNN overfitting the healthy set.
    """
    st = torch.tensor(spec, dtype=torch.float32); at = torch.tensor(anc, dtype=torch.float32)
    n = st.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = max(1, int(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_val, best_state, bad, best_ep = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        tp = tr_idx[torch.randperm(tr_idx.shape[0])]; tot = 0.0
        for i in range(0, tp.shape[0], bs):
            idx = tp[i:i + bs]
            sb = st[idx].to(dev); ab = at[idx].to(dev)
            if augment:
                sb = _augment(sb)
            loss = -model.log_prob(sb, ab).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vt = 0.0
            for i in range(0, n_val, 512):
                vi = val_idx[i:i + 512]
                vt += float(-model.log_prob(st[vi].to(dev), at[vi].to(dev)).sum())
            val_nll = vt / n_val
        if val_nll < best_val - 1e-3:
            best_val, bad, best_ep = val_nll, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep == 0 or (ep + 1) % 5 == 0 or bad >= patience or ep == epochs - 1:
            print(f"    epoch {ep+1}/{epochs}  train_NLL={tot/tr_idx.shape[0]:.3f}  "
                  f"val_NLL={val_nll:.3f}  best@{best_ep+1}", flush=True)
        if bad >= patience:
            print(f"    early stop at epoch {ep+1} (best val @ {best_ep+1})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
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
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience (epochs)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="healthy val split for early stop")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--adapt-frac", type=float, default=0.0,
                    help="few-shot: fraction of a held-out campaign's healthy to "
                         "fine-tune/recalibrate on (0 = off)")
    ap.add_argument("--adapt-epochs", type=int, default=10)
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
        _train_one(m, sp, an, dev, 6, 32, 1e-3, augment, patience=2, val_frac=0.2)
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
               dev, args.epochs, args.batch_size, args.lr, augment, args.patience, args.val_frac)
    print("training accel model ...", flush=True)
    _train_one(acc, np.concatenate([H[d]["asp"] for d in args.fit_ds]), (fa - amu) / asd,
               dev, args.epochs, args.batch_size, args.lr, augment, args.patience, args.val_frac)

    hzm = _scores(mic, np.concatenate([H[d]["msp"] for d in args.fit_ds]), (fm - mmu) / msd, dev)
    hza = _scores(acc, np.concatenate([H[d]["asp"] for d in args.fit_ds]), (fa - amu) / asd, dev)
    mz = (hzm.mean(), hzm.std() + 1e-8); ma = (hza.mean(), hza.std() + 1e-8)

    def _fused(mic_m, acc_m, mz_, ma_, d, sel=slice(None)):
        zm = (_scores(mic_m, d["msp"][sel], (d["manc"][sel] - mmu) / msd, dev) - mz_[0]) / mz_[1]
        za = (_scores(acc_m, d["asp"][sel], (d["aanc"][sel] - amu) / asd, dev) - ma_[0]) / ma_[1]
        return zm + za

    hf = (hzm - mz[0]) / mz[1] + (hza - ma[0]) / ma[1]
    thr = float(np.percentile(hf, 100 * (1 - args.target_fpr)))
    res = {"fit_ds": args.fit_ds, "target_fpr": args.target_fpr,
           "healthy_fpr": float((hf > thr).mean()), "per_dataset": {}}
    print(f"\nhealthy FPR @ threshold = {res['healthy_fpr']:.3f}\n")
    print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'href_ROC':>9} {'flag@thr':>9} {'heldout':>8}")
    for dn in args.test_ds:
        zh = _fused(mic, acc, mz, ma, H[dn]); za = _fused(mic, acc, mz, ma, A[dn])
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

    # ---- few-shot domain adaptation to held-out campaigns ----
    if args.adapt_frac > 0:
        import copy
        print(f"\n=== few-shot adaptation (adapt on {args.adapt_frac:.0%} of held-out "
              f"healthy, fine-tune {args.adapt_epochs} ep) ===")
        print(f"{'ds':<5} {'reclvl_ROC':>10} {'reclvl_PR':>10} {'flag@thr':>9} {'eval_FPR':>9}")
        for dn in args.test_ds:
            if dn in args.fit_ds:
                continue
            nH = H[dn]["msp"].shape[0]
            idx = np.random.default_rng(0).permutation(nH)
            na = max(16, int(args.adapt_frac * nH))
            ai, ei = idx[:na], idx[na:]
            micA, accA = copy.deepcopy(mic), copy.deepcopy(acc)
            _train_one(micA, H[dn]["msp"][ai], (H[dn]["manc"][ai] - mmu) / msd, dev,
                       args.adapt_epochs, args.batch_size, args.lr * 0.2, augment,
                       patience=args.adapt_epochs, val_frac=0.2)
            _train_one(accA, H[dn]["asp"][ai], (H[dn]["aanc"][ai] - amu) / asd, dev,
                       args.adapt_epochs, args.batch_size, args.lr * 0.2, augment,
                       patience=args.adapt_epochs, val_frac=0.2)
            # recalibrate per-modality score-norm + threshold on the adapt slice
            zmh = _scores(micA, H[dn]["msp"][ai], (H[dn]["manc"][ai] - mmu) / msd, dev)
            zah = _scores(accA, H[dn]["asp"][ai], (H[dn]["aanc"][ai] - amu) / asd, dev)
            mzN = (zmh.mean(), zmh.std() + 1e-8); maN = (zah.mean(), zah.std() + 1e-8)
            hfa = (zmh - mzN[0]) / mzN[1] + (zah - maN[0]) / maN[1]
            thrA = float(np.percentile(hfa, 100 * (1 - args.target_fpr)))
            zh = _fused(micA, accA, mzN, maN, H[dn], ei)      # held-out eval healthy
            za = _fused(micA, accA, mzN, maN, A[dn])
            y = np.concatenate([np.zeros(zh.size), np.ones(za.size)]); s = np.concatenate([zh, za])
            rr, rp = roc_auc_score(y, s), average_precision_score(y, s)
            res["per_dataset"][dn]["adapted"] = {
                "reclvl_roc": float(rr), "reclvl_pr": float(rp),
                "flag_at_thr": float((za > thrA).mean()), "eval_fpr": float((zh > thrA).mean())}
            print(f"{dn:<5} {rr:>10.3f} {rp:>10.3f} {(za > thrA).mean():>9.3f} {(zh > thrA).mean():>9.3f}")

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
