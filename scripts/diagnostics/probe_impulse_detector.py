"""Probe: raw-waveform impulse detector evaluated on WINDOW-LEVEL knock labels.

Two upgrades over the mel-feature probes:
  1. Ground truth at WINDOW level, not cohort level: `derive_knock_intervals`
     (Hilbert envelope + iterative peak-picking) marks WHEN knocks occur, so a
     window is positive iff it overlaps a derived knock interval.  This removes
     the cohort-dilution that capped D4 and the arbitrary 5%-FPR operating point.
  2. SOTA impulsive-fault features on the RAW waveform (not the smeared mel):
     crest / impulse / clearance / shape factors, kurtosis, spectral kurtosis
     (Antoni), and envelope knock-count.  These are scale/regime-invariant
     ratios, so a plain healthy Mahalanobis (no context-norm) should suffice.

Detector is fit on HEALTHY windows only (unsupervised); labels are used ONLY for
evaluation (ROC-AUC + PR-AUC + best-F1), per dataset, per modality + SUM.

Run:  python -m scripts.diagnostics.probe_impulse_detector
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.modeling.anomaly.weak_labels import derive_knock_events, window_overlaps_any  # noqa: E402
from src.modeling.eval.rq2_three_paradigm_eval import _loader  # noqa: E402

WIN_S, STRIDE_S = 1.0, 0.5
DS = ("d2", "d3", "d4")


def _impulse_feats(w: np.ndarray, fs: float) -> np.ndarray:
    """Impulsive-fault condition indicators on a 1-D window `w`."""
    w = np.asarray(w, dtype=np.float64)
    if w.size < 8 or not np.any(w):
        return np.zeros(8)
    aw = np.abs(w)
    rms = np.sqrt(np.mean(w * w)) + 1e-12
    peak = float(aw.max())
    mean_abs = float(aw.mean()) + 1e-12
    crest = peak / rms
    impulse = peak / mean_abs
    clearance = peak / (np.mean(np.sqrt(aw)) ** 2 + 1e-12)
    shape = rms / mean_abs
    mu = w.mean(); sd = w.std() + 1e-12
    kurt = float(np.mean(((w - mu) / sd) ** 4) - 3.0)
    # spectral kurtosis (Antoni): kurtosis across time of each STFT bin, max.
    sk = 0.0
    try:
        from scipy.signal import stft
        nper = min(256, max(16, w.size // 8))
        _, _, Z = stft(w, fs=fs, nperseg=nper, noverlap=nper // 2)
        mag = np.abs(Z)  # (F, Tframes)
        if mag.shape[1] >= 4:
            m = mag.mean(1, keepdims=True); s = mag.std(1, keepdims=True) + 1e-12
            sk = float(np.nanmax((((mag - m) / s) ** 4).mean(1) - 3.0))
    except Exception:
        sk = 0.0
    # envelope knock-count: peaks above 3x median of |w| envelope.
    thr = 3.0 * (np.median(aw) + 1e-12)
    above = aw > thr
    kcount = float(np.sum(np.diff(above.astype(int)) == 1))
    return np.array([crest, impulse, clearance, shape, kurt, sk, kcount, peak / (np.median(aw) + 1e-12)])


def _collect(loader, is_anom: bool):
    """Return X_mic, X_acc (n_win, n_feat), labels (n_win,) over recordings."""
    Xm, Xa, y = [], [], []
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
        # window-level knock labels (only meaningful inside anomaly recordings;
        # healthy recordings are all-negative by construction).
        intervals = derive_knock_events(s, max_events=300, noise_floor_mult=3.0) if is_anom else []
        wm = np.sqrt(np.mean(mic.astype(np.float64) ** 2, axis=0))   # RMS-across-channels
        wa = (np.sqrt(np.mean(acc.astype(np.float64) ** 2, axis=0))
              if acc is not None and acc.size else None)
        T = wm.size; step = int(STRIDE_S * fs_m); wlen = int(WIN_S * fs_m)
        for i0 in range(0, max(1, T - wlen + 1), max(1, step)):
            t0, t1 = i0 / fs_m, (i0 + wlen) / fs_m
            Xm.append(_impulse_feats(wm[i0:i0 + wlen], fs_m))
            if wa is not None:
                a0, a1 = int(t0 * fs_a), int(t1 * fs_a)
                Xa.append(_impulse_feats(wa[a0:a1], fs_a))
            else:
                Xa.append(np.zeros(8))
            y.append(1 if (is_anom and window_overlaps_any(t0, t1, intervals)) else 0)
    return np.array(Xm), np.array(Xa), np.array(y)


def _maha_fit(Xh: np.ndarray):
    mu = Xh.mean(0); cov = np.cov(Xh, rowvar=False) + 1e-6 * np.eye(Xh.shape[1])
    inv = np.linalg.inv(cov); s = np.einsum("ij,jk,ik->i", Xh - mu, inv, Xh - mu)
    return mu, inv, s.mean(), s.std() + 1e-8


def _maha_z(X, p):
    mu, inv, m, sd = p
    return (np.einsum("ij,jk,ik->i", X - mu, inv, X - mu) - m) / sd


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score
    loaders = {dn: _loader(dn) for dn in DS}
    print("collecting raw-impulse features + window knock-labels ...", flush=True)
    Hm, Ha, Am, Aa, Ay, Ad = [], [], {}, {}, {}, {}
    for dn in DS:
        hm, ha, _ = _collect(loaders[dn], False)
        am, aa, ay = _collect(loaders[dn], True)
        Hm.append(hm); Ha.append(ha)
        Am[dn], Aa[dn], Ay[dn] = am, aa, ay
        print(f"  {dn}: healthy_win={hm.shape[0]} anom_win={am.shape[0]} "
              f"knock+={int(ay.sum())} ({ay.mean():.2f} of anom cohort)", flush=True)
    Hm = np.concatenate(Hm); Ha = np.concatenate(Ha)
    # standardize then Mahalanobis (per modality), fit on healthy.
    def _std(X, m, s): return (X - m) / s
    mm, ms = Hm.mean(0), Hm.std(0) + 1e-8
    am_, as_ = Ha.mean(0), Ha.std(0) + 1e-8
    pm = _maha_fit(_std(Hm, mm, ms)); pa = _maha_fit(_std(Ha, am_, as_))
    zhm = _maha_z(_std(Hm, mm, ms), pm); zha = _maha_z(_std(Ha, am_, as_), pa)

    print("\nWindow-level detection vs derived knock labels (fit on healthy only)")
    print(f"{'ds':<5} {'mic_ROC':>8} {'acc_ROC':>8} {'SUM_ROC':>8} {'SUM_PRAUC':>10} {'+rate':>7}")
    for dn in DS:
        y = Ay[dn]
        if y.sum() == 0 or y.sum() == y.size:
            print(f"{dn:<5} (no usable +/- split: {int(y.sum())}/{y.size})")
            continue
        zm = _maha_z(_std(Am[dn], mm, ms), pm)
        za = _maha_z(_std(Aa[dn], am_, as_), pa)
        zsum = zm + za
        print(f"{dn:<5} {roc_auc_score(y, zm):>8.3f} {roc_auc_score(y, za):>8.3f} "
              f"{roc_auc_score(y, zsum):>8.3f} {average_precision_score(y, zsum):>10.3f} "
              f"{y.mean():>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
