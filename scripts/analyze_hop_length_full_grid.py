"""Full-grid hop_length / n_fft / n_mels sweep on ALL 5 datasets.

Tests every reasonable combination of acoustic feature parameters
against three operational criteria:

  1. **Anomaly AUC** — ROC AUC of per-frame knock-band log-mel energy as a
     binary classifier for anomaly vs healthy.  This is the metric that
     actually predicts V3 / V4 downstream behaviour, more directly than
     dB SNR sums.
  2. **Mode separability** — K-means(K=3) purity / NMI on D1+D2 healthy
     recordings (the V1/V2 RQ1 evaluation criterion).
  3. **Compute cost** — per-recording wall-clock and memory.

Aggregates over MULTIPLE knock recordings per dataset (where available)
to reduce single-sample noise.  Writes a JSON dump of all results plus
a final per-criterion ranking.

Run:
    python -m scripts.analyze_hop_length_full_grid
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.audio_spectral import compute_log_mel_spectrogram
from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader


# Full sweep — every n_fft x hop combo where hop <= n_fft.
# Hops are chosen to hit the impulse-localization regime (knock < 10 ms = 160
# samples at 16 kHz so hop <= 160 means every knock lands in >= 1 frame),
# the convention range (256 / 512), and the very coarse end (1024 / 2048).
N_FFT_GRID = [512, 1024, 2048, 4096]
HOP_GRID = [32, 43, 64, 80, 128, 160, 256, 512, 1024, 2048, 4096]
N_MELS_GRID = [32, 48, 64, 96]
MAX_KNOCK_RECORDINGS_PER_DS = 3   # average over up to this many anomalies per dataset
SEG_DURATION_S = 8.0
SEG_OFFSET_S = 2.0
N_MICS_USED = 4
RANDOM_SEED = 42

OUTPUT_JSON = REPO_ROOT / "results" / "hop_grid_sweep.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def _valid_combos() -> list[tuple[int, int, int]]:
    """Enumerate (n_fft, hop, n_mels) with hop <= n_fft."""
    combos: list[tuple[int, int, int]] = []
    for n_fft in N_FFT_GRID:
        for hop in HOP_GRID:
            if hop > n_fft:
                continue
            for n_mels in N_MELS_GRID:
                combos.append((n_fft, hop, n_mels))
    return combos


def load_dataset_pairs(
    dataset_id: str, max_knock: int = MAX_KNOCK_RECORDINGS_PER_DS
) -> tuple[list[np.ndarray], np.ndarray | None, dict]:
    """Return (list of anomaly mic clips, healthy reference clip, meta).

    Each clip is the first N_MICS_USED channels cropped to SEG_DURATION_S
    starting at SEG_OFFSET_S.  Anomaly clips capped at max_knock.
    """
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{dataset_id}.yaml")
    loader = TestDatasetLoader(spec)
    anom_clips: list[np.ndarray] = []
    anom_ids: list[str] = []
    healthy: np.ndarray | None = None
    healthy_id: str | None = None
    for seg in loader.list_segments():
        fs = seg.segment.mic_sample_rate
        n_start = int(SEG_OFFSET_S * fs)
        n_keep = int(SEG_DURATION_S * fs)
        if seg.segment.mic_data.shape[1] < n_start + n_keep:
            if seg.segment.mic_data.shape[1] < int(1.0 * fs):
                continue
            n_start = 0
            n_keep = seg.segment.mic_data.shape[1]
        mic = seg.segment.mic_data[:N_MICS_USED, n_start : n_start + n_keep].astype(np.float32)
        if seg.is_anomaly:
            if len(anom_clips) < max_knock:
                anom_clips.append(mic)
                anom_ids.append(seg.recording_id)
        elif healthy is None:
            healthy = mic
            healthy_id = seg.recording_id
    return anom_clips, healthy, {
        "dataset": dataset_id,
        "anomaly_ids": anom_ids,
        "healthy_id": healthy_id,
        "n_anom_clips": len(anom_clips),
    }


def _per_mic_log_mel(
    mic_data: np.ndarray, fs: int, n_fft: int, hop: int, n_mels: int
) -> np.ndarray:
    """Compute (n_mels, T) log-mel averaged across mics for one clip."""
    mels = []
    for ch in range(mic_data.shape[0]):
        try:
            m = compute_log_mel_spectrogram(
                mic_data[ch], fs, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            )
        except ValueError:
            return np.zeros((n_mels, 1), dtype=np.float32)
        mels.append(m)
    return np.stack(mels, axis=0).mean(axis=0)


def _knock_auc(
    anom_clips: list[np.ndarray],
    healthy: np.ndarray,
    n_fft: int,
    hop: int,
    n_mels: int,
    fs: int = 16_000,
) -> float:
    """Per-frame ROC AUC of knock-band energy as anomaly vs healthy classifier.

    Concatenates per-frame band energies from all anomaly clips (positives)
    against healthy frames (negatives).  Knock band = upper half of n_mels.
    """
    from sklearn.metrics import roc_auc_score

    band = slice(n_mels // 2, n_mels)
    pos_scores: list[float] = []
    for clip in anom_clips:
        mel = _per_mic_log_mel(clip, fs, n_fft, hop, n_mels)
        pos_scores.extend(mel[band].sum(axis=0).tolist())
    neg_mel = _per_mic_log_mel(healthy, fs, n_fft, hop, n_mels)
    neg_scores = neg_mel[band].sum(axis=0).tolist()
    if not pos_scores or not neg_scores:
        return float("nan")
    y = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores)),
    ])
    s = np.concatenate([np.asarray(pos_scores), np.asarray(neg_scores)])
    return float(roc_auc_score(y, s))


def _mode_purity_nmi(
    d1d2_samples: list[tuple[np.ndarray, str]],
    n_fft: int,
    hop: int,
    n_mels: int,
    fs: int = 16_000,
) -> tuple[float, float]:
    """K-means(K=3) on per-recording mean log-mel features."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    if len(d1d2_samples) < 3:
        return float("nan"), float("nan")
    label_map = {"Pump": 0, "Standstill": 1, "Turbine": 2}
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for mic, mode in d1d2_samples:
        mel = _per_mic_log_mel(mic, fs, n_fft, hop, n_mels)
        feats.append(mel.mean(axis=1))
        labels.append(label_map[mode])
    X = np.stack(feats, axis=0)
    y = np.asarray(labels)
    X_c = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    km = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_SEED).fit(X_c)
    # purity
    total = 0
    for c in np.unique(km.labels_):
        mask = km.labels_ == c
        if mask.sum() == 0:
            continue
        _, counts = np.unique(y[mask], return_counts=True)
        total += int(counts.max())
    purity = total / len(y)
    nmi = float(normalized_mutual_info_score(y, km.labels_))
    return purity, nmi


def _compute_cost(
    clip: np.ndarray, n_fft: int, hop: int, n_mels: int, fs: int = 16_000
) -> tuple[float, float]:
    """(wall_clock_seconds, feature_memory_MB) for one clip."""
    t0 = time.time()
    total_bytes = 0
    for ch in range(clip.shape[0]):
        try:
            m = compute_log_mel_spectrogram(
                clip[ch], fs, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            )
            total_bytes += m.nbytes
        except ValueError:
            return float("nan"), float("nan")
    return time.time() - t0, total_bytes / 1e6


def load_mode_samples() -> list[tuple[np.ndarray, str]]:
    """D1 + D2 healthy recordings with Pump/Standstill/Turbine labels."""
    out: list[tuple[np.ndarray, str]] = []
    for ds_id in ("d1", "d2"):
        spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{ds_id}.yaml")
        loader = TestDatasetLoader(spec)
        for seg in loader.list_segments():
            if seg.mode_label not in ("Pump", "Standstill", "Turbine"):
                continue
            fs = seg.segment.mic_sample_rate
            n_start = int(SEG_OFFSET_S * fs)
            n_keep = int(SEG_DURATION_S * fs)
            if seg.segment.mic_data.shape[1] < n_start + n_keep:
                if seg.segment.mic_data.shape[1] < int(1.0 * fs):
                    continue
                n_start = 0
                n_keep = seg.segment.mic_data.shape[1]
            mic = seg.segment.mic_data[:N_MICS_USED, n_start : n_start + n_keep]
            out.append((mic.astype(np.float32), seg.mode_label))
    return out


def main() -> int:
    log("=" * 78)
    log("FULL GRID SWEEP — (n_fft, hop, n_mels) across all 5 datasets")
    log("=" * 78)
    combos = _valid_combos()
    log(f"n_fft grid     : {N_FFT_GRID}")
    log(f"hop_length grid: {HOP_GRID}")
    log(f"n_mels grid    : {N_MELS_GRID}")
    log(f"Total valid combos: {len(combos)}")
    log(f"Datasets       : d1, d2, d3, d4, d5")
    log(f"Anomaly clips/ds (max): {MAX_KNOCK_RECORDINGS_PER_DS}")
    log("")

    # Load data once.
    log("Loading data ...")
    t0 = time.time()
    pairs: dict[str, tuple[list[np.ndarray], np.ndarray, dict]] = {}
    for ds_id in ("d1", "d2", "d3", "d4", "d5"):
        anom_clips, healthy, meta = load_dataset_pairs(ds_id)
        if not anom_clips or healthy is None:
            log(f"  {ds_id}: SKIP — missing pair")
            continue
        pairs[ds_id] = (anom_clips, healthy, meta)
        log(f"  {ds_id}: {meta['n_anom_clips']} anomaly clips, 1 healthy ref "
            f"(ids: anom={meta['anomaly_ids']}, healthy={meta['healthy_id']!r})")
    mode_samples = load_mode_samples()
    log(f"  mode samples (D1+D2 labelled): {len(mode_samples)}")
    log(f"Load time: {time.time() - t0:.1f}s")

    # Pre-pick one mic clip per dataset for compute-cost measurement.
    cost_clips = {ds: clips[0] for ds, (clips, _, _) in pairs.items()}

    # Sweep.
    log("")
    log("Sweeping ...")
    results: list[dict] = []
    n_done = 0
    t_sweep_start = time.time()
    for n_fft, hop, n_mels in combos:
        row: dict = {"n_fft": n_fft, "hop": hop, "n_mels": n_mels}
        # Anomaly AUC per dataset
        for ds_id, (anom_clips, healthy, _) in pairs.items():
            row[f"auc_{ds_id}"] = _knock_auc(anom_clips, healthy, n_fft, hop, n_mels)
        # Mode separability (D1+D2 pooled)
        purity, nmi = _mode_purity_nmi(mode_samples, n_fft, hop, n_mels)
        row["mode_purity"] = purity
        row["mode_nmi"] = nmi
        # Compute cost: average across the 5 dataset clips
        cost_s_total = 0.0
        mem_MB_total = 0.0
        n_cost = 0
        for ds_id, clip in cost_clips.items():
            s, m = _compute_cost(clip, n_fft, hop, n_mels)
            if not np.isnan(s):
                cost_s_total += s
                mem_MB_total += m
                n_cost += 1
        row["cost_s_avg"] = cost_s_total / max(1, n_cost)
        row["mem_MB_avg"] = mem_MB_total / max(1, n_cost)
        results.append(row)
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t_sweep_start
            rate = n_done / elapsed
            eta = (len(combos) - n_done) / rate
            log(f"  [{n_done:>3}/{len(combos)}] eta={eta / 60:.1f}min")

    log(f"\nSweep complete in {(time.time() - t_sweep_start) / 60:.1f} min "
        f"({len(combos)} combos)")

    # Write JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "grid": {
                    "n_fft": N_FFT_GRID,
                    "hop_length": HOP_GRID,
                    "n_mels": N_MELS_GRID,
                },
                "datasets": list(pairs.keys()),
                "n_combos": len(combos),
                "results": results,
            },
            fh,
            indent=2,
        )
    log(f"Wrote {OUTPUT_JSON.relative_to(REPO_ROOT)}")

    # Final ranking summary.
    log("\n" + "=" * 78)
    log("RANKING — top 10 by mean anomaly AUC across all 5 datasets")
    log("=" * 78)
    for r in results:
        aucs = [r[f"auc_{ds}"] for ds in pairs if f"auc_{ds}" in r and not np.isnan(r[f"auc_{ds}"])]
        r["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
        r["min_auc"] = float(np.min(aucs)) if aucs else float("nan")
    sorted_by_auc = sorted(
        [r for r in results if not np.isnan(r["mean_auc"])],
        key=lambda r: -r["mean_auc"],
    )
    header = (f"{'rank':>4} {'n_fft':>6} {'hop':>5} {'n_mels':>6} "
              f"{'mean_auc':>10} {'min_auc':>9} "
              f"{'auc_d1':>7} {'auc_d2':>7} {'auc_d3':>7} {'auc_d4':>7} {'auc_d5':>7} "
              f"{'cost_s':>8} {'mem_MB':>8}")
    log(header)
    for i, r in enumerate(sorted_by_auc[:15], 1):
        log(
            f"{i:>4} {r['n_fft']:>6} {r['hop']:>5} {r['n_mels']:>6} "
            f"{r['mean_auc']:>10.4f} {r['min_auc']:>9.4f} "
            f"{r.get('auc_d1', float('nan')):>7.3f} "
            f"{r.get('auc_d2', float('nan')):>7.3f} "
            f"{r.get('auc_d3', float('nan')):>7.3f} "
            f"{r.get('auc_d4', float('nan')):>7.3f} "
            f"{r.get('auc_d5', float('nan')):>7.3f} "
            f"{r['cost_s_avg']:>8.3f} {r['mem_MB_avg']:>8.2f}"
        )

    # Pareto: AUC vs cost
    log("\n" + "=" * 78)
    log("PARETO — configs that are not dominated on (mean_auc, cost)")
    log("=" * 78)
    valid = [r for r in results if not np.isnan(r["mean_auc"])]
    pareto = []
    for r in valid:
        dominated = False
        for s in valid:
            if (
                s["mean_auc"] >= r["mean_auc"]
                and s["cost_s_avg"] <= r["cost_s_avg"]
                and (s["mean_auc"] > r["mean_auc"] or s["cost_s_avg"] < r["cost_s_avg"])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["cost_s_avg"])
    log(header)
    for i, r in enumerate(pareto, 1):
        log(
            f"{i:>4} {r['n_fft']:>6} {r['hop']:>5} {r['n_mels']:>6} "
            f"{r['mean_auc']:>10.4f} {r['min_auc']:>9.4f} "
            f"{r.get('auc_d1', float('nan')):>7.3f} "
            f"{r.get('auc_d2', float('nan')):>7.3f} "
            f"{r.get('auc_d3', float('nan')):>7.3f} "
            f"{r.get('auc_d4', float('nan')):>7.3f} "
            f"{r.get('auc_d5', float('nan')):>7.3f} "
            f"{r['cost_s_avg']:>8.3f} {r['mem_MB_avg']:>8.2f}"
        )

    log("\n" + "=" * 78)
    log("MODE separability check (D1+D2 pooled, K=3 K-means)")
    log("=" * 78)
    unique_purity = sorted({(round(r["mode_purity"], 3), round(r["mode_nmi"], 3)) for r in valid})
    log(f"Unique (purity, NMI) pairs across all {len(valid)} configs: {len(unique_purity)}")
    for p, n in unique_purity:
        log(f"  purity={p:.3f}, NMI={n:.3f}")
    log("If only one pair appears -> mode clustering is COMPLETELY invariant to "
        "(n_fft, hop, n_mels) on the available labelled data.")

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
