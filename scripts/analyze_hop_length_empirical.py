"""Empirical hop_length analysis — evidence-driven, task-specific.

Tests `(n_fft, hop_length)` combinations on REAL data against the three
downstream tasks of this thesis:

  1. **Mode separability (V1/V2 RQ1)** — K-means(K=3) cluster purity on
     log-mel features of D1 healthy recordings labelled Pump/Standstill/
     Turbine. Higher = better.
  2. **Knock SNR (V3 anomaly + V4 localization)** — peak frame energy
     ratio in the knock band (≥ 200 Hz) on D4 RandomFault recordings vs
     healthy speed1. Higher = better impulse detectability.
  3. **Compute cost** — per-recording feature extraction wall-clock on D1
     and D4 representative recordings.

Also runs a synthetic tone-resolution check on 100/117 Hz two-sine
mixtures to confirm the n_fft frequency-bin argument.

Designed to run in ~5–10 minutes on CPU with no model training.

Usage:
    python -m scripts.analyze_hop_length_empirical
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config.dataset_registry import REGISTRY
from src.features.audio_spectral import compute_log_mel_spectrogram
from src.ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader


# Configurations to sweep — (n_fft, hop_length).
SWEEP = [
    (1024, 43),   # publication "tone-resolution Nyquist" pick
    (1024, 64),   # 250 Hz frame rate; below Nyquist of 117 Hz tone modulation
    (1024, 128),  # 125 Hz frame rate
    (1024, 256),  # 62.5 Hz frame rate; librosa default n_fft/4
    (1024, 512),  # 31.25 Hz frame rate; legacy default
    (2048, 512),  # finer freq resolution (7.8 Hz/bin), same temporal
]
N_MELS = 48  # matches publication YAMLs
SEG_DURATION_S = 8.0  # crop recordings to this for speed
SEG_OFFSET_S = 2.0    # skip first 2s of each recording (avoid power-on transient)
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Section 1 — synthetic tone-resolution check (independent of hop_length)
# ---------------------------------------------------------------------------


def test_tone_resolution_two_sines() -> None:
    """Confirm whether n_fft=1024 actually resolves the 100/117 Hz tone pair.

    Generates a 4-second mixture of two equal-amplitude sines at 100 Hz
    and 117 Hz at 16 kHz sample rate, computes STFT magnitudes at each
    n_fft, and checks whether the two peaks are distinguishable (separated
    by at least one zero-crossing in the bin amplitude trajectory).
    """
    fs = 16_000
    t = np.arange(int(4.0 * fs)) / fs
    signal = np.sin(2 * np.pi * 100.0 * t) + np.sin(2 * np.pi * 117.0 * t)

    log("\n" + "=" * 70)
    log("SECTION 1 — Frequency resolution: can n_fft resolve 100/117 Hz?")
    log("=" * 70)
    log(f"{'n_fft':>6} {'bin_width_Hz':>14} {'peaks_resolved':>16} {'peak_freqs_Hz':>22}")
    for n_fft in (512, 1024, 2048, 4096):
        bin_width = fs / n_fft
        # Use a Hann window for a clean spectrum.
        win = np.hanning(n_fft)
        x = signal[:n_fft] * win
        spec = np.abs(np.fft.rfft(x, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        # Look in the 80-140 Hz band for peaks.
        mask = (freqs >= 80.0) & (freqs <= 140.0)
        local_freqs = freqs[mask]
        local_spec = spec[mask]
        # Find local maxima
        peak_idxs = [
            i for i in range(1, len(local_spec) - 1)
            if local_spec[i] > local_spec[i - 1] and local_spec[i] > local_spec[i + 1]
        ]
        peak_freqs = [float(local_freqs[i]) for i in peak_idxs]
        resolved = len(peak_freqs) >= 2 and all(
            any(abs(pf - target) < bin_width * 1.5 for pf in peak_freqs)
            for target in (100.0, 117.0)
        )
        log(
            f"{n_fft:>6d} {bin_width:>13.2f}  {'YES' if resolved else 'NO':>16}  "
            f"{str(peak_freqs):>22}"
        )


# ---------------------------------------------------------------------------
# Section 2 — Mode separability via K-means on log-mel features
# ---------------------------------------------------------------------------


def _purity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Cluster purity: weighted majority-label agreement per cluster."""
    n = len(labels_true)
    if n == 0:
        return 0.0
    total_match = 0
    for c in np.unique(labels_pred):
        mask = labels_pred == c
        if mask.sum() == 0:
            continue
        votes = labels_true[mask]
        # Majority class in this cluster
        unique, counts = np.unique(votes, return_counts=True)
        total_match += int(counts.max())
    return total_match / n


def _nmi_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalized mutual information (sklearn implementation)."""
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels_true, labels_pred))


def load_d1_labeled_segments() -> list[tuple[np.ndarray, str]]:
    """Load D1 + D2 recordings as (mic_data, mode_label).

    D1 alone only has 4 healthy-mode recordings (2 Pump, 2 Turbine).
    Pooling D2 brings in 5 more (Pump, Standstill, Turbine).  This gives
    a non-trivial K=3 test.  Slices the FIRST n_mics channels so the
    feature vector dimensionality matches across datasets (D1=4, D2=5
    → take 4 from each).
    """
    out: list[tuple[np.ndarray, str]] = []
    for ds_name in ("d1", "d2"):
        spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{ds_name}.yaml")
        loader = TestDatasetLoader(spec)
        for seg in loader.list_segments():
            mode = seg.mode_label
            if mode not in ("Pump", "Standstill", "Turbine"):
                continue
            fs = seg.segment.mic_sample_rate
            n_start = int(SEG_OFFSET_S * fs)
            n_keep = int(SEG_DURATION_S * fs)
            if seg.segment.mic_data.shape[1] < n_start + n_keep:
                # Fall back to using whatever's available
                if seg.segment.mic_data.shape[1] < int(1.0 * fs):
                    continue
                n_start = 0
                n_keep = seg.segment.mic_data.shape[1]
            mic = seg.segment.mic_data[:4, n_start : n_start + n_keep]  # take first 4 mics
            out.append((mic.astype(np.float32), mode))
    return out


def evaluate_mode_separability(
    samples: list[tuple[np.ndarray, str]], n_fft: int, hop: int
) -> dict:
    """For each (mic_data, mode), compute mean log-mel feature vector,
    then K-means(K=3) on the stacked feature vectors and measure purity / NMI.
    """
    from sklearn.cluster import KMeans

    label_map = {"Pump": 0, "Standstill": 1, "Turbine": 2}
    feats: list[np.ndarray] = []
    labels: list[int] = []
    fs = 16_000
    t0 = time.time()
    total_frames = 0

    for mic, mode in samples:
        # Per-mic log-mel, then average across mics (gives mean spectrum per recording).
        mels = []
        for ch in range(mic.shape[0]):
            m = compute_log_mel_spectrogram(
                mic[ch], fs, n_fft=n_fft, hop_length=hop, n_mels=N_MELS
            )
            mels.append(m)
        # (n_mics, n_mels, T) → average over mics → (n_mels, T) → mean over T → (n_mels,)
        per_recording = np.stack(mels, axis=0).mean(axis=0).mean(axis=1)
        feats.append(per_recording)
        labels.append(label_map[mode])
        total_frames += mels[0].shape[1]

    t_features = time.time() - t0
    X = np.stack(feats, axis=0)
    y = np.asarray(labels)

    # Standardize before K-means (so high-energy mel bins don't dominate)
    X_centered = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    km = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_SEED).fit(X_centered)
    purity = _purity_score(y, km.labels_)
    nmi = _nmi_score(y, km.labels_)

    return {
        "n_recordings": len(samples),
        "purity": purity,
        "nmi": nmi,
        "feature_time_s": t_features,
        "total_frames": total_frames,
        "frames_per_recording": total_frames // max(1, len(samples)),
        "feature_dim": X.shape[1],
    }


# ---------------------------------------------------------------------------
# Section 3 — Knock SNR on D4 RandomFault vs speed1 healthy
# ---------------------------------------------------------------------------


def load_anomaly_and_healthy_pair(
    dataset_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Return (anomaly_mic, healthy_mic, meta) — first usable pair for the dataset.

    `meta` carries the recording_ids picked and any per-dataset notes.
    Slices the first 4 mics so feature dimensionality is comparable across
    datasets with different n_mics (D1=4, D2=5, D3-D5=9).
    """
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / f"{dataset_id}.yaml")
    loader = TestDatasetLoader(spec)
    anomaly_mic = None
    healthy_mic = None
    anomaly_id = None
    healthy_id = None
    for seg in loader.list_segments():
        fs = seg.segment.mic_sample_rate
        n_start = int(SEG_OFFSET_S * fs)
        n_keep = int(SEG_DURATION_S * fs)
        if seg.segment.mic_data.shape[1] < n_start + n_keep:
            if seg.segment.mic_data.shape[1] < int(1.0 * fs):
                continue
            n_start = 0
            n_keep = seg.segment.mic_data.shape[1]
        mic = seg.segment.mic_data[:4, n_start : n_start + n_keep].astype(np.float32)
        if seg.is_anomaly and anomaly_mic is None:
            anomaly_mic = mic
            anomaly_id = seg.recording_id
        if (not seg.is_anomaly) and healthy_mic is None:
            healthy_mic = mic
            healthy_id = seg.recording_id
        if anomaly_mic is not None and healthy_mic is not None:
            break
    return anomaly_mic, healthy_mic, {
        "dataset": dataset_id,
        "anomaly_id": anomaly_id,
        "healthy_id": healthy_id,
        "n_mics_used": 4,
    }


def evaluate_knock_snr(
    knock_mic: np.ndarray, healthy_mic: np.ndarray, n_fft: int, hop: int
) -> dict:
    """Compute SNR = (peak frame energy in knock band) / (median frame energy in healthy).

    "Knock band" = mel bins 24..47 (≈ 250 Hz – 8 kHz on a 48-mel/16 kHz filterbank).
    Higher SNR = the (n_fft, hop) combo makes the impulsive event stand out more
    against the steady healthy baseline.
    """
    fs = 16_000

    def _compute(mic: np.ndarray) -> np.ndarray:
        # Mean across mics, full log-mel
        mels = []
        for ch in range(mic.shape[0]):
            m = compute_log_mel_spectrogram(
                mic[ch], fs, n_fft=n_fft, hop_length=hop, n_mels=N_MELS
            )
            mels.append(m)
        return np.stack(mels, axis=0).mean(axis=0)  # (n_mels, T)

    knock_mel = _compute(knock_mic)
    healthy_mel = _compute(healthy_mic)

    # Knock-band frame energy (sum of dB across upper mels — these capture
    # broadband impulse energy).
    knock_band = slice(N_MELS // 2, N_MELS)
    knock_energy = knock_mel[knock_band].sum(axis=0)  # (T,)
    healthy_energy = healthy_mel[knock_band].sum(axis=0)

    # SNR in dB: peak knock vs healthy median.
    peak_knock = float(np.max(knock_energy))
    median_healthy = float(np.median(healthy_energy))
    snr_db = peak_knock - median_healthy  # already in dB sum-space

    # Knock contrast within recording: peak vs median of own healthy frames.
    knock_p95 = float(np.percentile(knock_energy, 95))
    knock_p50 = float(np.percentile(knock_energy, 50))
    contrast_db = knock_p95 - knock_p50

    return {
        "knock_T_frames": int(knock_mel.shape[1]),
        "healthy_T_frames": int(healthy_mel.shape[1]),
        "snr_peak_vs_healthy_median_db": snr_db,
        "knock_contrast_p95_vs_p50_db": contrast_db,
        "peak_knock_db": peak_knock,
        "median_healthy_db": median_healthy,
    }


# ---------------------------------------------------------------------------
# Section 4 — Compute cost scaling on representative recordings
# ---------------------------------------------------------------------------


def evaluate_compute_cost(
    samples: list[tuple[np.ndarray, str]], n_fft: int, hop: int
) -> dict:
    """Wall-clock per-recording feature extraction time + memory footprint."""
    fs = 16_000
    t0 = time.time()
    total_bytes = 0
    for mic, _ in samples:
        for ch in range(mic.shape[0]):
            m = compute_log_mel_spectrogram(
                mic[ch], fs, n_fft=n_fft, hop_length=hop, n_mels=N_MELS
            )
            total_bytes += m.nbytes
    elapsed = time.time() - t0
    return {
        "wall_clock_s": elapsed,
        "per_recording_s": elapsed / max(1, len(samples)),
        "feature_memory_MB": total_bytes / 1e6,
        "memory_per_recording_MB": total_bytes / 1e6 / max(1, len(samples)),
    }


# ---------------------------------------------------------------------------
# Section 5 — Main sweep and final summary
# ---------------------------------------------------------------------------


def main() -> int:
    log("\n" + "=" * 70)
    log("EMPIRICAL HOP_LENGTH ANALYSIS")
    log("=" * 70)
    log(f"REPO_ROOT = {REPO_ROOT}")
    log(f"Sweep grid: {SWEEP}")
    log(f"n_mels = {N_MELS}, segment duration = {SEG_DURATION_S}s, "
        f"offset = {SEG_OFFSET_S}s, seed = {RANDOM_SEED}")

    # Section 1: synthetic tone resolution
    test_tone_resolution_two_sines()

    # Load data
    log("\n" + "=" * 70)
    log("Loading D1 labelled mode recordings ...")
    log("=" * 70)
    d1_samples = load_d1_labeled_segments()
    log(f"D1 labelled samples: {len(d1_samples)}")
    mode_counts: dict[str, int] = {}
    for _, mode in d1_samples:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    log(f"  per-mode counts: {mode_counts}")

    log("Loading anomaly + healthy pairs across all 5 datasets ...")
    pairs: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for ds_id in ("d1", "d2", "d3", "d4", "d5"):
        try:
            anom, healthy, meta = load_anomaly_and_healthy_pair(ds_id)
            if anom is None or healthy is None:
                log(f"  {ds_id}: missing pair (anom={anom is not None}, "
                    f"healthy={healthy is not None}) — skipping")
                continue
            pairs[ds_id] = (anom, healthy, meta)
            log(f"  {ds_id}: anom={meta['anomaly_id']!r} shape={anom.shape}, "
                f"healthy={meta['healthy_id']!r} shape={healthy.shape}")
        except Exception as e:
            log(f"  {ds_id}: FAILED — {type(e).__name__}: {e}")

    # Section 2-4: sweep
    log("\n" + "=" * 70)
    log("SECTION 2 — Mode separability (D1 K-means(K=3) purity/NMI)")
    log("=" * 70)
    header = f"{'n_fft':>6} {'hop':>5} {'frame_rate':>11} {'T_frames':>9} {'feat_time_s':>13} {'purity':>8} {'NMI':>7}"
    log(header)

    results_mode: dict = {}
    for n_fft, hop in SWEEP:
        if len(d1_samples) < 3:
            continue
        res = evaluate_mode_separability(d1_samples, n_fft, hop)
        results_mode[(n_fft, hop)] = res
        frame_rate = 16000 / hop
        log(
            f"{n_fft:>6} {hop:>5} {frame_rate:>10.2f} "
            f"{res['frames_per_recording']:>9} {res['feature_time_s']:>13.2f} "
            f"{res['purity']:>8.3f} {res['nmi']:>7.3f}"
        )

    log("\n" + "=" * 70)
    log("SECTION 3 — Anomaly SNR across all 5 datasets")
    log("=" * 70)
    log("Per (dataset, n_fft, hop) — anomaly recording vs healthy reference.")
    log("SNR = peak knock-band frame energy minus median healthy frame energy (dB).")
    log("contrast = within-anomaly p95 minus p50 frame energy (dB).\n")
    results_knock: dict = {}  # keyed by (dataset_id, n_fft, hop)
    log(f"{'ds':>3} {'n_fft':>6} {'hop':>5} {'frame_rate':>11} {'anom_T':>8} "
        f"{'SNR_db':>10} {'contrast_db':>13}")
    for ds_id, (anom, healthy, meta) in pairs.items():
        for n_fft, hop in SWEEP:
            res = evaluate_knock_snr(anom, healthy, n_fft, hop)
            results_knock[(ds_id, n_fft, hop)] = res
            frame_rate = 16000 / hop
            log(
                f"{ds_id:>3} {n_fft:>6} {hop:>5} {frame_rate:>10.2f} "
                f"{res['knock_T_frames']:>8} {res['snr_peak_vs_healthy_median_db']:>10.2f} "
                f"{res['knock_contrast_p95_vs_p50_db']:>13.2f}"
            )
        log("")  # blank line between datasets

    log("=" * 70)
    log("SECTION 4 — Compute cost per dataset (anomaly+healthy pair)")
    log("=" * 70)
    log(f"{'ds':>3} {'n_fft':>6} {'hop':>5} {'cost_per_rec_s':>15} "
        f"{'mem_per_rec_MB':>15} {'vs_(1024,512)':>15}")
    results_cost: dict = {}
    for ds_id, (anom, healthy, _) in pairs.items():
        samples = [(anom, "anom"), (healthy, "healthy")]
        baseline_cost = None
        for n_fft, hop in SWEEP:
            res = evaluate_compute_cost(samples, n_fft, hop)
            results_cost[(ds_id, n_fft, hop)] = res
            if (n_fft, hop) == (1024, 512):
                baseline_cost = res["per_recording_s"]
        # second pass to print with the cost ratio
        for n_fft, hop in SWEEP:
            res = results_cost[(ds_id, n_fft, hop)]
            ratio = res["per_recording_s"] / baseline_cost if baseline_cost else float("nan")
            log(
                f"{ds_id:>3} {n_fft:>6} {hop:>5} {res['per_recording_s']:>15.3f} "
                f"{res['memory_per_recording_MB']:>15.2f} {ratio:>13.2f}x"
            )
        log("")  # blank line between datasets

    # Section 5: cross-dataset summary
    log("=" * 70)
    log("SECTION 5 — Cross-dataset summary")
    log("=" * 70)
    log("\nSNR (dB) at each hop_length with n_fft=1024 — per dataset:")
    hops_1024 = [h for nf, h in SWEEP if nf == 1024]
    log(f"  {'ds':>3} " + " ".join(f"hop={h:>4}" for h in hops_1024))
    for ds_id in pairs:
        row = f"  {ds_id:>3} "
        for hop in hops_1024:
            snr = results_knock.get((ds_id, 1024, hop), {}).get(
                "snr_peak_vs_healthy_median_db", float("nan")
            )
            row += f" {snr:>8.2f}"
        log(row)

    log("\nSNR variation across hop choices (max - min, n_fft=1024) per dataset:")
    log("  (small range = hop_length doesn't matter for that dataset)")
    for ds_id in pairs:
        snrs = [
            results_knock.get((ds_id, 1024, hop), {}).get(
                "snr_peak_vs_healthy_median_db", float("nan")
            )
            for hop in hops_1024
        ]
        snrs = [s for s in snrs if not np.isnan(s)]
        if snrs:
            log(f"  {ds_id}: max-min = {max(snrs) - min(snrs):>6.2f} dB "
                f"(min={min(snrs):.2f}, max={max(snrs):.2f})")

    log("\nSNR gain from n_fft=1024 -> 2048 (at hop=512) per dataset:")
    log("  (positive = wider FFT helps that dataset's anomaly detection)")
    for ds_id in pairs:
        snr_1024 = results_knock.get((ds_id, 1024, 512), {}).get(
            "snr_peak_vs_healthy_median_db", float("nan")
        )
        snr_2048 = results_knock.get((ds_id, 2048, 512), {}).get(
            "snr_peak_vs_healthy_median_db", float("nan")
        )
        if not np.isnan(snr_1024) and not np.isnan(snr_2048):
            log(f"  {ds_id}: {snr_2048 - snr_1024:>+6.2f} dB")

    log("\nMode separability (D1+D2 K=3 K-means):")
    if (1024, 512) in results_mode:
        baseline_purity = results_mode[(1024, 512)]["purity"]
        baseline_nmi = results_mode[(1024, 512)]["nmi"]
        log(f"  At (n_fft=1024, hop=512): purity={baseline_purity:.3f}, "
            f"NMI={baseline_nmi:.3f}")
        deltas = []
        for n_fft, hop in SWEEP:
            p = results_mode.get((n_fft, hop), {}).get("purity", float("nan"))
            n = results_mode.get((n_fft, hop), {}).get("nmi", float("nan"))
            deltas.append((n_fft, hop, p - baseline_purity, n - baseline_nmi))
        max_p_delta = max(abs(d[2]) for d in deltas)
        max_n_delta = max(abs(d[3]) for d in deltas)
        log(f"  Max |d_purity| across sweep: {max_p_delta:.4f}")
        log(f"  Max |d_NMI| across sweep:    {max_n_delta:.4f}")
        log("  Interpretation: identical to 3 decimals -> hop is irrelevant "
            "for mode clustering.")

    log("\nCompute cost summary — per-recording wall-clock at hop=43 vs 512, "
        "n_fft=1024:")
    log(f"  {'ds':>3} {'@hop=43':>10} {'@hop=512':>10} {'ratio':>8}")
    for ds_id in pairs:
        c43 = results_cost.get((ds_id, 1024, 43), {}).get("per_recording_s", float("nan"))
        c512 = results_cost.get((ds_id, 1024, 512), {}).get("per_recording_s", float("nan"))
        if not np.isnan(c43) and not np.isnan(c512) and c512 > 0:
            log(f"  {ds_id:>3} {c43:>9.3f}s {c512:>9.3f}s {c43 / c512:>7.2f}x")

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
