"""Raw-signal evidence figures, computed from the real recordings on disk
(figures 6, 7, 8, 9 of the figure plan).

   6  cross-modal synchronization evidence (envelope cross-correlation)
   7  per-mode acoustic signatures (log-mel + CWT, ST/TU/PU)
   8  vibration feature triplet, healthy vs knock (D4 raw waveform)
   9  acoustic PSD with the ROW II characteristic frequencies marked

Loads recordings through the production ingestion pipeline
(``TestDatasetLoader``) and the production feature extractors, so the
figures show exactly what the model sees.

Run with:  python -m scripts.figures.fig_signals
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    MODE_COLORS,
    VIBRATION,
    save,
)

from src.features.audio_spectral import compute_log_mel_spectrogram
from src.features.acoustic_representations import compute_cwt_scalogram
from src.features.vibration_temporal import compute_vibration_input_stack
from src.ingestion.sync_verification import (
    _decimate_to_rate,
    _hilbert_envelope_mean,
    _normalised_xcorr,
    _vibration_envelope_mean,
    verify_paired_sync,
)
from src.modeling.orchestration.full_run import resolved_loader

# Characteristic ROW II frequencies (Results ch., tab:characteristic_frequencies)
# (frequency, label, annotation row 0/1 — staggered to avoid collisions)
CHARACTERISTIC_HZ = [
    (5.87, "shaft fund.\n5.87 Hz", 0),
    (43.75, "runner-blade\npassing (7x)", 1),
    (50.0, "mains\n50 Hz", 0),
    (100.0, "rotor-pole\npassing (16x)", 1),
    (117.3, "guide-vane\npassing (20x)", 0),
]


def _segments(dataset_yaml: str):
    return resolved_loader(dataset_yaml).list_segments()


# ─────────────────────────────────────────────────────────────────────────
# 6 — synchronization evidence
# ─────────────────────────────────────────────────────────────────────────
def fig06_sync_evidence() -> None:
    # Pick the D2 recording with the clearest envelope correlation peak
    # (the run-log audit reports confidence 4.95 on one D2 recording).
    best, best_seg = None, None
    for seg in _segments("d2.yaml"):
        ds = seg.segment
        try:
            res = verify_paired_sync(
                ds.mic_data, ds.accel_data, ds.mic_sample_rate, ds.accel_sample_rate
            )
        except Exception:
            continue
        if best is None or res.confidence > best.confidence:
            best, best_seg = res, seg
    assert best is not None and best_seg is not None
    ds = best_seg.segment

    # Recompute the envelope pair + full r(k) curve for the plot.
    fv = best.vibration_fs_used
    m_env = _decimate_to_rate(
        _hilbert_envelope_mean(ds.mic_data), ds.mic_sample_rate, fv
    )
    v_env = _vibration_envelope_mean(ds.accel_data)
    n = min(len(m_env), len(v_env))
    m_env, v_env = m_env[:n] - m_env[:n].mean(), v_env[:n] - v_env[:n].mean()
    K = max(1, int(np.floor(best.max_offset_s * fv)))
    r = _normalised_xcorr(m_env, v_env, K)
    lags_s = np.arange(-K, K + 1) / fv

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) cross-correlation
    ax1.plot(lags_s, r, color="0.3", lw=1.2, marker="o", ms=3.5)
    k_peak = int(np.argmax(np.abs(r))) - K
    ax1.axvline(best.offset_s, color=ANOMALY, lw=1.0, ls="--")
    ax1.annotate(
        f"|r| peak at {best.offset_s * 1e3:+.0f} ms\nconfidence {best.confidence:.2f}",
        xy=(best.offset_s, r[k_peak + K]),
        xytext=(0.96, 0.28), textcoords="axes fraction",
        fontsize=7.5, color=ANOMALY, va="top", ha="right",
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5},
        arrowprops={"arrowstyle": "-|>", "color": ANOMALY, "lw": 0.9, "shrinkB": 4},
    )
    ax1.set_xlabel("lag $k/f_v$ (s)", labelpad=4)
    ax1.set_ylabel("normalized cross-correlation $r(k)$")
    ax1.set_title(f"(a) envelope cross-correlation ($f_v = {fv:.0f}$ Hz)", fontsize=8.5)

    # (b) aligned envelopes, zoomed to the strongest JOINT activity — the
    # region the cross-correlation actually keyed on
    t = np.arange(n) / fv
    shift = int(round(best.offset_s * fv))
    m_al = np.roll(m_env, -shift) if shift else m_env
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)  # noqa: E731
    zm, zv = z(m_al), z(v_env)
    joint = np.abs(zm) * np.abs(zv)
    guard = int(2 * fv)
    joint[:guard] = joint[-guard:] = 0  # ignore the roll wrap-around
    i_peak = int(np.argmax(joint))
    half = int(30 * fv)
    span = slice(max(0, i_peak - half), min(n, i_peak + half))
    ax2.plot(t[span], zm[span], color=ACOUSTIC, lw=0.9,
             label="acoustic envelope (decimated, shifted)")
    ax2.plot(t[span], zv[span], color=VIBRATION, lw=0.9, alpha=0.85,
             label="vibration envelope")
    ax2.set_xlabel("time (s)", labelpad=4)
    ax2.set_ylabel("envelope (z-scored)")
    ax2.set_title("(b) the envelope pair being correlated, offset applied", fontsize=8.5)
    ax2.legend(loc="upper right", frameon=False, fontsize=6.8)

    fig.suptitle(
        f"The cross-modal sync check on the D2 recording with the corpus' strongest "
        f"envelope correlation ({best_seg.recording_id})",
        fontsize=9, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig06_sync_evidence")


# ─────────────────────────────────────────────────────────────────────────
# 7 — per-mode acoustic signatures
# ─────────────────────────────────────────────────────────────────────────
def fig07_mode_signatures() -> None:
    wanted = {"Standstill": None, "Turbine": None, "Pump": None}
    for seg in _segments("d2.yaml"):
        if seg.mode_label in wanted and wanted[seg.mode_label] is None and not seg.is_anomaly:
            wanted[seg.mode_label] = seg
    missing = [m for m, s in wanted.items() if s is None]
    if missing:  # fall back to D1 for any missing mode
        for seg in _segments("d1.yaml"):
            if seg.mode_label in missing and wanted[seg.mode_label] is None and not seg.is_anomaly:
                wanted[seg.mode_label] = seg

    slice_s = 12.0
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.8), constrained_layout=True)
    for col, mode in enumerate(["Standstill", "Turbine", "Pump"]):
        seg = wanted[mode]
        ds = seg.segment
        fs = ds.mic_sample_rate
        T = ds.mic_data.shape[1]
        n = min(int(slice_s * fs), T)
        start = max(0, (T - n) // 2)
        x = ds.mic_data[0, start: start + n]

        mel = compute_log_mel_spectrogram(x, fs)
        ax = axes[0, col]
        im0 = ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
                        extent=(0, n / fs, 0, mel.shape[0]),
                        vmin=mel.max() - 80, vmax=mel.max())
        ax.set_title(mode, fontsize=9, color=MODE_COLORS[mode], fontweight="bold")
        if col == 0:
            ax.set_ylabel("log-mel band\n(20 Hz - 8 kHz)", fontsize=7.5)
        ax.set_xticks([])
        ax.grid(False)

        cwt = compute_cwt_scalogram(x, fs)
        ax = axes[1, col]
        im1 = ax.imshow(cwt, origin="lower", aspect="auto", cmap="magma",
                        extent=(0, n / fs, 0, cwt.shape[0]))
        if col == 0:
            ax.set_ylabel("CWT scale\n(20 - 250 Hz, log)", fontsize=7.5)
        ax.set_xlabel("time (s)", fontsize=7.5)
        ax.grid(False)

    fig.colorbar(im0, ax=axes[0, :], shrink=0.85, pad=0.01, label="dB")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.85, pad=0.01, label="log power")
    fig.suptitle(
        "Operating mode is visibly present in the acoustic signal "
        "(one microphone, D2 recordings; encoder input representation)",
        fontsize=9.5,
    )
    save(fig, "fig07_mode_signatures")


# ─────────────────────────────────────────────────────────────────────────
# 8 — vibration triplet, healthy vs knock
# ─────────────────────────────────────────────────────────────────────────
def fig08_vibration_triplet() -> None:
    segs = _segments("d4.yaml")
    healthy = next(s for s in segs if not s.is_anomaly)
    knock = next(s for s in segs if s.is_anomaly)

    win_s = 6.0
    rows = ["amplitude (z)", "Hilbert envelope (z)", "impulsiveness\n(excess kurtosis)"]
    fig, axes = plt.subplots(3, 2, figsize=(6.8, 3.9), sharex="col", constrained_layout=True)

    for col, (seg, title, color) in enumerate(
        [(healthy, "healthy (D4, speed bucket)", VIBRATION),
         (knock, f"knock recording (D4, {knock.source_dir.name})", ANOMALY)]
    ):
        ds = seg.segment
        fs = ds.accel_sample_rate
        stack = compute_vibration_input_stack(ds.accel_data, sample_rate=fs)
        ch = stack[0]  # first accelerometer
        n = int(win_s * fs)
        if col == 0:
            start = max(0, (ch.shape[1] - n) // 2)
        else:  # center the slice on the strongest impulsive event
            start = int(np.clip(np.argmax(ch[2]) - n // 2, 0, ch.shape[1] - n))
        t = np.arange(n) / fs
        for row in range(3):
            ax = axes[row, col]
            ax.plot(t, ch[row, start: start + n], color=color, lw=0.7)
            if row == 0:
                ax.set_title(title, fontsize=8.5, color=color)
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=7.2)
        axes[2, col].set_xlabel("time (s)", fontsize=8)

    # share y-limits per row so the contrast is honest
    for row in range(3):
        lims = [axes[row, c].get_ylim() for c in range(2)]
        lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
        for c in range(2):
            axes[row, c].set_ylim(lo, hi)

    fig.suptitle(
        "What the vibration encoder sees (one accelerometer, raw-waveform D4): "
        "a knock dominates all three channels",
        fontsize=9.5,
    )
    save(fig, "fig08_vibration_triplet")


# ─────────────────────────────────────────────────────────────────────────
# 9 — annotated PSD
# ─────────────────────────────────────────────────────────────────────────
def fig09_psd() -> None:
    seg = next(
        s for s in _segments("d2.yaml") if s.mode_label == "Turbine" and not s.is_anomaly
    )
    ds = seg.segment
    fs = ds.mic_sample_rate
    f, pxx = welch(ds.mic_data, fs=fs, nperseg=1 << 16, axis=-1)
    pxx_db = 10 * np.log10(pxx.mean(axis=0) + 1e-20)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    sel = f <= 150
    ax.plot(f[sel], pxx_db[sel], color="0.25", lw=0.9)
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for fc, label, row in CHARACTERISTIC_HZ:
        ax.axvline(fc, ymax=(0.74 if row else 0.86), color=ACOUSTIC, lw=0.9,
                   ls="--", alpha=0.8)
        y_text = ymin + span * (0.78 if row else 0.90)
        ha = "left" if fc < 12 else "center"
        ax.annotate(label, xy=(fc, y_text), ha=ha, va="bottom",
                    fontsize=6.8, color=ACOUSTIC)
    ax.set_xlim(0, 150)
    ax.set_ylim(ymin, ymin + span * 1.30)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz), mic average")
    ax.set_title(
        "Acoustic PSD of a Turbine-mode D2 recording with the ROW II characteristic\n"
        "frequencies the analysis grid is sized to resolve "
        "($n_{fft}=4096 \\Rightarrow$ 3.9 Hz bins)",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig09_psd_characteristic_freqs")


def main() -> None:
    style.apply_style()
    fig06_sync_evidence()
    fig07_mode_signatures()
    fig08_vibration_triplet()
    fig09_psd()


if __name__ == "__main__":
    main()
