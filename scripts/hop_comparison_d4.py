"""Quick empirical check: acoustic features at hop=512 vs hop=43 on a D4 knock recording.

Loads one mic channel from a D4 RandomFault knock recording, computes log-mel at
both hop lengths over a 2 s window, and reports:
  - frame count, feature_fs, tensor shape
  - basic statistics (mean, std, dynamic range)
  - where peak energy sits in time (proxy for whether the knock onset is resolved)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.features.audio_spectral import compute_log_mel_spectrogram


REPO = Path(__file__).resolve().parents[1]
REC = REPO / "data" / "fourth_test_dataset" / "RandomFault_knock_unter_speed1" / "(2, -4, 8)"
WAV = REC / "recorded_E.wav"  # one of the 9 mics

MIC_SR = 16_000
WINDOW_S = 2.0
N_FFT = 1024
N_MELS = 64
HOPS = {"default_hop_512": 512, "match_vib_hop_43": 43}


def load_mic_window(path: Path, sr: int, window_s: float, start_s: float = 0.0) -> np.ndarray:
    fs, x = wavfile.read(path)
    if fs != sr:
        raise SystemExit(f"unexpected sample rate {fs} in {path}")
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32) / max(1.0, float(np.iinfo(np.int16).max))
    n = int(round(window_s * sr))
    i0 = int(round(start_s * sr))
    return x[i0 : i0 + n]


def find_knock_window(path: Path, sr: int, window_s: float) -> tuple[np.ndarray, float]:
    """Scan the recording for the loudest 2 s window — likely contains a knock."""
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32) / max(1.0, float(np.iinfo(np.int16).max))
    n_win = int(round(window_s * sr))
    # coarse scan: 0.25 s stride
    stride = sr // 4
    best_start = 0
    best_rms = -1.0
    for i in range(0, len(x) - n_win + 1, stride):
        seg = x[i : i + n_win]
        # peakiness proxy: max abs / rms (impulsive content)
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        peak = float(np.max(np.abs(seg)))
        score = peak / rms
        if score > best_rms:
            best_rms = score
            best_start = i
    return x[best_start : best_start + n_win], best_start / sr


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"missing wav: {WAV}")

    print(f"== Recording: {REC.relative_to(REPO)}")
    print(f"== Mic file:  {WAV.name}")
    print(f"== Window:    {WINDOW_S} s @ {MIC_SR} Hz  ->  {int(WINDOW_S*MIC_SR)} samples")
    print()

    # Pick the most impulsive 2 s window — most likely contains a knock.
    sig, start_s = find_knock_window(WAV, MIC_SR, WINDOW_S)
    print(f"== Most-impulsive window starts at t={start_s:.2f}s")
    print(f"   raw waveform: shape={sig.shape}, peak={np.max(np.abs(sig)):.3f}, rms={np.sqrt(np.mean(sig**2)):.4f}")
    print()

    rows = []
    for label, hop in HOPS.items():
        mel = compute_log_mel_spectrogram(
            sig, fs=MIC_SR, n_fft=N_FFT, hop_length=hop, n_mels=N_MELS
        )
        feature_fs = MIC_SR / hop
        n_frames = mel.shape[1]
        # peak-energy frame location (proxy for onset resolution)
        per_frame_energy = mel.sum(axis=0)
        peak_frame = int(np.argmax(per_frame_energy))
        peak_time_ms = 1000.0 * peak_frame / feature_fs  # noqa
        # spread of energy around the peak (5%-95% quantile width in time)
        cum = np.cumsum(per_frame_energy)
        cum = cum / cum[-1]
        i5 = int(np.searchsorted(cum, 0.05))
        i95 = int(np.searchsorted(cum, 0.95))
        spread_ms = 1000.0 * (i95 - i5) / feature_fs
        rows.append(
            dict(
                label=label,
                hop=hop,
                feature_fs_hz=feature_fs,
                frame_ms=1000.0 / feature_fs,
                n_frames=n_frames,
                mel_shape=mel.shape,
                mean=float(mel.mean()),
                std=float(mel.std()),
                dyn_range=float(mel.max() - mel.min()),
                peak_frame=peak_frame,
                peak_time_ms=peak_time_ms,
                energy_5_95_spread_ms=spread_ms,
            )
        )

    print("== Per-hop log-mel comparison (1 mic channel, 2 s window):")
    print()
    for r in rows:
        print(f"  [{r['label']}]  hop={r['hop']}")
        print(f"    feature_fs      = {r['feature_fs_hz']:.2f} Hz  ({r['frame_ms']:.2f} ms / frame)")
        print(f"    n_frames        = {r['n_frames']}")
        print(f"    mel tensor      = {r['mel_shape']}  (n_mels, n_frames)")
        print(f"    mean / std      = {r['mean']:.4f} / {r['std']:.4f}")
        print(f"    dynamic range   = {r['dyn_range']:.4f}")
        print(f"    peak-energy at  = frame {r['peak_frame']}  ~ t = {r['peak_time_ms']:.1f} ms")
        print(f"    5-95% energy spread (temporal) = {r['energy_5_95_spread_ms']:.0f} ms")
        print()

    # Cost ratio
    r_a, r_b = rows[0], rows[1]
    ratio = r_b["n_frames"] / r_a["n_frames"]
    bytes_a = r_a["n_frames"] * r_a["mel_shape"][0] * 4 * 9  # 9 mics, float32
    bytes_b = r_b["n_frames"] * r_b["mel_shape"][0] * 4 * 9
    print("== Cost ratio (per 2 s window, 9 mics, float32 log-mel only):")
    print(f"   frames:  {r_a['n_frames']} -> {r_b['n_frames']}  ({ratio:.1f}x more)")
    print(f"   memory:  {bytes_a/1024:.1f} KB -> {bytes_b/1024:.1f} KB  ({ratio:.1f}x more)")


if __name__ == "__main__":
    main()
