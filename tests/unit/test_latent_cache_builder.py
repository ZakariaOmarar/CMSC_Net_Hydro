from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.io import wavfile


torch = pytest.importorskip("torch", exc_type=ImportError)

from src.modeling.latent.builder import build_latent_cache


def _write_wav(
    path, *, sr: int = 16_000, duration_s: float = 1.0, freq_hz: float = 220.0
) -> None:
    t = np.arange(int(sr * duration_s)) / sr
    x = (0.2 * np.sin(2 * np.pi * freq_hz * t) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, x)


def _write_vibration_csv(
    path, *, rows: int = 12, amplitude_base: float = 100.0
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["esp_time_us", "amplitude", "frequency"]
        )
        writer.writeheader()
        for i in range(rows):
            writer.writerow(
                {
                    "esp_time_us": 1_000_000 + i * 250_000,
                    "amplitude": amplitude_base + i,
                    "frequency": 125.0,
                }
            )


def _build_flat_dataset(root) -> None:
    for idx, sensor in enumerate(["B", "C", "D", "E"]):
        _write_wav(root / f"recorded_{sensor}_Pump.wav", freq_hz=220.0 + idx)
        _write_vibration_csv(
            root / f"vibration_{sensor}_Pump.csv", amplitude_base=120.0 + idx
        )


def test_build_latent_cache_creates_npz(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _build_flat_dataset(data_root)

    output_dir = tmp_path / "latents"
    summaries = build_latent_cache(
        data_root=data_root,
        output_dir=output_dir,
        mode="mic_vibration",
        acoustic_rep="mfcc",
        window_s=0.5,
        overlap=0.0,
        context_len=4,
        n_mfcc=8,
        n_fft=256,
        hop_length=64,
        device="cpu",
    )

    assert summaries
    assert all(item.output_path.exists() for item in summaries)

    with np.load(summaries[0].output_path, allow_pickle=False) as blob:
        assert "z" in blob
        assert "c" in blob
        assert "recording_id" in blob
        assert "is_transition_window" in blob

        z = np.asarray(blob["z"])
        c = np.asarray(blob["c"])
        assert z.ndim == 2
        assert c.ndim == 2
        assert z.shape[0] == c.shape[0]
        assert z.shape[0] > 0


def test_build_latent_cache_reuses_existing_files(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _build_flat_dataset(data_root)

    output_dir = tmp_path / "latents"
    first = build_latent_cache(
        data_root=data_root,
        output_dir=output_dir,
        mode="mic_vibration",
        acoustic_rep="mfcc",
        window_s=0.5,
        overlap=0.0,
        context_len=4,
        n_mfcc=8,
        n_fft=256,
        hop_length=64,
        device="cpu",
        reuse_existing=True,
        log_progress=False,
    )
    assert first

    target = first[0].output_path
    mtime_before = target.stat().st_mtime_ns

    second = build_latent_cache(
        data_root=data_root,
        output_dir=output_dir,
        mode="mic_vibration",
        acoustic_rep="mfcc",
        window_s=0.5,
        overlap=0.0,
        context_len=4,
        n_mfcc=8,
        n_fft=256,
        hop_length=64,
        device="cpu",
        reuse_existing=True,
        log_progress=False,
    )
    assert second

    mtime_after = target.stat().st_mtime_ns
    assert mtime_after == mtime_before
