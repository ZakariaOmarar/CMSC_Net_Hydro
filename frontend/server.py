"""FastAPI backend for the bench-top second-dataset monitoring dashboard.

Serves the static frontend files and exposes a REST API backed by the
pre-computed results in results/second/fault_positions/.

Usage:
    cd <repo-root>
    .venv\\Scripts\\python.exe -m uvicorn frontend.server:app --reload --port 8050

Then open http://localhost:8050 in your browser.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = _REPO_ROOT / "results" / "second"
_FAULT_POSITIONS_ROOT = _RESULTS_ROOT / "fault_positions"
_DATA_ROOT = _REPO_ROOT / "data" / "second_test_dataset" / "RandomFault"
_FRONTEND_DIR = Path(__file__).resolve().parent

# Mic channel name → index (order used by compute_gcc_stack_s2_multiwindow)
_MIC_NAMES = ["D", "E", "F", "G", "I"]  # recorded_D.wav … recorded_I.wav
_VIB_NAMES = ["A", "B", "C", "D", "E"]  # vibration_A.csv … vibration_E.csv

# ---------------------------------------------------------------------------
# Sensor geometry (from localization_head.py, in metres)
# ---------------------------------------------------------------------------
_S2_MIC_XYZ_CM = {
    "mic_D": [0.0, 41.0, 15.0],
    "mic_E": [0.0, 31.0, 16.0],
    "mic_F": [10.0, 0.0, 24.0],
    "mic_G": [15.5, 5.0, 15.0],
    "mic_I": [0.0, 10.0, 15.0],
}
_S2_VIB_XYZ_CM = {
    "vibration_A": [10.0, 0.0, 23.0],
    "vibration_B": [15.5, 6.0, 15.0],
    "vibration_C": [0.0, 17.0, 12.0],
    "vibration_D": [0.0, 40.0, 15.0],
    "vibration_E": [15.5, 30.0, 16.0],
}
_S2_FAULT_POSITIONS_CM = {
    "pos_(10,0,23)": [10.0, 0.0, 23.0],
    "pos_(15,6,15)": [15.0, 6.0, 15.0],
    "pos_(0,17,12)": [0.0, 17.0, 12.0],
    "pos_(0,40,15)": [0.0, 40.0, 15.0],
    "pos_(15,30,15)": [15.0, 30.0, 15.0],
}

# Healthy (nominal) operating mode recordings — no fault position info
_HEALTHY_CATEGORIES: dict[str, dict] = {
    "Pump": {
        "state_code": "PU",
        "data_dir": _REPO_ROOT / "data" / "second_test_dataset" / "Pump",
    },
    "Turbine": {
        "state_code": "TU",
        "data_dir": _REPO_ROOT / "data" / "second_test_dataset" / "Turbine",
    },
    "Standstill": {
        "state_code": "ST",
        "data_dir": _REPO_ROOT / "data" / "second_test_dataset" / "Standstill",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_wav_duration_s(
    data_dir: Path, preferred_name: str = "recorded_D.wav"
) -> float:
    """Return the actual duration of the first readable WAV file in *data_dir*.

    Falls back to 60.0 s if no WAV file can be read (directory absent, no wavs).
    """
    candidates = [data_dir / preferred_name] + sorted(data_dir.glob("recorded_*.wav"))
    for path in candidates:
        if path.exists():
            try:
                with wave.open(str(path)) as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                continue
    return 60.0


# ---------------------------------------------------------------------------
# Healthy segment helpers
# ---------------------------------------------------------------------------


def _is_healthy_seg(seg_id: str) -> bool:
    return seg_id.startswith("healthy_")


def _get_data_dir(seg_id: str) -> Path | None:
    """Return the raw-data directory for either a healthy or a fault-position segment."""
    if _is_healthy_seg(seg_id):
        cat = seg_id[len("healthy_") :]
        info = _HEALTHY_CATEGORIES.get(cat)
        return info["data_dir"] if info else None
    else:
        folder = _seg_id_to_folder(seg_id)  # results/second/fault_positions/{pos}
        return (_DATA_ROOT / folder.name) if folder else None


def _get_seg_state_code(seg_id: str) -> str:
    if _is_healthy_seg(seg_id):
        cat = seg_id[len("healthy_") :]
        return _HEALTHY_CATEGORIES.get(cat, {}).get("state_code", "ST")
    return "RF"


def _list_healthy_segments() -> list[dict[str, Any]]:
    segs = []
    for cat, info in sorted(_HEALTHY_CATEGORIES.items()):
        if info["data_dir"].exists():
            segs.append(
                {
                    "id": f"healthy_{cat}",
                    "folder": cat,
                    "index": len(segs),
                    "ground_truth_cm": None,
                    "operating_modes": [info["state_code"]],
                    "best_method": None,
                    "best_error_cm": None,
                    "duration_s": _get_wav_duration_s(info["data_dir"]),
                    "state_code": info["state_code"],
                    "is_healthy": True,
                }
            )
    return segs


def _list_all_segments() -> list[dict[str, Any]]:
    """Return healthy segments first, then fault-position segments."""
    healthy = _list_healthy_segments()
    faults = [{**f, "is_healthy": False} for f in _list_fault_positions()]
    return healthy + faults


# ---------------------------------------------------------------------------
# Fault-position ID helpers
# ---------------------------------------------------------------------------


def _folder_to_seg_id(folder_name: str) -> str:
    """Convert a fault position folder name to a URL-safe segment ID.

    Example: 'pos_(0,17,12)_turbine_pump' → 'pos_0_17_12_turbine_pump'
    """
    return re.sub(r"[(),]", "_", folder_name).replace("__", "_").strip("_")


def _seg_id_to_folder(seg_id: str) -> Path | None:
    """Resolve a segment ID back to a fault position folder path."""
    for d in sorted(_FAULT_POSITIONS_ROOT.iterdir()):
        if d.is_dir() and _folder_to_seg_id(d.name) == seg_id:
            return d
    return None


def _list_fault_positions() -> list[dict[str, Any]]:
    """Return a sorted list of fault-position metadata dicts."""
    positions = []
    for i, d in enumerate(sorted(_FAULT_POSITIONS_ROOT.iterdir())):
        if not d.is_dir():
            continue
        loc_path = d / "localization.json"
        loc = json.loads(loc_path.read_text()) if loc_path.exists() else {}
        positions.append(
            {
                "id": _folder_to_seg_id(d.name),
                "folder": d.name,
                "index": i,
                "ground_truth_cm": loc.get("ground_truth_cm"),
                "operating_modes": loc.get("operating_modes", []),
                "best_method": loc.get("best_method"),
                "best_error_cm": loc.get("best_error_cm"),
                "duration_s": _get_wav_duration_s(_DATA_ROOT / d.name),
                "state_code": "RF",  # RandomFault
            }
        )
    return positions


def _load_localization(folder: Path) -> dict[str, Any]:
    p = folder / "localization.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_cnf_infer(folder: Path) -> dict[str, Any]:
    p = folder / "cnf_infer.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _read_wav_mono(
    path: Path, channel: int = 0, max_samples: int = 80000
) -> tuple[np.ndarray, int]:
    """Read one channel from a WAV file, capped at max_samples."""
    with wave.open(str(path), "rb") as wf:
        fs = wf.getframerate()
        n_channels = wf.getnchannels()
        sampw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype = np.int16 if sampw == 2 else np.int32
    samples = np.frombuffer(raw, dtype=dtype).reshape(-1, n_channels)
    ch_data = samples[:, min(channel, n_channels - 1)].astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.iinfo(dtype).max
    ch_data /= float(peak)

    # Downsample if needed
    if len(ch_data) > max_samples:
        step = len(ch_data) // max_samples
        ch_data = ch_data[::step][:max_samples]
        fs_out = max(fs // step, 1)  # guard against division yielding 0
    else:
        fs_out = fs

    return ch_data, fs_out


import csv as _csv_mod  # noqa: E402 — placed here to keep imports near usage


def _read_vib_csv(path: Path) -> dict[str, np.ndarray]:
    """Parse a vibration CSV (pc_time, esp_time_us, amplitude, frequency).

    Returns a dict with keys 'amplitude', 'frequency', 'times_s' as float32/float64
    arrays. Returns empty arrays if the file cannot be parsed.
    """
    amps: list[float] = []
    freqs: list[float] = []
    times_us: list[int] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = _csv_mod.DictReader(fh)
        for row in reader:
            try:
                amps.append(float(row["amplitude"]))
                freqs.append(float(row["frequency"]))
                times_us.append(int(row["esp_time_us"]))
            except (KeyError, ValueError):
                continue
    if not amps:
        empty = np.array([], dtype=np.float32)
        return {
            "amplitude": empty,
            "frequency": empty,
            "times_s": empty.astype(np.float64),
        }
    amp_arr = np.asarray(amps, dtype=np.float32)
    freq_arr = np.asarray(freqs, dtype=np.float32)
    t_us = np.asarray(times_us, dtype=np.int64)
    times_s = (t_us - t_us[0]).astype(np.float64) / 1e6
    return {"amplitude": amp_arr, "frequency": freq_arr, "times_s": times_s}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Bench-Top Localization Dashboard")


# --- Static files -----------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


# --- Health -----------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def config():
    return {
        "dataset": "second_test_dataset",
        "n_mics": 5,
        "n_accels": 5,
        "fs_hz": 16000,
        "box_cm": [41, 41, 40],
    }


# --- Overview ---------------------------------------------------------------
@app.get("/api/overview")
async def overview():
    all_segs = _list_all_segments()
    faults = [s for s in all_segs if not s["is_healthy"]]
    errors = [p["best_error_cm"] for p in faults if p["best_error_cm"] is not None]

    return {
        "health": 100,
        "current_state": "RF",
        "n_segments": len(all_segs),
        "n_anomalies": len(faults),
        "n_healthy": len(all_segs) - len(faults),
        "n_transitions": 0,
        "mean_localization_error_cm": float(np.mean(errors)) if errors else None,
        "best_localization_error_cm": float(min(errors)) if errors else None,
    }


# --- Timeline ---------------------------------------------------------------
@app.get("/api/timeline")
async def timeline():
    all_segs = _list_all_segments()
    offset = 0.0
    states = []
    anomalies = []

    for seg in all_segs:
        dur = seg["duration_s"]
        states.append(
            {
                "segment_id": seg["id"],
                "state_code": seg["state_code"],
                "offset_s": offset,
                "duration_s": dur,
            }
        )
        # Only fault positions get anomaly markers
        if not seg["is_healthy"]:
            anomalies.append(
                {
                    "segment_id": seg["id"],
                    "offset_s": offset + dur * 0.5,
                    "score": (100.0 - (seg["best_error_cm"] or 50.0)) / 100.0,
                }
            )
        offset += dur

    return {"states": states, "anomalies": anomalies, "transitions": []}


# --- Segments ---------------------------------------------------------------
@app.get("/api/segments")
async def list_segments():
    all_segs = _list_all_segments()
    return [
        {
            "id": p["id"],
            "state_code": p["state_code"],
            "duration_s": p["duration_s"],
            "folder": p["folder"],
            "ground_truth_cm": p.get("ground_truth_cm"),
            "operating_modes": p.get("operating_modes", []),
            "is_healthy": p.get("is_healthy", False),
        }
        for p in all_segs
    ]


@app.get("/api/segments/{seg_id}/waveform")
async def segment_waveform(
    seg_id: str,
    sensor: str = Query("mic", pattern="^(mic|accel)$"),
    channel: int = Query(0, ge=0),
    max_points: int = Query(5000, ge=100, le=100000),
):  # waveform
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Data directory not found for {seg_id}"
        )

    if sensor == "mic":
        names = _MIC_NAMES
        wav_paths = [data_dir / f"recorded_{n}.wav" for n in names]
        wav_path = wav_paths[min(channel, len(wav_paths) - 1)]
        if not wav_path.exists():
            raise HTTPException(
                status_code=404, detail=f"WAV file not found: {wav_path.name}"
            )
        samples, fs = _read_wav_mono(wav_path, channel=0, max_samples=max_points)
        times = (np.arange(len(samples)) / fs).tolist()
        return {
            "times": times,
            "values": samples.tolist(),
            "sample_rate": fs,
            "channel": channel,
            "sensor": "mic",
            "channel_name": names[min(channel, len(names) - 1)],
        }
    else:
        # Vibration: CSV has columns pc_time, esp_time_us, amplitude, frequency
        names = _VIB_NAMES
        csv_name = f"vibration_{names[min(channel, len(names) - 1)]}.csv"
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found: {csv_name}")
        vdata = _read_vib_csv(csv_path)
        return {
            "times": vdata["times_s"].tolist(),
            "values": vdata["amplitude"].tolist(),
            "sample_rate": 4,  # ~4 measurements/s
            "channel": channel,
            "sensor": "accel",
            "channel_name": names[min(channel, len(names) - 1)],
        }


@app.get("/api/segments/{seg_id}/spectrum")
async def segment_spectrum(
    seg_id: str,
    sensor: str = Query("mic", pattern="^(mic|accel)$"),
    channel: int = Query(0, ge=0),
    max_points: int = Query(4096, ge=64, le=65536),
):
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")

    if sensor == "mic":
        wav_path = data_dir / f"recorded_{_MIC_NAMES[min(channel, 4)]}.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="WAV file not found")
        samples, fs = _read_wav_mono(wav_path, channel=0, max_samples=48000)
    else:
        csv_path = data_dir / f"vibration_{_VIB_NAMES[min(channel, 4)]}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV not found")
        vdata = _read_vib_csv(csv_path)
        samples = vdata["amplitude"]
        fs = 4  # ~4 Hz measurement rate

    n_fft = min(max_points, len(samples))
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(samples[:n_fft] * window))
    mag = 20 * np.log10(mag + 1e-12)  # dBFS

    return {"freqs": freqs.tolist(), "magnitude": mag.tolist()}


@app.get("/api/segments/{seg_id}/spectrogram")
async def segment_spectrogram(
    seg_id: str,
    sensor: str = Query("mic", pattern="^(mic|accel)$"),
    channel: int = Query(0, ge=0),
):
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")

    if sensor == "mic":
        wav_path = data_dir / f"recorded_{_MIC_NAMES[min(channel, 4)]}.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="WAV file not found")
        samples, fs = _read_wav_mono(wav_path, channel=0, max_samples=48000)
    else:
        csv_path = data_dir / f"vibration_{_VIB_NAMES[min(channel, 4)]}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV not found")
        vdata = _read_vib_csv(csv_path)
        samples = vdata["amplitude"]
        fs = 4  # ~4 Hz measurement rate

    n_fft = 512
    hop = n_fft // 4
    n_frames = (len(samples) - n_fft) // hop
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs).tolist()
    times = []
    power = []

    for i in range(min(n_frames, 256)):
        frame = samples[i * hop : i * hop + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))
        power.append((20 * np.log10(mag + 1e-12)).tolist())
        times.append(float(i * hop / fs))

    return {"times": times, "freqs": freqs, "power": power}


@app.get("/api/segments/{seg_id}/all-waveforms")
async def segment_all_waveforms(
    seg_id: str,
    sensor: str = Query("mic", pattern="^(mic|accel)$"),
    max_points: int = Query(3000, ge=100, le=10000),
):
    """Return all channels for the given sensor type in a single response."""
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")

    channels = []
    if sensor == "mic":
        for name in _MIC_NAMES:
            path = data_dir / f"recorded_{name}.wav"
            if not path.exists():
                continue
            samples, fs = _read_wav_mono(path, channel=0, max_samples=max_points)
            t = (np.arange(len(samples)) / fs).tolist()
            channels.append(
                {"name": f"Mic {name}", "times": t, "values": samples.tolist()}
            )
    else:
        for name in _VIB_NAMES:
            path = data_dir / f"vibration_{name}.csv"
            if not path.exists():
                continue
            vdata = _read_vib_csv(path)
            channels.append(
                {
                    "name": f"Vib {name}",
                    "times": vdata["times_s"].tolist(),
                    "values": vdata["amplitude"].tolist(),
                }
            )

    return {"channels": channels, "sensor": sensor}


@app.get("/api/segments/{seg_id}/vib-profiles")
async def segment_vib_profiles(seg_id: str):
    """Return vibration amplitude + dominant frequency per sensor over time."""
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")

    channels = []
    for name in _VIB_NAMES:
        path = data_dir / f"vibration_{name}.csv"
        if not path.exists():
            continue
        vdata = _read_vib_csv(path)
        channels.append(
            {
                "name": f"Vib {name}",
                "times_s": vdata["times_s"].tolist(),
                "amplitude": vdata["amplitude"].tolist(),
                "frequency": vdata["frequency"].tolist(),
            }
        )
    return {"channels": channels}


@app.get("/api/segments/{seg_id}/signal-stats")
async def segment_signal_stats(seg_id: str):
    """Per-channel signal statistics for mic (RMS, crest, kurtosis, dom. freq)
    and vibration sensors (amplitude stats, mean frequency)."""
    data_dir = _get_data_dir(seg_id)
    if data_dir is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Data directory not found")

    mic_stats = []
    for name in _MIC_NAMES:
        path = data_dir / f"recorded_{name}.wav"
        if not path.exists():
            continue
        samples, fs = _read_wav_mono(path, channel=0, max_samples=48000)
        rms = float(np.sqrt(np.mean(samples**2)))
        peak = float(np.max(np.abs(samples)))
        crest = float(peak / (rms + 1e-12))
        mean = np.mean(samples)
        std = float(np.std(samples)) + 1e-12
        kurt = float(np.mean(((samples - mean) / std) ** 4))
        n = min(len(samples), 8192)
        w = np.hanning(n)
        mag = np.abs(np.fft.rfft(samples[:n] * w))
        dom_hz = float(np.fft.rfftfreq(n, 1.0 / fs)[np.argmax(mag)])
        mic_stats.append(
            {
                "name": f"Mic {name}",
                "rms_db": round(float(20 * np.log10(rms + 1e-12)), 1),
                "peak_db": round(float(20 * np.log10(peak + 1e-12)), 1),
                "crest_factor": round(crest, 2),
                "kurtosis": round(kurt, 2),
                "dominant_hz": round(dom_hz, 1),
            }
        )

    vib_stats = []
    for name in _VIB_NAMES:
        path = data_dir / f"vibration_{name}.csv"
        if not path.exists():
            continue
        vdata = _read_vib_csv(path)
        amp, freq = vdata["amplitude"], vdata["frequency"]
        vib_stats.append(
            {
                "name": f"Vib {name}",
                "mean_amplitude": round(float(np.mean(amp)), 1),
                "max_amplitude": round(float(np.max(amp)), 1),
                "std_amplitude": round(float(np.std(amp)), 1),
                "mean_freq_hz": round(float(np.mean(freq)), 2),
            }
        )

    return {"mic": mic_stats, "vib": vib_stats}


@app.get("/api/segments/{seg_id}/features")
async def segment_features(seg_id: str):
    # Healthy segments have no localization/CNF results
    if _is_healthy_seg(seg_id):
        return {"state": _get_seg_state_code(seg_id), "is_healthy": 1.0}

    folder = _seg_id_to_folder(seg_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")

    loc = _load_localization(folder)
    cnf = _load_cnf_infer(folder)

    features: dict[str, float] = {}

    if loc:
        gt = loc.get("ground_truth_cm", [])
        features["gt_x_cm"] = float(gt[0]) if len(gt) > 0 else 0.0
        features["gt_y_cm"] = float(gt[1]) if len(gt) > 1 else 0.0
        features["gt_z_cm"] = float(gt[2]) if len(gt) > 2 else 0.0

        for method_key in ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused"):
            m = loc.get(method_key, {})
            if m and m.get("error_cm") is not None:
                features[f"error_{method_key}_cm"] = float(m["error_cm"])
                est = m.get("estimated_cm", [])
                if len(est) >= 3:
                    features[f"{method_key}_x_cm"] = float(est[0])
                    features[f"{method_key}_y_cm"] = float(est[1])
                    features[f"{method_key}_z_cm"] = float(est[2])

    if cnf:
        events = cnf.get("anomaly_events", [])
        if events:
            scores = [e["score"] for e in events if "score" in e]
            features["cnf_mean_score"] = float(np.mean(scores))
            features["cnf_max_score"] = float(np.max(scores))
            features["cnf_n_anomalies"] = float(
                len([s for s in scores if s > cnf.get("threshold_default", 0)])
            )

    return features


@app.get("/api/segments/{seg_id}/detection")
async def segment_detection(seg_id: str):
    # Healthy recordings have no fault → return a clean "no anomaly" response
    if _is_healthy_seg(seg_id):
        state_code = _get_seg_state_code(seg_id)
        return {
            "segment_id": seg_id,
            "state_code": state_code,
            "state_confidence": 1.0,
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "detector_votes": {},
            "trained": True,
            "localization": {},
            "direction": None,
            "direction_confidence": 0.0,
            "region": {},
        }

    folder = _seg_id_to_folder(seg_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")

    loc = _load_localization(folder)
    cnf = _load_cnf_infer(folder)

    # Anomaly detection from CNF
    n_anom = cnf.get("n_anomalies", 0)
    n_windows = cnf.get("n_windows", 1)
    anomaly_score = float(n_anom) / max(float(n_windows), 1.0)
    is_anomaly = n_anom > 0

    # Detector votes (all models that have results)
    votes: dict[str, bool] = {}
    for model_file in [
        "cnf_infer.json",
        "ocsvm_anomaly_infer.json",
        "cnn_ae_anomaly_infer.json",
        "lstm_ae_anomaly_infer.json",
    ]:
        p = folder / model_file
        if p.exists():
            d = json.loads(p.read_text())
            label = model_file.replace("_infer.json", "")
            votes[label] = d.get("n_anomalies", 0) > 0

    # Build localization sub-object for frontend
    localization: dict[str, Any] = {}
    if loc:
        gt = loc.get("ground_truth_cm", [])
        methods = {}

        for key in ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused"):
            m = loc.get(key, {})
            if m and m.get("estimated_cm"):
                methods[key] = {
                    "estimated_cm": m["estimated_cm"],
                    "error_cm": m.get("error_cm"),
                    "method": m.get("method", key),
                }

        localization = {
            "ground_truth_cm": gt,
            "methods": methods,
            "best_method": loc.get("best_method"),
            "best_error_cm": loc.get("best_error_cm"),
            "mic_positions_cm": [v for v in _S2_MIC_XYZ_CM.values()],
            "mic_names": list(_S2_MIC_XYZ_CM.keys()),
            "vib_positions_cm": [v for v in _S2_VIB_XYZ_CM.values()],
            "vib_names": list(_S2_VIB_XYZ_CM.keys()),
            "all_fault_positions_cm": _S2_FAULT_POSITIONS_CM,
            "operating_modes": loc.get("operating_modes", []),
        }

    return {
        "segment_id": seg_id,
        "state_code": "RF",
        "state_confidence": 1.0,
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "detector_votes": votes,
        "trained": True,
        "localization": localization,
        # Legacy fields kept for frontend compatibility (not meaningful here)
        "direction": None,
        "direction_confidence": 0.0,
        "region": {},
    }


# --- Machine diagram --------------------------------------------------------
@app.get("/api/machine-diagram")
async def machine_diagram(seg_id: str = Query("")):
    """Return bench-top sensor layout for XZ cross-section rendering."""
    folder = None
    loc: dict[str, Any] = {}
    if seg_id and not _is_healthy_seg(seg_id):
        folder = _seg_id_to_folder(seg_id)
        loc = _load_localization(folder) if folder else {}

    gt = loc.get("ground_truth_cm")
    best = loc.get("best_method")
    best_est = None
    if best and best in loc:
        best_est = loc[best].get("estimated_cm")

    return {
        "type": "bench_top",
        "box_cm": [41, 41, 40],
        "mic_positions_cm": {k: v for k, v in _S2_MIC_XYZ_CM.items()},
        "vib_positions_cm": {k: v for k, v in _S2_VIB_XYZ_CM.items()},
        "known_fault_positions_cm": _S2_FAULT_POSITIONS_CM,
        "ground_truth_cm": gt,
        "best_estimate_cm": best_est,
        "best_method": best,
    }


# --- Anomaly region (re-purposed for 3D position scatter) ------------------
@app.get("/api/anomaly-region")
async def anomaly_region(seg_id: str = Query("")):
    """Return per-method localization estimates for 3D scatter rendering."""
    # Healthy segments have no fault location
    if not seg_id or _is_healthy_seg(seg_id):
        return {
            "methods": {},
            "ground_truth_cm": None,
            "mic_positions_cm": {k: v for k, v in _S2_MIC_XYZ_CM.items()},
            "vib_positions_cm": {k: v for k, v in _S2_VIB_XYZ_CM.items()},
            "all_fault_positions_cm": _S2_FAULT_POSITIONS_CM,
            "best_method": None,
            "best_error_cm": None,
        }

    folder = _seg_id_to_folder(seg_id)
    if folder is None:
        return {"methods": {}, "ground_truth_cm": None}

    loc = _load_localization(folder)
    methods = {}
    for key in ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused"):
        m = loc.get(key, {})
        if m and m.get("estimated_cm"):
            methods[key] = {
                "estimated_cm": m["estimated_cm"],
                "error_cm": m.get("error_cm"),
                "method": m.get("method", key),
            }

    return {
        "methods": methods,
        "ground_truth_cm": loc.get("ground_truth_cm"),
        "mic_positions_cm": {k: v for k, v in _S2_MIC_XYZ_CM.items()},
        "vib_positions_cm": {k: v for k, v in _S2_VIB_XYZ_CM.items()},
        "all_fault_positions_cm": _S2_FAULT_POSITIONS_CM,
        "best_method": loc.get("best_method"),
        "best_error_cm": loc.get("best_error_cm"),
    }


# --- Alerts -----------------------------------------------------------------
@app.get("/api/alerts")
async def alerts():
    positions = _list_fault_positions()
    alert_list = []
    for pos in positions:
        err = pos["best_error_cm"]
        if err is not None:
            level = "INFO" if err < 5 else ("WARNING" if err < 20 else "CRITICAL")
            alert_list.append(
                {
                    "level": level,
                    "time": pos["folder"],
                    "source": pos.get("best_method", "localization"),
                    "message": (
                        f"{pos['folder']}: {pos['best_method']} error = "
                        f"{err:.2f} cm  (GT: {pos['ground_truth_cm']})"
                    ),
                }
            )
    return alert_list


# --- File browser -----------------------------------------------------------
@app.get("/api/browse")
async def browse(path: str = Query("~")):
    import os

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        resolved = _REPO_ROOT

    entries = []
    try:
        for entry in sorted(resolved.iterdir()):
            entries.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "type": "dir" if entry.is_dir() else "file",
                }
            )
    except PermissionError:
        pass

    parent = str(resolved.parent) if resolved.parent != resolved else None
    return {"current": str(resolved), "parent": parent, "entries": entries}


# --- Stub endpoints (not used but called by the frontend) ------------------
@app.post("/api/ingest")
async def ingest_stub():
    return {"status": "done", "n_segments": len(_list_fault_positions())}


@app.get("/api/ingest/progress")
async def ingest_progress():
    return {"status": "idle"}


@app.post("/api/train")
async def train_stub():
    return {
        "status": "trained",
        "n_segments": len(_list_fault_positions()),
        "message": "Pre-computed models from results/second/ are used directly.",
    }


@app.get("/api/train/status")
async def train_status():
    return {
        "status": "trained",
        "is_trained": True,
        "n_segments": len(_list_fault_positions()),
    }


@app.post("/api/detect-all")
async def detect_all():
    return {
        "n_segments": len(_list_fault_positions()),
        "n_anomalies": len(_list_fault_positions()),
    }


# --- Models status ----------------------------------------------------------
@app.get("/api/models-status")
async def models_status():
    """Return training artefact status for each detection / localization model."""
    models: dict[str, dict] = {
        "cnf": {
            "display": "Normalizing Flow (CNF)",
            "artifact": _RESULTS_ROOT / "cnf" / "anomaly" / "flow.pt",
        },
        "ocsvm": {
            "display": "OC-SVM (ν=0.05)",
            "artifact": _RESULTS_ROOT / "ocsvm" / "anomaly" / "anomaly_model.pkl",
        },
        "ocsvm_nu_001": {
            "display": "OC-SVM (ν=0.01)",
            "artifact": _RESULTS_ROOT
            / "ocsvm"
            / "anomaly_nu_001"
            / "anomaly_model.pkl",
        },
        "ocsvm_nu_003": {
            "display": "OC-SVM (ν=0.03)",
            "artifact": _RESULTS_ROOT
            / "ocsvm"
            / "anomaly_nu_003"
            / "anomaly_model.pkl",
        },
        "ocsvm_nu_01": {
            "display": "OC-SVM (ν=0.1)",
            "artifact": _RESULTS_ROOT / "ocsvm" / "anomaly_nu_01" / "anomaly_model.pkl",
        },
        "cnn_ae": {
            "display": "CNN Autoencoder",
            "artifact": _RESULTS_ROOT / "cnn_ae" / "anomaly" / "anomaly_model.pkl",
        },
        "lstm_ae": {
            "display": "LSTM Autoencoder",
            "artifact": _RESULTS_ROOT / "lstm_ae" / "anomaly" / "anomaly_model.pkl",
        },
        "localization_cnn": {
            "display": "Localization CNN (S2)",
            "artifact": _RESULTS_ROOT / "localization_cnn_s2.pt",
        },
    }
    result: dict[str, dict] = {}
    for key, info in models.items():
        p: Path = info["artifact"]
        result[key] = {
            "display": info["display"],
            "trained": p.exists(),
            "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else None,
        }
    return result


# --- Reports ----------------------------------------------------------------
@app.get("/api/reports")
async def reports():
    """Aggregate per-method localization errors for all fault positions."""
    positions = _list_fault_positions()
    rows = []
    for pos in positions:
        folder = _seg_id_to_folder(pos["id"])
        if folder is None:
            continue
        loc = _load_localization(folder)
        gt = loc.get("ground_truth_cm")
        row: dict[str, Any] = {
            "id": pos["id"],
            "folder": pos["folder"],
            "ground_truth_cm": gt,
            "best_method": loc.get("best_method"),
            "best_error_cm": loc.get("best_error_cm"),
            "methods": {},
        }
        for method in ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused"):
            m = loc.get(method, {})
            if m and m.get("estimated_cm"):
                row["methods"][method] = {
                    "estimated_cm": m.get("estimated_cm"),
                    "error_cm": m.get("error_cm"),
                }
        rows.append(row)
    return {"rows": rows}
