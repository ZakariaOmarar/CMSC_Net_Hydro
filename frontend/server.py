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
# S3 (third dataset) paths, geometry and categories
# ---------------------------------------------------------------------------
_S3_RESULTS_ROOT = _REPO_ROOT / "results" / "third"
_S3_FAULT_POSITIONS_ROOT = _S3_RESULTS_ROOT / "fault_positions"
_S3_DATA_ROOT = _REPO_ROOT / "data" / "third_test_dataset"

_S3_MIC_NAMES = ["D_l", "D_r", "E", "F_l", "F_r", "G_l", "G_r", "J_l", "J_r"]
_S3_VIB_NAMES = ["D", "E", "F", "J"]

_S3_MIC_XYZ_CM: dict[str, list[float]] = {
    "mic_Dl": [6.0, 5.0, 1.0],
    "mic_Dr": [11.0, 1.0, 1.0],
    "mic_E": [9.0, -4.0, 1.0],
    "mic_Fl": [6.0, -5.0, 8.0],
    "mic_Fr": [0.0, 0.0, 1.0],
    "mic_Gl": [4.0, -5.0, 1.0],
    "mic_Gr": [11.0, 0.0, 8.0],
    "mic_Jl": [6.0, 5.0, 8.0],
    "mic_Jr": [0.0, 0.0, 8.0],
}
_S3_VIB_XYZ_CM: dict[str, list[float]] = {
    "vibration_D": [6.0, -5.0, 3.0],
    "vibration_E": [11.0, 0.0, 3.0],
    "vibration_F": [0.0, 0.0, 3.0],
    "vibration_J": [6.0, 5.0, 3.0],
}

_S3_HEALTHY_CATEGORIES: dict[str, dict] = {
    "speed1": {
        "state_code": "SP1",
        "data_dir": _S3_DATA_ROOT / "speed1",
    },
    "speed2": {
        "state_code": "SP2",
        "data_dir": _S3_DATA_ROOT / "speed2",
    },
    "speed3": {
        "state_code": "SP3",
        "data_dir": _S3_DATA_ROOT / "speed3",
    },
}
_S3_NORMAL_MODES: frozenset[str] = frozenset({"speed1", "speed2", "speed3"})

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
    return seg_id.startswith("healthy_") or seg_id.startswith("s3_healthy_")


def _get_data_dir(seg_id: str) -> Path | None:
    """Return the raw-data directory for either a healthy or a fault-position segment."""
    # S3 healthy
    if _is_s3_healthy(seg_id):
        cat = seg_id[len("s3_healthy_") :]
        info = _S3_HEALTHY_CATEGORIES.get(cat)
        return info["data_dir"] if info else None
    # S3 fault
    if _is_s3_seg(seg_id):
        folder = _s3_seg_id_to_folder(seg_id)
        return (_S3_DATA_ROOT / folder.name) if folder else None
    # S2 healthy
    if _is_healthy_seg(seg_id):
        cat = seg_id[len("healthy_") :]
        info = _HEALTHY_CATEGORIES.get(cat)
        return info["data_dir"] if info else None
    # S2 fault
    folder = _seg_id_to_folder(seg_id)
    return (_DATA_ROOT / folder.name) if folder else None


def _get_seg_state_code(seg_id: str) -> str:
    if _is_s3_healthy(seg_id):
        cat = seg_id[len("s3_healthy_") :]
        return _S3_HEALTHY_CATEGORIES.get(cat, {}).get("state_code", "SP1")
    if _is_s3_seg(seg_id):
        return "RF"
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
    """Return S2 healthy, S2 faults, S3 healthy, S3 faults — in that order."""
    s2_healthy = [{"dataset": "second", **s} for s in _list_healthy_segments()]
    s2_faults = [
        {"dataset": "second", "is_healthy": False, **f} for f in _list_fault_positions()
    ]
    s3_healthy = _list_s3_healthy_segments()
    s3_faults = [{**f, "is_healthy": False} for f in _list_s3_fault_positions()]
    return s2_healthy + s2_faults + s3_healthy + s3_faults


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _is_s3_seg(seg_id: str) -> bool:
    return seg_id.startswith("s3_")


def _is_s3_healthy(seg_id: str) -> bool:
    return seg_id.startswith("s3_healthy_")


def _s3_folder_to_seg_id(folder_name: str) -> str:
    """Convert an S3 fault folder name to a URL-safe segment ID (prefixed 's3_')."""
    return "s3_" + re.sub(r"[(),]", "_", folder_name).replace("__", "_").strip("_")


def _s3_seg_id_to_folder(seg_id: str) -> Path | None:
    """Resolve an S3 segment ID back to its fault-position folder path."""
    if not _S3_FAULT_POSITIONS_ROOT.exists():
        return None
    for d in sorted(_S3_FAULT_POSITIONS_ROOT.iterdir()):
        if d.is_dir() and _s3_folder_to_seg_id(d.name) == seg_id:
            return d
    return None


def _normalize_s3_loc(loc: dict[str, Any]) -> dict[str, Any]:
    """Map S3 localization JSON fields to the S2-compatible frontend contract.

    * approx_gt_cm          → ground_truth_cm
    * best_approx_error_cm  → best_error_cm
    * best_method "tdoa"    → "tdoa_triangulation"
    * per-method approx_error_cm → error_cm  (alias added, original kept)
    * neural_cnn_s3         → also exposed as neural_cnn (compat key)
    """
    out = dict(loc)
    if "approx_gt_cm" in out and "ground_truth_cm" not in out:
        out["ground_truth_cm"] = out["approx_gt_cm"]
    if "best_approx_error_cm" in out and "best_error_cm" not in out:
        out["best_error_cm"] = out["best_approx_error_cm"]
    bm = out.get("best_method")
    if bm == "tdoa":
        out["best_method"] = "tdoa_triangulation"
    for mkey in (
        "srp_phat",
        "tdoa_triangulation",
        "neural_cnn_s3",
        "neural_cnn_s2_zeroshot",
        "fused",
    ):
        m = out.get(mkey)
        if isinstance(m, dict) and "approx_error_cm" in m and "error_cm" not in m:
            out[mkey] = {**m, "error_cm": m["approx_error_cm"]}
    # Expose S3 neural model under the legacy key so existing frontend code renders it
    if "neural_cnn_s3" in out and "neural_cnn" not in out:
        out["neural_cnn"] = out["neural_cnn_s3"]
    return out


def _list_s3_fault_positions() -> list[dict[str, Any]]:
    """Return sorted list of S3 fault-position metadata dicts."""
    if not _S3_FAULT_POSITIONS_ROOT.exists():
        return []
    positions = []
    for i, d in enumerate(sorted(_S3_FAULT_POSITIONS_ROOT.iterdir())):
        if not d.is_dir():
            continue
        loc_path = d / "localization.json"
        raw_loc = json.loads(loc_path.read_text()) if loc_path.exists() else {}
        loc = _normalize_s3_loc(raw_loc) if raw_loc else {}
        positions.append(
            {
                "id": _s3_folder_to_seg_id(d.name),
                "folder": d.name,
                "index": i,
                "ground_truth_cm": loc.get("ground_truth_cm"),
                "operating_modes": loc.get("operating_modes", []),
                "best_method": loc.get("best_method"),
                "best_error_cm": loc.get("best_error_cm"),
                "duration_s": _get_wav_duration_s(_S3_DATA_ROOT / d.name),
                "state_code": "RF",
                "dataset": "third",
            }
        )
    return positions


def _list_s3_healthy_segments() -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for cat, info in sorted(_S3_HEALTHY_CATEGORIES.items()):
        if info["data_dir"].exists():
            segs.append(
                {
                    "id": f"s3_healthy_{cat}",
                    "folder": cat,
                    "index": len(segs),
                    "ground_truth_cm": None,
                    "operating_modes": [info["state_code"]],
                    "best_method": None,
                    "best_error_cm": None,
                    "duration_s": _get_wav_duration_s(info["data_dir"]),
                    "state_code": info["state_code"],
                    "is_healthy": True,
                    "dataset": "third",
                }
            )
    return segs


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
    index_path = _FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")
    # Inject mtime-based cache-busting version so browsers always load the
    # latest app.js and style.css after code changes.
    for asset in ("app.js", "style.css"):
        asset_path = _FRONTEND_DIR / asset
        v = int(asset_path.stat().st_mtime) if asset_path.exists() else 0
        content = content.replace(f'"/static/{asset}"', f'"/static/{asset}?v={v}"')
    return HTMLResponse(content)


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
        "datasets": [
            {
                "id": "second",
                "label": "S2 — Bench-top (5-mic/5-vib)",
                "n_mics": 5,
                "n_accels": 5,
            },
            {
                "id": "third",
                "label": "S3 — Bench-top (9-mic/4-vib)",
                "n_mics": 9,
                "n_accels": 4,
            },
        ],
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
            "dataset": p.get("dataset", "second"),
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

    mic_names = _S3_MIC_NAMES if _is_s3_seg(seg_id) else _MIC_NAMES
    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES

    if sensor == "mic":
        names = mic_names
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
        names = vib_names
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

    mic_names = _S3_MIC_NAMES if _is_s3_seg(seg_id) else _MIC_NAMES
    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES

    if sensor == "mic":
        wav_path = (
            data_dir / f"recorded_{mic_names[min(channel, len(mic_names) - 1)]}.wav"
        )
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="WAV file not found")
        samples, fs = _read_wav_mono(wav_path, channel=0, max_samples=48000)
    else:
        csv_path = (
            data_dir / f"vibration_{vib_names[min(channel, len(vib_names) - 1)]}.csv"
        )
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

    mic_names = _S3_MIC_NAMES if _is_s3_seg(seg_id) else _MIC_NAMES
    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES

    if sensor == "mic":
        wav_path = (
            data_dir / f"recorded_{mic_names[min(channel, len(mic_names) - 1)]}.wav"
        )
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="WAV file not found")
        samples, fs = _read_wav_mono(wav_path, channel=0, max_samples=48000)
    else:
        csv_path = (
            data_dir / f"vibration_{vib_names[min(channel, len(vib_names) - 1)]}.csv"
        )
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

    mic_names = _S3_MIC_NAMES if _is_s3_seg(seg_id) else _MIC_NAMES
    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES
    channels = []
    if sensor == "mic":
        for name in mic_names:
            path = data_dir / f"recorded_{name}.wav"
            if not path.exists():
                continue
            samples, fs = _read_wav_mono(path, channel=0, max_samples=max_points)
            t = (np.arange(len(samples)) / fs).tolist()
            channels.append(
                {"name": f"Mic {name}", "times": t, "values": samples.tolist()}
            )
    else:
        for name in vib_names:
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

    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES
    channels = []
    for name in vib_names:
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

    mic_names = _S3_MIC_NAMES if _is_s3_seg(seg_id) else _MIC_NAMES
    vib_names = _S3_VIB_NAMES if _is_s3_seg(seg_id) else _VIB_NAMES

    mic_stats = []
    for name in mic_names:
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
    for name in vib_names:
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

    if _is_s3_seg(seg_id):
        folder = _s3_seg_id_to_folder(seg_id)
    else:
        folder = _seg_id_to_folder(seg_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")

    raw_loc = _load_localization(folder)
    loc = _normalize_s3_loc(raw_loc) if _is_s3_seg(seg_id) else raw_loc
    cnf = _load_cnf_infer(folder)

    features: dict[str, float] = {}

    if loc:
        gt = loc.get("ground_truth_cm", [])
        features["gt_x_cm"] = float(gt[0]) if len(gt) > 0 else 0.0
        features["gt_y_cm"] = float(gt[1]) if len(gt) > 1 else 0.0
        features["gt_z_cm"] = float(gt[2]) if len(gt) > 2 else 0.0

        method_keys = (
            (
                "neural_cnn",
                "neural_cnn_s3",
                "neural_cnn_s2_zeroshot",
                "srp_phat",
                "tdoa_triangulation",
                "fused",
            )
            if _is_s3_seg(seg_id)
            else ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused")
        )
        for method_key in method_keys:
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

    if _is_s3_seg(seg_id):
        folder = _s3_seg_id_to_folder(seg_id)
    else:
        folder = _seg_id_to_folder(seg_id)
    if folder is None:
        raise HTTPException(status_code=404, detail=f"Segment {seg_id!r} not found")

    raw_loc = _load_localization(folder)
    loc = _normalize_s3_loc(raw_loc) if _is_s3_seg(seg_id) else raw_loc
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

        method_keys = (
            [
                "neural_cnn",
                "neural_cnn_s3",
                "neural_cnn_s2_zeroshot",
                "srp_phat",
                "tdoa_triangulation",
                "fused",
            ]
            if _is_s3_seg(seg_id)
            else ["neural_cnn", "srp_phat", "tdoa_triangulation", "fused"]
        )
        for key in method_keys:
            m = loc.get(key, {})
            if m and m.get("estimated_cm"):
                methods[key] = {
                    "estimated_cm": m["estimated_cm"],
                    "error_cm": m.get("error_cm"),
                    "method": m.get("method", key),
                }

        if _is_s3_seg(seg_id):
            mic_xyz = _S3_MIC_XYZ_CM
            vib_xyz = _S3_VIB_XYZ_CM
            extra: dict[str, Any] = {"gt_is_approx": True}
        else:
            mic_xyz = _S2_MIC_XYZ_CM
            vib_xyz = _S2_VIB_XYZ_CM
            extra = {"all_fault_positions_cm": _S2_FAULT_POSITIONS_CM}

        localization = {
            "ground_truth_cm": gt,
            "methods": methods,
            "best_method": loc.get("best_method"),
            "best_error_cm": loc.get("best_error_cm"),
            "mic_positions_cm": [v for v in mic_xyz.values()],
            "mic_names": list(mic_xyz.keys()),
            "vib_positions_cm": [v for v in vib_xyz.values()],
            "vib_names": list(vib_xyz.keys()),
            "operating_modes": loc.get("operating_modes", []),
            **extra,
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
        if _is_s3_seg(seg_id):
            folder = _s3_seg_id_to_folder(seg_id)
            raw = _load_localization(folder) if folder else {}
            loc = _normalize_s3_loc(raw) if raw else {}
        else:
            folder = _seg_id_to_folder(seg_id)
            loc = _load_localization(folder) if folder else {}

    gt = loc.get("ground_truth_cm")
    best = loc.get("best_method")
    best_est = None
    if best and best in loc:
        best_est = loc[best].get("estimated_cm")

    is_s3 = _is_s3_seg(seg_id)
    return {
        "type": "bench_top",
        "box_cm": [12, 11, 9] if is_s3 else [41, 41, 40],
        "mic_positions_cm": {
            k: v for k, v in (_S3_MIC_XYZ_CM if is_s3 else _S2_MIC_XYZ_CM).items()
        },
        "vib_positions_cm": {
            k: v for k, v in (_S3_VIB_XYZ_CM if is_s3 else _S2_VIB_XYZ_CM).items()
        },
        "known_fault_positions_cm": {} if is_s3 else _S2_FAULT_POSITIONS_CM,
        "ground_truth_cm": gt,
        "best_estimate_cm": best_est,
        "best_method": best,
    }


# --- Anomaly region (re-purposed for 3D position scatter) ------------------
@app.get("/api/anomaly-region")
async def anomaly_region(seg_id: str = Query("")):
    """Return per-method localization estimates for 3D scatter rendering."""
    is_s3 = _is_s3_seg(seg_id)
    mic_xyz = _S3_MIC_XYZ_CM if is_s3 else _S2_MIC_XYZ_CM
    vib_xyz = _S3_VIB_XYZ_CM if is_s3 else _S2_VIB_XYZ_CM

    # Healthy segments have no fault location
    if not seg_id or _is_healthy_seg(seg_id):
        return {
            "methods": {},
            "ground_truth_cm": None,
            "mic_positions_cm": {k: v for k, v in mic_xyz.items()},
            "vib_positions_cm": {k: v for k, v in vib_xyz.items()},
            "all_fault_positions_cm": {} if is_s3 else _S2_FAULT_POSITIONS_CM,
            "best_method": None,
            "best_error_cm": None,
        }

    if is_s3:
        folder = _s3_seg_id_to_folder(seg_id)
        if folder is None:
            return {"methods": {}, "ground_truth_cm": None}
        raw_loc = _load_localization(folder)
        loc = _normalize_s3_loc(raw_loc) if raw_loc else {}
        method_keys = [
            "neural_cnn",
            "neural_cnn_s3",
            "neural_cnn_s2_zeroshot",
            "srp_phat",
            "tdoa_triangulation",
            "fused",
        ]
    else:
        folder = _seg_id_to_folder(seg_id)
        if folder is None:
            return {"methods": {}, "ground_truth_cm": None}
        loc = _load_localization(folder)
        method_keys = ["neural_cnn", "srp_phat", "tdoa_triangulation", "fused"]

    methods = {}
    for key in method_keys:
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
        "mic_positions_cm": {k: v for k, v in mic_xyz.items()},
        "vib_positions_cm": {k: v for k, v in vib_xyz.items()},
        "all_fault_positions_cm": {} if is_s3 else _S2_FAULT_POSITIONS_CM,
        "best_method": loc.get("best_method"),
        "best_error_cm": loc.get("best_error_cm"),
        "gt_is_approx": True if is_s3 else False,
    }


# --- Alerts -----------------------------------------------------------------
@app.get("/api/alerts")
async def alerts():
    all_positions = list(_list_fault_positions()) + list(_list_s3_fault_positions())
    alert_list = []
    for pos in all_positions:
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
        # --- S3 (third dataset) models ---
        "s3_cnf": {
            "display": "[S3] Normalizing Flow (CNF)",
            "artifact": _S3_RESULTS_ROOT / "cnf" / "anomaly" / "flow.pt",
        },
        "s3_ocsvm": {
            "display": "[S3] OC-SVM (\u03bd=0.05)",
            "artifact": _S3_RESULTS_ROOT / "ocsvm" / "anomaly" / "anomaly_model.pkl",
        },
        "s3_ocsvm_nu_001": {
            "display": "[S3] OC-SVM (\u03bd=0.01)",
            "artifact": _S3_RESULTS_ROOT
            / "ocsvm"
            / "anomaly_nu_001"
            / "anomaly_model.pkl",
        },
        "s3_ocsvm_nu_003": {
            "display": "[S3] OC-SVM (\u03bd=0.03)",
            "artifact": _S3_RESULTS_ROOT
            / "ocsvm"
            / "anomaly_nu_003"
            / "anomaly_model.pkl",
        },
        "s3_ocsvm_nu_01": {
            "display": "[S3] OC-SVM (\u03bd=0.1)",
            "artifact": _S3_RESULTS_ROOT
            / "ocsvm"
            / "anomaly_nu_01"
            / "anomaly_model.pkl",
        },
        "s3_cnn_ae": {
            "display": "[S3] CNN Autoencoder",
            "artifact": _S3_RESULTS_ROOT / "cnn_ae" / "anomaly" / "anomaly_model.pkl",
        },
        "s3_lstm_ae": {
            "display": "[S3] LSTM Autoencoder",
            "artifact": _S3_RESULTS_ROOT / "lstm_ae" / "anomaly" / "anomaly_model.pkl",
        },
        "s3_localization_cnn_s3": {
            "display": "[S3] LocalizationCNNS3",
            "artifact": _S3_RESULTS_ROOT / "localization_cnn_s3.pt",
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
    rows = []
    # S2 fault positions
    for pos in _list_fault_positions():
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
            "dataset": "second",
        }
        for method in ("neural_cnn", "srp_phat", "tdoa_triangulation", "fused"):
            m = loc.get(method, {})
            if m and m.get("estimated_cm"):
                row["methods"][method] = {
                    "estimated_cm": m.get("estimated_cm"),
                    "error_cm": m.get("error_cm"),
                }
        rows.append(row)
    # S3 fault positions
    for pos in _list_s3_fault_positions():
        folder = _s3_seg_id_to_folder(pos["id"])
        if folder is None:
            continue
        raw_loc = _load_localization(folder)
        loc = _normalize_s3_loc(raw_loc) if raw_loc else {}
        gt = loc.get("ground_truth_cm")
        row = {
            "id": pos["id"],
            "folder": pos["folder"],
            "ground_truth_cm": gt,
            "best_method": loc.get("best_method"),
            "best_error_cm": loc.get("best_error_cm"),
            "methods": {},
            "dataset": "third",
        }
        for method in (
            "neural_cnn",
            "neural_cnn_s3",
            "neural_cnn_s2_zeroshot",
            "srp_phat",
            "tdoa_triangulation",
            "fused",
        ):
            m = loc.get(method, {})
            if m and m.get("estimated_cm"):
                row["methods"][method] = {
                    "estimated_cm": m.get("estimated_cm"),
                    "error_cm": m.get("error_cm"),
                }
        rows.append(row)
    return {"rows": rows}


# ===========================================================================
# Illwerke real-plant campaign dashboard API
# ===========================================================================

_ILLWERKE_LATENT_DIR = _REPO_ROOT / "artifacts" / "latents_illwerke"
_ILLWERKE_RESULTS_DIR = _REPO_ROOT / "results" / "illwerke"
_ILLWERKE_SUMMARY_PATH = _ILLWERKE_RESULTS_DIR / "illwerke_pipeline_summary.json"
_DTSS_ARTIFACT_DIR = _REPO_ROOT / "results" / "illwerke" / "unsupervised_mode"
_PIPELINE_DIR = _ILLWERKE_RESULTS_DIR / "pipeline"


# ---------------------------------------------------------------------------
# Campaign epoch — offset between relative seconds (0-based from first sample)
# and absolute UTC.  Set by pipeline.yaml `campaign_epoch_utc`; if missing we
# fall back to the campaign file-naming convention (2026-04-16T00:00:00Z).
# ---------------------------------------------------------------------------
def _get_campaign_epoch_s() -> float:
    """Return the campaign start time as a POSIX UTC timestamp (float seconds)."""
    from datetime import datetime, timezone as _tz

    # True campaign epoch: April 15 2026 midnight UTC.
    # The mode_timeline.json t=0 corresponds to April 15, not April 16 as previously
    # assumed.  The Synchronized_Campaign_20260416.json covers day-2 of the campaign.
    _default = datetime(2026, 4, 15, tzinfo=_tz.utc).timestamp()
    cfg_path = _REPO_ROOT / "configs" / "illwerke" / "pipeline.yaml"
    if cfg_path.exists():
        try:
            import yaml as _yaml  # type: ignore

            with open(cfg_path, encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            raw = _cfg.get("campaign_epoch_utc")
            if raw:
                return (
                    datetime.fromisoformat(str(raw)).replace(tzinfo=_tz.utc).timestamp()
                )
        except Exception:
            pass
    # Prefer reading the first timestamp from mode_timeline.json (absolute nanoseconds)
    tl_path = _PIPELINE_DIR / "mode_timeline.json"
    if tl_path.exists():
        try:
            tl = json.loads(tl_path.read_text())
            if tl:
                t0_ns = tl[0].get("t_start_ns", 0)
                if t0_ns > 946_684_800_000_000_000:  # after year 2000 in ns
                    return t0_ns / 1e9
        except Exception:
            pass
    return _default


_CAMPAIGN_EPOCH_S: float | None = None  # lazy-loaded


def _campaign_epoch() -> float:
    global _CAMPAIGN_EPOCH_S
    if _CAMPAIGN_EPOCH_S is None:
        _CAMPAIGN_EPOCH_S = _get_campaign_epoch_s()
    return _CAMPAIGN_EPOCH_S


def _rel_s_to_iso(rel_s: float) -> str:
    """Convert a pipeline-relative second offset → UTC ISO-8601 string."""
    from datetime import datetime, timezone as _tz

    ts = _campaign_epoch() + rel_s
    return datetime.fromtimestamp(ts, tz=_tz.utc).isoformat()


def _rel_s_to_ts(rel_s: float) -> float:
    """Convert a pipeline-relative second offset → absolute POSIX UTC seconds."""
    return _campaign_epoch() + rel_s


# ---------------------------------------------------------------------------
# Pipeline helper: load mode_timeline.json with absolute UTC timestamps
# ---------------------------------------------------------------------------
_pipeline_timeline_abs_cache: list[dict[str, Any]] | None = None


def _load_pipeline_timeline_abs() -> list[dict[str, Any]]:
    """Return pipeline mode_timeline.json with t_start/end as absolute UTC ts (float s)."""
    global _pipeline_timeline_abs_cache
    if _pipeline_timeline_abs_cache is not None:
        return _pipeline_timeline_abs_cache
    tl_path = _PIPELINE_DIR / "mode_timeline.json"
    if not tl_path.exists():
        return []
    try:
        raw: list[dict[str, Any]] = json.loads(tl_path.read_text())
    except Exception:
        return []
    epoch = _campaign_epoch()
    result = []
    for seg in raw:
        t0_ns = seg.get("t_start_ns", 0)
        t1_ns = seg.get("t_end_ns", 0)
        # year-2000 boundary in ns — distinguishes absolute from relative timestamps
        if t0_ns > 946_684_800_000_000_000:
            s_ts = t0_ns / 1e9
            e_ts = t1_ns / 1e9
        else:
            s_ts = epoch + seg.get("t_start_s", t0_ns / 1e9)
            e_ts = epoch + seg.get("t_end_s", t1_ns / 1e9)
        result.append({**seg, "_s_ts": s_ts, "_e_ts": e_ts})
    _pipeline_timeline_abs_cache = result
    return result


_LABEL_TO_MODE: dict[str, str] = {
    "ST": "Standstill",
    "TU": "Turbine",
    "PU": "Pump",
    "PH": "Phasenschieber",
}

_illwerke_cache: dict[str, Any] | None = None
_dtss_training_status: dict[str, Any] = {"status": "idle"}
_pipeline_anomaly_summary_cache: dict[str, Any] | None = None
_pipeline_validation_cache: dict[str, Any] | None = None


def _load_pipeline_anomaly_summary() -> dict[str, Any]:
    global _pipeline_anomaly_summary_cache
    if _pipeline_anomaly_summary_cache is not None:
        return _pipeline_anomaly_summary_cache
    path = _PIPELINE_DIR / "anomaly_summary.json"
    if path.exists():
        try:
            _pipeline_anomaly_summary_cache = json.loads(path.read_text())
            return _pipeline_anomaly_summary_cache
        except Exception:
            pass
    return {}


def _load_pipeline_validation() -> dict[str, Any]:
    global _pipeline_validation_cache
    if _pipeline_validation_cache is not None:
        return _pipeline_validation_cache
    path = _PIPELINE_DIR / "validation_report_L1.json"
    if path.exists():
        try:
            _pipeline_validation_cache = json.loads(path.read_text())
            return _pipeline_validation_cache
        except Exception:
            pass
    return {}

# Mapping: anomaly_train key → inference result key (differ by naming convention)
_IW_INFER_KEY_MAP: dict[str, str] = {
    "cnf": "cnf",
    "ocsvm_anomaly": "ocsvm",
    "ocsvm_anomaly_nu_001": "ocsvm_nu_001",
    "ocsvm_anomaly_nu_003": "ocsvm_nu_003",
    "ocsvm_anomaly_nu_01": "ocsvm_nu_01",
    "lstm_ae": "lstm_ae",
    "cnn_ae": "cnn_ae",
}

# Paths to each model's inference result JSON (flags + scores)
_IW_MODEL_PATHS: dict[str, "Path"] = {
    "cnf": _ILLWERKE_RESULTS_DIR / "cnf" / "anomaly" / "infer_result.json",
    "ocsvm_anomaly": _ILLWERKE_RESULTS_DIR / "ocsvm" / "anomaly" / "infer_result.json",
    "ocsvm_anomaly_nu_001": _ILLWERKE_RESULTS_DIR
    / "ocsvm"
    / "anomaly_nu_001"
    / "infer_result.json",
    "ocsvm_anomaly_nu_003": _ILLWERKE_RESULTS_DIR
    / "ocsvm"
    / "anomaly_nu_003"
    / "infer_result.json",
    "ocsvm_anomaly_nu_01": _ILLWERKE_RESULTS_DIR
    / "ocsvm"
    / "anomaly_nu_01"
    / "infer_result.json",
    "lstm_ae": _ILLWERKE_RESULTS_DIR / "lstm_ae" / "anomaly" / "infer_result.json",
    "cnn_ae": _ILLWERKE_RESULTS_DIR / "cnn_ae" / "anomaly" / "infer_result.json",
}

# Mode classifier artifacts (one per backbone family + SOTA model)
_IW_MODE_MODEL_PATHS: dict[str, "Path"] = {
    "cnf": _ILLWERKE_RESULTS_DIR / "cnf" / "mode" / "mode_classifier.pt",
    "ocsvm": _ILLWERKE_RESULTS_DIR / "ocsvm" / "mode" / "mode_classifier.pt",
    "lstm_ae": _ILLWERKE_RESULTS_DIR / "lstm_ae" / "mode" / "mode_classifier.pt",
    "cnn_ae": _ILLWERKE_RESULTS_DIR / "cnn_ae" / "mode" / "mode_classifier.pt",
}

# SOTA model uses a dedicated artifact with Viterbi CRF embedded
_IW_SOTA_MODE_PATH: "Path" = (
    _ILLWERKE_RESULTS_DIR / "mode_sota" / "mode_classifier_sota.pt"
)


def _load_dtss_spans() -> list[dict[str, Any]]:
    """Load DTSS mode_timeline.json and convert segments to the Gantt span format."""
    from datetime import datetime, timezone

    timeline_path = _DTSS_ARTIFACT_DIR / "mode_timeline.json"
    if not timeline_path.exists():
        return []
    try:
        timeline: list[dict[str, Any]] = json.loads(timeline_path.read_text())
    except Exception:
        return []

    epoch = _campaign_epoch()
    spans: list[dict[str, Any]] = []
    for seg in timeline:
        t0_ns = seg.get("t_start_ns", 0)
        t1_ns = seg.get("t_end_ns", 0)
        if t1_ns <= t0_ns:
            continue
        s_ts = t0_ns / 1e9
        e_ts = t1_ns / 1e9
        # Relative timestamps (< year-2000 epoch) → add campaign epoch
        if s_ts < 946_684_800:
            s_ts += epoch
            e_ts += epoch
        if e_ts <= s_ts:
            continue
        label = seg.get("label", "ST")
        mode = _LABEL_TO_MODE.get(
            label,
            "Transitioning" if label.startswith("Transitioning") else "Standstill",
        )
        ds = datetime.fromtimestamp(s_ts, tz=timezone.utc)
        de = datetime.fromtimestamp(e_ts, tz=timezone.utc)
        rpm_v, pwr_v, flow_v = _seg_proc_vars(s_ts, e_ts, label)
        spans.append(
            {
                "mode": mode,
                "start_ts": s_ts,
                "end_ts": e_ts,
                "start_iso": ds.isoformat(),
                "end_iso": de.isoformat(),
                "start_hm": ds.strftime("%H:%M"),
                "end_hm": de.strftime("%H:%M"),
                "day": ds.strftime("%Y-%m-%d"),
                "n_windows": int(seg.get("duration_s", e_ts - s_ts)),
                "rpm_mean": rpm_v,
                "power_mean": pwr_v,
                "flow_mean": flow_v,
                "cluster_id": seg.get("cluster_id"),
                "micro_event": seg.get("micro_event", False),
                "posterior_entropy": seg.get("posterior_entropy", 0.0),
            }
        )
    return spans


def _build_illwerke_cache() -> dict[str, Any]:
    """Build in-memory cache from physics pipeline numpy arrays.

    Windows are at 60-second intervals across the 8-day campaign.
    Process variables come from the proc-var cache (Synchronized_Campaign_*.json);
    for days without exact data, per-mode medians are used as fallback.
    """
    from datetime import datetime, timezone

    seq_path = _PIPELINE_DIR / "smoothed_state_sequence.npy"
    if not seq_path.exists():
        return {
            "windows": [],
            "spans": [],
            "model_spans": {},
            "cnf_threshold": None,
            "summary": {},
            "days": [],
        }

    seq = np.load(str(seq_path))  # (633600,) int32  0=ST 1=TU 2=PU 3=PH
    z_path = _PIPELINE_DIR / "anomaly_z_scores.npy"
    alert_path = _PIPELINE_DIR / "anomaly_alert_level.npy"
    sub_path = _PIPELINE_DIR / "tu_sub_mode_labels.npy"
    z_arr = (
        np.load(str(z_path))
        if z_path.exists()
        else np.full(len(seq), float("nan"), dtype=np.float32)
    )
    al_arr = (
        np.load(str(alert_path))
        if alert_path.exists()
        else np.zeros(len(seq), dtype=np.int8)
    )
    sub_arr = (
        np.load(str(sub_path))
        if sub_path.exists()
        else np.full(len(seq), -1, dtype=np.int8)
    )

    epoch = _campaign_epoch()
    n = len(seq)
    _IDX_TO_LABEL: dict[int, str] = {0: "ST", 1: "TU", 2: "PU", 3: "PH"}

    # Build 60-second windows (10 560 for a 633 600-sample campaign)
    all_windows: list[dict[str, Any]] = []
    for i in range(0, n, 60):
        label = _IDX_TO_LABEL.get(int(seq[i]), "ST")
        mode = _LABEL_TO_MODE.get(label, "Standstill")
        t_s = epoch + i
        rpm, power, flow = _seg_proc_vars(t_s, t_s + 60.0, label)
        z_val = float(z_arr[i]) if not np.isnan(z_arr[i]) else None
        all_windows.append(
            {
                "ts": t_s,
                "mode": mode,
                "rpm": round(rpm, 1),
                "power_mw": round(power, 1),
                "flow_m3s": round(flow, 1),
                "cnf_score": z_val,
                "z_score": z_val,
                "alert_level": int(al_arr[i]),
                "sub_mode": int(sub_arr[i]),
                "is_anomaly": int(al_arr[i]) > 0,
                "model_flags": {},
            }
        )

    # Optionally attach existing anomaly model inference flags
    cnf_threshold: float | None = None
    for model_key, mpath in _IW_MODEL_PATHS.items():
        if not mpath.exists():
            continue
        try:
            with open(mpath, encoding="utf-8") as fh:
                ir = json.load(fh)
            flags = [bool(v) for v in ir.get("flags", [])]
            if model_key == "cnf":
                cnf_threshold = ir.get("threshold")
            for idx, w in enumerate(all_windows):
                w["model_flags"][model_key] = flags[idx] if idx < len(flags) else None
        except Exception as exc:
            print(f"[illwerke_cache] model flags '{model_key}': {exc}")

    # Build pipeline spans from mode_timeline.json using absolute timestamps
    pipeline_spans: list[dict[str, Any]] = []
    for seg in _load_pipeline_timeline_abs():
        s_ts = seg["_s_ts"]
        e_ts = seg["_e_ts"]
        label = seg.get("label", "ST")
        mode = _LABEL_TO_MODE.get(label, "Standstill")
        ds = datetime.fromtimestamp(s_ts, tz=timezone.utc)
        de = datetime.fromtimestamp(e_ts, tz=timezone.utc)
        rpm_v, pwr_v, flow_v = _seg_proc_vars(s_ts, e_ts, label)
        pipeline_spans.append(
            {
                "mode": mode,
                "start_ts": s_ts,
                "end_ts": e_ts,
                "start_iso": ds.isoformat(),
                "end_iso": de.isoformat(),
                "start_hm": ds.strftime("%H:%M"),
                "end_hm": de.strftime("%H:%M"),
                "day": ds.strftime("%Y-%m-%d"),
                "n_windows": int(e_ts - s_ts),
                "rpm_mean": rpm_v,
                "power_mean": pwr_v,
                "flow_mean": flow_v,
                "micro_event": seg.get("micro_event", False),
            }
        )

    # Load anomaly events with absolute UTC timestamps
    anomaly_events: list[dict[str, Any]] = []
    events_path = _PIPELINE_DIR / "anomaly_events.json"
    if events_path.exists():
        try:
            for ev in json.loads(events_path.read_text()):
                s_ts = epoch + ev["t_start_s"]
                e_ts = epoch + ev["t_end_s"]
                dt_ev = datetime.fromtimestamp(s_ts, tz=timezone.utc)
                midnight = datetime(
                    dt_ev.year, dt_ev.month, dt_ev.day, tzinfo=timezone.utc
                ).timestamp()
                anomaly_events.append(
                    {
                        **ev,
                        "ts_start": s_ts,
                        "ts_end": e_ts,
                        "ts_iso": dt_ev.isoformat() + "Z",
                        "day": dt_ev.strftime("%Y-%m-%d"),
                        "hour": round((s_ts - midnight) / 3600.0, 3),
                    }
                )
        except Exception as exc:
            print(f"[illwerke_cache] anomaly_events.json: {exc}")

    model_spans: dict[str, list[dict[str, Any]]] = {"pipeline": pipeline_spans}
    dtss_spans = _load_dtss_spans()
    if dtss_spans:
        model_spans["dtss"] = dtss_spans

    days = sorted(
        {
            datetime.fromtimestamp(w["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            for w in all_windows
        }
    )

    return {
        "windows": all_windows,
        "spans": pipeline_spans,
        "model_spans": model_spans,
        "cnf_threshold": cnf_threshold,
        "summary": {},
        "days": days,
        "anomaly_events": anomaly_events,
        "anomaly_summary": _load_pipeline_anomaly_summary(),
        "validation_report": _load_pipeline_validation(),
    }


def _get_illwerke_cache() -> dict[str, Any]:
    global _illwerke_cache
    if _illwerke_cache is None:
        _illwerke_cache = _build_illwerke_cache()
    return _illwerke_cache


@app.get("/api/illwerke/overview")
async def illwerke_overview() -> dict[str, Any]:
    cache = _get_illwerke_cache()
    total = len(cache["windows"])

    pipeline_events = cache.get("anomaly_summary") or _load_pipeline_anomaly_summary()
    vr = cache.get("validation_report") or _load_pipeline_validation()
    dwell_ratios: dict[str, float] = vr.get("dwell_ratios", {})

    # Mode duration from timeline (precise seconds per operating state)
    tl = _load_pipeline_timeline_abs()
    mode_duration_s: dict[str, float] = {}
    for seg in tl:
        lbl = seg.get("label", "")
        mode_name = _LABEL_TO_MODE.get(lbl, lbl)
        mode_duration_s[mode_name] = (
            mode_duration_s.get(mode_name, 0.0) + seg["_e_ts"] - seg["_s_ts"]
        )

    date_range = (
        [cache["days"][0], cache["days"][-1]] if len(cache["days"]) >= 2 else []
    )
    if not date_range and tl:
        from datetime import datetime, timezone as _tz

        date_range = [
            datetime.fromtimestamp(tl[0]["_s_ts"], tz=_tz.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(tl[-1]["_e_ts"], tz=_tz.utc).strftime("%Y-%m-%d"),
        ]

    return {
        "n_days": len(cache["days"]),
        "n_days_skipped": 0,
        "total_windows": total,
        "mode_counts": {},
        "cnf_anomaly_rate": None,
        "cnf_n_anomalous": None,
        "cnf_threshold": cache.get("cnf_threshold"),
        "models_trained": False,
        "date_range": date_range,
        "window_s": 60,
        "step_s": 60,
        # ---- pipeline fields ----
        "pipeline_events": pipeline_events,
        "dwell_ratios": dwell_ratios,
        "mode_duration_s": mode_duration_s,
        "pipeline_available": True,
    }


@app.get("/api/illwerke/gantt")
async def illwerke_gantt(
    model: str = Query(""),
    mode_source: str = Query("kmeans"),
) -> dict[str, Any]:
    """Mode spans clipped to daily boundaries for a multi-row Gantt chart.

    - ``model``       – anomaly model key; flagged windows returned as
                        ``anomaly_pts`` for scatter-overlay rendering.
    - ``mode_source`` – ``kmeans`` (default) uses K-Means labels; any other
                        value picks the pre-computed mode-classifier spans
                        (cnf / ocsvm / lstm_ae / cnn_ae).
    """
    from datetime import datetime, timedelta, timezone

    cache = _get_illwerke_cache()

    # Select which span list to use; "pipeline" is the default source
    model_spans = cache.get("model_spans", {})
    if mode_source in model_spans:
        active_spans = model_spans[mode_source]
        effective_source = mode_source
    else:
        active_spans = cache.get("spans", [])
        effective_source = "pipeline"

    gantt_rows: list[dict[str, Any]] = []

    for sp in active_spans:
        s_ts, e_ts = sp["start_ts"], sp["end_ts"]
        dt_s = datetime.fromtimestamp(s_ts, tz=timezone.utc)
        dt_e = datetime.fromtimestamp(e_ts, tz=timezone.utc)
        # Walk each calendar day this span may touch
        day = dt_s.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end_limit = dt_e.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= day_end_limit:
            d_str = day.strftime("%Y-%m-%d")
            d_s_ts = day.timestamp()
            d_e_ts = d_s_ts + 86400.0
            c_s = max(s_ts, d_s_ts)
            c_e = min(e_ts, d_e_ts)
            if c_e > c_s:
                gantt_rows.append(
                    {
                        "day": d_str,
                        "mode": sp["mode"],
                        "start_h": round((c_s - d_s_ts) / 3600.0, 4),
                        "duration_h": round((c_e - c_s) / 3600.0, 4),
                        "start_hm": datetime.fromtimestamp(
                            c_s, tz=timezone.utc
                        ).strftime("%H:%M"),
                        "end_hm": datetime.fromtimestamp(c_e, tz=timezone.utc).strftime(
                            "%H:%M"
                        ),
                        "rpm_mean": sp["rpm_mean"],
                        "power_mean": sp["power_mean"],
                        "flow_mean": sp["flow_mean"],
                        "n_windows": sp["n_windows"],
                    }
                )
            day += timedelta(days=1)

    # Anomaly overlay: collect (day, hour) for every window flagged by the
    # requested model.  Silently ignore unknown / missing model keys.
    anomaly_pts: list[dict[str, Any]] = []
    if model and model in _IW_MODEL_PATHS:
        for w in cache["windows"]:
            mf = w.get("model_flags", {})
            if mf.get(model):
                dt = datetime.fromtimestamp(w["ts"], tz=timezone.utc)
                base_ts = datetime(
                    dt.year, dt.month, dt.day, tzinfo=timezone.utc
                ).timestamp()
                anomaly_pts.append(
                    {
                        "day": dt.strftime("%Y-%m-%d"),
                        "hour": round((w["ts"] - base_ts) / 3600.0, 4),
                    }
                )

    # Available mode-source options: pipeline first, then DTSS if present
    available_mode_sources = list(cache.get("model_spans", {}).keys())
    if "pipeline" not in available_mode_sources:
        available_mode_sources.insert(0, "pipeline")
    else:
        available_mode_sources = ["pipeline"] + [
            s for s in available_mode_sources if s != "pipeline"
        ]

    return {
        "rows": gantt_rows,
        "days": cache.get("days", []),
        "anomaly_model": model or None,
        "anomaly_pts": anomaly_pts,
        "mode_source": effective_source,
        "available_mode_sources": available_mode_sources,
    }


@app.get("/api/illwerke/daily/{date_str}")
async def illwerke_daily(date_str: str) -> dict[str, Any]:
    """Process-variable time series for one calendar day (UTC)."""
    from datetime import datetime, timezone

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    cache = _get_illwerke_cache()
    day_ws = sorted(
        (
            w
            for w in cache["windows"]
            if datetime.fromtimestamp(w["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            == date_str
        ),
        key=lambda w: w["ts"],
    )
    if not day_ws:
        raise HTTPException(status_code=404, detail=f"No data for {date_str}")

    base_ts = datetime.fromisoformat(date_str + "T00:00:00+00:00").timestamp()
    return {
        "date": date_str,
        "hours": [round((w["ts"] - base_ts) / 3600.0, 4) for w in day_ws],
        "rpm": [w["rpm"] for w in day_ws],
        "power_mw": [w["power_mw"] for w in day_ws],
        "flow_m3s": [w["flow_m3s"] for w in day_ws],
        "modes": [w["mode"] for w in day_ws],
        "cnf_scores": [w["cnf_score"] for w in day_ws],
        "z_scores": [w.get("z_score") for w in day_ws],
        "alert_levels": [w.get("alert_level", 0) for w in day_ws],
        "sub_modes": [w.get("sub_mode", -1) for w in day_ws],
        "is_anomaly": [w["is_anomaly"] for w in day_ws],
        "watch_sigma": 4.0,
        "alert_sigma": 6.0,
    }


@app.get("/api/illwerke/scores")
async def illwerke_scores(
    max_points: int = Query(3000, ge=100, le=20000),
) -> dict[str, Any]:
    """Timestamped anomaly z-scores (decimated) for the full campaign."""
    from datetime import datetime, timezone

    cache = _get_illwerke_cache()
    ws = [w for w in cache["windows"] if w.get("z_score") is not None]
    if not ws:
        return {
            "ts_iso": [],
            "scores": [],
            "modes": [],
            "is_anomaly": [],
            "threshold": None,
            "watch_threshold": 4.0,
            "alert_threshold": 6.0,
            "score_label": "Reconstruction Z-Score",
        }

    step = max(1, len(ws) // max_points)
    dec = ws[::step]
    return {
        "ts_iso": [
            datetime.fromtimestamp(w["ts"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
            for w in dec
        ],
        "scores": [w["z_score"] for w in dec],
        "modes": [w["mode"] for w in dec],
        "is_anomaly": [w["is_anomaly"] for w in dec],
        "threshold": None,
        "watch_threshold": 4.0,
        "alert_threshold": 6.0,
        "score_label": "Reconstruction Z-Score",
    }


@app.get("/api/illwerke/models")
async def illwerke_models() -> dict[str, Any]:
    """Training status and inference statistics for all Illwerke anomaly models."""
    train_summary: dict[str, Any] = {}
    infer_summary: dict[str, Any] = {}
    if _ILLWERKE_SUMMARY_PATH.exists():
        with open(_ILLWERKE_SUMMARY_PATH, encoding="utf-8") as fh:
            full = json.load(fh)
        stages = full.get("stages", {})
        train_summary = stages.get("anomaly_train", {})
        infer_summary = stages.get("inference", {})

    model_meta: list[tuple[str, str]] = [
        ("cnf", "CNF — Normalizing Flow"),
        ("ocsvm_anomaly", "OC-SVM (default nu)"),
        ("ocsvm_anomaly_nu_001", "OC-SVM (nu = 0.01)"),
        ("ocsvm_anomaly_nu_003", "OC-SVM (nu = 0.03)"),
        ("ocsvm_anomaly_nu_01", "OC-SVM (nu = 0.10)"),
        ("lstm_ae", "LSTM Autoencoder"),
        ("cnn_ae", "CNN Autoencoder"),
    ]
    rows: list[dict[str, Any]] = []
    for key, display in model_meta:
        train_info = train_summary.get(key, {})
        infer_key = _IW_INFER_KEY_MAP.get(key, key)
        inf = infer_summary.get(infer_key, {})
        rows.append(
            {
                "key": key,
                "display": display,
                "status": train_info.get("status", "missing"),
                "n_windows": inf.get("n_windows"),
                "n_anomalous": inf.get("n_anomalous"),
                "anomaly_rate": inf.get("anomaly_rate"),
                "score_mean": inf.get("score_mean"),
            }
        )
    return {"models": rows}


# ===========================================================================
# DTSS — Unsupervised Mode Detection API
# ===========================================================================


@app.get("/api/illwerke/dtss/status")
async def dtss_status() -> dict[str, Any]:
    """Return DTSS training status and a summary of any existing artifacts."""
    artifacts_exist = (_DTSS_ARTIFACT_DIR / "mode_timeline.json").exists()
    validation: dict[str, Any] = {}
    cluster_label_map: dict[str, str] = {}
    n_segments = 0
    if artifacts_exist:
        try:
            tl = json.loads((_DTSS_ARTIFACT_DIR / "mode_timeline.json").read_text())
            n_segments = len(tl)
        except Exception:
            pass
        vr_path = _DTSS_ARTIFACT_DIR / "validation_report.json"
        if vr_path.exists():
            try:
                validation = json.loads(vr_path.read_text())
            except Exception:
                pass
        cl_path = _DTSS_ARTIFACT_DIR / "cluster_to_label.json"
        if cl_path.exists():
            try:
                cluster_label_map = json.loads(cl_path.read_text())
            except Exception:
                pass
    return {
        "training_status": _dtss_training_status.get("status", "idle"),
        "artifacts_exist": artifacts_exist,
        "artifact_dir": str(_DTSS_ARTIFACT_DIR),
        "n_segments": n_segments,
        "cluster_to_label": cluster_label_map,
        "validation": validation,
        "error": _dtss_training_status.get("error"),
    }


@app.post("/api/illwerke/dtss/train")
async def dtss_train() -> dict[str, Any]:
    """Start DTSS training in a background thread.  Returns immediately."""
    import threading
    import time

    global _dtss_training_status, _illwerke_cache

    if _dtss_training_status.get("status") == "training":
        return {
            "status": "already_running",
            "message": "Training is already in progress.",
        }

    _dtss_training_status = {"status": "training", "started_at": time.time()}

    def _run() -> None:
        global _dtss_training_status, _illwerke_cache
        import sys as _sys

        _sys.path.insert(0, str(_REPO_ROOT))
        try:
            from src.modeling.mode.illwerke_unsupervised.train import train_dtss  # type: ignore

            # Read data_root from config if yaml is available, else fall back.
            data_root = "E:/MasterThesisData/illwerke-data-230426"
            cfg_path = _REPO_ROOT / "configs" / "illwerke" / "unsupervised_mode.yaml"
            if cfg_path.exists():
                try:
                    import yaml as _yaml  # type: ignore

                    with open(cfg_path, encoding="utf-8") as _f:
                        _cfg = _yaml.safe_load(_f)
                    data_root = _cfg.get("data_root", data_root)
                except Exception:
                    pass

            train_dtss(
                data_root=data_root,
                output_dir=str(_DTSS_ARTIFACT_DIR),
                device="cpu",
            )
            _dtss_training_status = {"status": "done", "finished_at": time.time()}
            _illwerke_cache = (
                None  # invalidate so DTSS spans are reloaded on next request
            )
        except Exception as exc:
            _dtss_training_status = {"status": "failed", "error": str(exc)}

    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "started",
        "message": "DTSS training started in background (~45 min on CPU).",
    }


# ===========================================================================
# Physics Pipeline — dedicated read-only API (no cache build required)
# ===========================================================================

_IW_MODE_LABEL_TO_DISPLAY: dict[str, str] = {
    "ST": "Standstill",
    "TU": "Turbine",
    "PU": "Pump",
    "PH": "Phasenschieber",
    "TRANSITION": "Transitioning",
    "UNKNOWN": "Unknown",
}

# ---------------------------------------------------------------------------
# Process-variable cache (RPM / Power / Flow / Head at 1-minute resolution)
# Loaded once in a background thread from Synchronized_Campaign_*.json files.
# Falls back to per-mode p50 medians (L1 exploration) until ready.
# ---------------------------------------------------------------------------
import threading as _proc_threading

_PROC_CACHE: dict[str, Any] | None = None
_PROC_CACHE_LOCK = _proc_threading.Lock()
_PROC_CACHE_READY: bool = False

# Per-mode p50 fallback values: (rpm, power_mw, flow_m3s)
_PROC_MODE_FALLBACK: dict[str, tuple[float, float, float]] = {
    "ST": (0.2, 0.0, 0.0),
    "TU": (378.6, 209.8, 70.6),
    "PU": (-377.8, -280.5, 72.9),
    "PH": (-377.9, -3.4, 0.0),
    "TRANSITION": (0.0, 0.0, 0.0),
    "UNKNOWN": (0.0, 0.0, 0.0),
}


def _build_proc_cache_sync() -> None:
    """Load all Synchronized_Campaign_*.json files and build minute-averaged arrays."""
    global _PROC_CACHE, _PROC_CACHE_READY  # noqa: PLW0603
    from datetime import datetime, timezone as _tz

    ws_root = _REPO_ROOT
    files = sorted(ws_root.glob("Synchronized_Campaign_*.json"))
    if not files:
        return

    all_ts_s: list[float] = []
    all_rpm: list[float] = []
    all_power: list[float] = []
    all_flow_tu: list[float] = []
    all_flow_pu: list[float] = []
    all_head: list[float] = []

    for fp in files:
        try:
            raw: dict[str, Any] = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        channels: list[str] = raw.get("channels_allg", [])
        data: list[list[float]] = raw.get("data_allg", [])  # [n_ts × n_ch]
        ts_ns_raw: list[int] = raw.get("timestamps_ns", [])
        meta: dict[str, Any] = raw.get("metadata", {})

        if not data or not ts_ns_raw:
            continue

        # Determine this file's absolute UTC epoch from metadata.date or filename
        date_str: str = str(meta.get("date", ""))
        if not date_str:
            # Try to parse from filename e.g. "Synchronized_Campaign_20260416"
            stem = fp.stem  # "Synchronized_Campaign_20260416"
            date_str = stem.split("_")[-1]  # "20260416"
        if len(date_str) == 10:  # "YYYY-MM-DD"
            file_epoch = (
                datetime.fromisoformat(date_str + "T00:00:00")
                .replace(tzinfo=_tz.utc)
                .timestamp()
            )
        elif len(date_str) == 8:  # "YYYYMMDD"
            file_epoch = datetime(
                int(date_str[:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
                tzinfo=_tz.utc,
            ).timestamp()
        else:
            file_epoch = _campaign_epoch()

        arr = np.array(data, dtype=np.float32)  # (n_ts, n_ch)
        ts_ns = np.array(ts_ns_raw, dtype=np.float64)
        ts_s = file_epoch + ts_ns / 1e9

        ch_idx: dict[str, int] = {ch: i for i, ch in enumerate(channels)}
        rpm_i = ch_idx.get("1_Drehzahl UPM")
        pow_i = ch_idx.get("1_P_Ist")
        ftu_i = ch_idx.get("Durchfluss TU")
        fpu_i = ch_idx.get("Durchfluss PU")
        hd_i = ch_idx.get("Oberwasserpegel")

        # Decimate to 60-second non-overlapping means
        step = 60
        n = len(ts_s)
        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            chunk = arr[i0:i1]
            mid = i0 + (i1 - i0) // 2
            all_ts_s.append(float(ts_s[mid]))
            all_rpm.append(
                round(float(np.mean(chunk[:, rpm_i])), 1) if rpm_i is not None else 0.0
            )
            all_power.append(
                round(float(np.mean(chunk[:, pow_i])), 1) if pow_i is not None else 0.0
            )
            all_flow_tu.append(
                round(float(np.mean(chunk[:, ftu_i])), 2) if ftu_i is not None else 0.0
            )
            all_flow_pu.append(
                round(float(np.mean(chunk[:, fpu_i])), 2) if fpu_i is not None else 0.0
            )
            all_head.append(
                round(float(np.mean(chunk[:, hd_i])), 2) if hd_i is not None else 0.0
            )

    with _PROC_CACHE_LOCK:
        _PROC_CACHE = {
            "ts": np.array(all_ts_s, dtype=np.float64),
            "rpm": np.array(all_rpm, dtype=np.float32),
            "power": np.array(all_power, dtype=np.float32),
            "flow_tu": np.array(all_flow_tu, dtype=np.float32),
            "flow_pu": np.array(all_flow_pu, dtype=np.float32),
            "head": np.array(all_head, dtype=np.float32),
        }
        _PROC_CACHE_READY = True


def _proc_cache_get() -> dict[str, Any] | None:
    """Return cache dict if ready, else None (thread-safe read)."""
    if _PROC_CACHE_READY:
        with _PROC_CACHE_LOCK:
            return _PROC_CACHE
    return None


def _seg_proc_vars(s_ts: float, e_ts: float, label: str) -> tuple[float, float, float]:
    """Return (rpm, power_mw, flow_m3s) for a segment using cached data or fallback."""
    cache = _proc_cache_get()
    if cache is not None:
        ts = cache["ts"]
        mask = (ts >= s_ts) & (ts < e_ts)
        n = int(mask.sum())
        if n > 0:
            rpm = round(float(np.mean(cache["rpm"][mask])), 1)
            power = round(float(np.mean(cache["power"][mask])), 1)
            ftu = float(np.mean(cache["flow_tu"][mask]))
            fpu = float(np.mean(cache["flow_pu"][mask]))
            flow = round(max(ftu, fpu), 1)
            return rpm, power, flow
    # Fallback: per-mode p50 from L1 exploration
    return _PROC_MODE_FALLBACK.get(label, (0.0, 0.0, 0.0))


# Start background process-var cache build immediately (daemon thread)
_proc_threading.Thread(
    target=_build_proc_cache_sync, daemon=True, name="proc-var-cache"
).start()


@app.get("/api/illwerke/pipeline/gantt")
async def pipeline_gantt() -> dict[str, Any]:
    """Daily Gantt rows derived from the physics pipeline mode_timeline.json.

    Response schema is intentionally identical to /api/illwerke/gantt so the
    frontend can reuse renderIllwerkeGantt() unchanged.
    """
    from datetime import datetime, timedelta, timezone as _tz

    tl = _load_pipeline_timeline_abs()
    if not tl:
        return {
            "rows": [],
            "days": [],
            "anomaly_pts": [],
            "mode_source": "pipeline",
            "available_mode_sources": ["pipeline"],
        }

    gantt_rows: list[dict[str, Any]] = []
    all_days: set[str] = set()

    for seg in tl:
        if seg.get("micro_event"):
            continue  # skip transitions < 60 s
        s_ts: float = seg["_s_ts"]
        e_ts: float = seg["_e_ts"]
        label = seg.get("label", "ST")
        mode = _IW_MODE_LABEL_TO_DISPLAY.get(label, label)

        dt_s = datetime.fromtimestamp(s_ts, tz=_tz.utc)
        dt_e = datetime.fromtimestamp(e_ts, tz=_tz.utc)
        day = dt_s.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end_limit = dt_e.replace(hour=0, minute=0, second=0, microsecond=0)

        while day <= day_end_limit:
            d_str = day.strftime("%Y-%m-%d")
            all_days.add(d_str)
            d_s_ts = day.timestamp()
            d_e_ts = d_s_ts + 86400.0
            c_s = max(s_ts, d_s_ts)
            c_e = min(e_ts, d_e_ts)
            if c_e > c_s:
                rpm_v, pwr_v, flow_v = _seg_proc_vars(c_s, c_e, label)
                gantt_rows.append(
                    {
                        "day": d_str,
                        "mode": mode,
                        "start_h": round((c_s - d_s_ts) / 3600.0, 4),
                        "duration_h": round((c_e - c_s) / 3600.0, 4),
                        "start_hm": datetime.fromtimestamp(c_s, tz=_tz.utc).strftime(
                            "%H:%M"
                        ),
                        "end_hm": datetime.fromtimestamp(c_e, tz=_tz.utc).strftime(
                            "%H:%M"
                        ),
                        "rpm_mean": rpm_v,
                        "power_mean": pwr_v,
                        "flow_mean": flow_v,
                        "n_windows": seg.get("duration_s", int(e_ts - s_ts)),
                        "micro_event": False,
                    }
                )
            day += timedelta(days=1)

    days = sorted(all_days)
    return {
        "rows": gantt_rows,
        "days": days,
        "anomaly_pts": [],
        "anomaly_model": None,
        "mode_source": "pipeline",
        "available_mode_sources": ["pipeline"],
    }


@app.get("/api/illwerke/pipeline/events")
async def pipeline_events(
    severity: str = Query("", description="Filter: 'alert', 'watch', or '' for all"),
    max_events: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    """Anomaly events from the L5 physics pipeline with UTC timestamps."""
    ev_path = _PIPELINE_DIR / "anomaly_events.json"
    summary_path = _PIPELINE_DIR / "anomaly_summary.json"

    if not ev_path.exists():
        return {"events": [], "summary": {}, "epoch_iso": _rel_s_to_iso(0)}

    try:
        raw_events: list[dict[str, Any]] = json.loads(ev_path.read_text())
    except Exception:
        return {"events": [], "summary": {}, "epoch_iso": _rel_s_to_iso(0)}

    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    filtered = [e for e in raw_events if not severity or e.get("severity") == severity]
    out_events: list[dict[str, Any]] = []
    for ev in filtered[:max_events]:
        s_ts = _rel_s_to_ts(ev["t_start_s"])
        e_ts = _rel_s_to_ts(ev["t_end_s"])
        out_events.append(
            {
                "t_start_iso": _rel_s_to_iso(ev["t_start_s"]),
                "t_end_iso": _rel_s_to_iso(ev["t_end_s"]),
                "t_start_s": s_ts,
                "t_end_s": e_ts,
                "duration_s": ev.get("duration_s", ev["t_end_s"] - ev["t_start_s"]),
                "severity": ev.get("severity", "watch"),
                "mode": ev.get("mode", ""),
                "sub_mode": ev.get("sub_mode", ""),
                "peak_z_score": ev.get("peak_z_score"),
                "mean_z_score": ev.get("mean_z_score"),
                # For Gantt overlay
                "day": _rel_s_to_iso(ev["t_start_s"])[:10],
                "hour": round((s_ts - int(s_ts // 86400) * 86400) / 3600.0, 4),
            }
        )

    return {
        "events": out_events,
        "summary": summary,
        "epoch_iso": _rel_s_to_iso(0),
        "n_total": len(raw_events),
        "n_filtered": len(filtered),
    }


@app.get("/api/illwerke/pipeline/scores")
async def pipeline_scores(
    max_points: int = Query(4000, ge=100, le=20000),
    date: str = Query("", description="YYYY-MM-DD filter; empty = full campaign"),
) -> dict[str, Any]:
    """Decimated per-second anomaly z-scores from the L5 pipeline.

    Returns both the z-score array and the alert-level integer array (0=normal,
    1=watch, 2=alert) alongside watch_threshold and alert_threshold values.
    """
    zscores_path = _PIPELINE_DIR / "anomaly_z_scores.npy"
    alert_path = _PIPELINE_DIR / "anomaly_alert_level.npy"
    state_seq_path = _PIPELINE_DIR / "smoothed_state_sequence.npy"

    if not zscores_path.exists():
        return {
            "ts_iso": [],
            "z_scores": [],
            "alert_level": [],
            "modes": [],
            "watch_threshold": 4.0,
            "alert_threshold": 6.0,
        }

    z = np.load(str(zscores_path))
    al = (
        np.load(str(alert_path))
        if alert_path.exists()
        else np.zeros(len(z), dtype=np.int8)
    )
    ss = (
        np.load(str(state_seq_path))
        if state_seq_path.exists()
        else np.zeros(len(z), dtype=np.int8)
    )

    # Map state_id → mode name
    _IDX_TO_MODE = {0: "Standstill", 1: "Turbine", 2: "Pump", 3: "Phasenschieber"}
    epoch = _campaign_epoch()

    # Clamp extreme outliers for display (cap at 200σ so axis doesn't blow out)
    z_disp = np.clip(z, -10.0, 200.0)

    if date and re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        from datetime import datetime, timezone as _tz

        day_start = datetime.fromisoformat(date + "T00:00:00+00:00").timestamp()
        day_end = day_start + 86400.0
        t_arr = epoch + np.arange(len(z), dtype=np.float64)
        mask = (t_arr >= day_start) & (t_arr < day_end)
        indices = np.where(mask)[0]
        rel_origin = day_start
    else:
        indices = np.arange(len(z))
        rel_origin = epoch

    step = max(1, len(indices) // max_points)
    dec_idx = indices[::step]

    t_arr = epoch + dec_idx.astype(np.float64)
    from datetime import datetime, timezone as _tz

    if step == 1:
        ts_format = "%Y-%m-%dT%H:%M:%S"
    else:
        ts_format = "%Y-%m-%dT%H:%M"

    hours = [(float(t) - rel_origin) / 3600.0 for t in t_arr]

    watch_thr = 4.0
    alert_thr = 6.0
    # Read from pipeline.yaml if available
    cfg_path = _REPO_ROOT / "configs" / "illwerke" / "pipeline.yaml"
    if cfg_path.exists():
        try:
            import yaml as _yaml

            with open(cfg_path, encoding="utf-8") as _f:
                _pcfg = _yaml.safe_load(_f)
            l5 = _pcfg.get("layer5", {})
            watch_thr = float(l5.get("watch_sigma", 4.0))
            alert_thr = float(l5.get("alert_sigma", 6.0))
        except Exception:
            pass

    return {
        "ts_iso": [
            datetime.fromtimestamp(float(t), tz=_tz.utc).strftime(ts_format)
            for t in t_arr
        ],
        "hours": [round(h, 4) for h in hours],
        "z_scores": z_disp[dec_idx].tolist(),
        "alert_level": al[dec_idx].astype(int).tolist(),
        "modes": [_IDX_TO_MODE.get(int(ss[i]), "Unknown") for i in dec_idx],
        "watch_threshold": watch_thr,
        "alert_threshold": alert_thr,
        "date": date or None,
    }


@app.get("/api/illwerke/pipeline/transitions")
async def pipeline_transitions() -> dict[str, Any]:
    """Typed machine state transitions from L2 with UTC timestamps and signature validation."""
    tr_path = _PIPELINE_DIR / "transition_segments.json"
    if not tr_path.exists():
        return {"transitions": [], "n_total": 0}

    try:
        raw: list[dict[str, Any]] = json.loads(tr_path.read_text())
    except Exception:
        return {"transitions": [], "n_total": 0}

    out = []
    for tr in raw:
        s_ts = _rel_s_to_ts(tr["t_start_s"])
        e_ts = _rel_s_to_ts(tr["t_end_s"])
        sig = tr.get("signature_details", {})
        sig_match = tr.get("signature_match")
        out.append(
            {
                "t_start_iso": _rel_s_to_iso(tr["t_start_s"]),
                "t_end_iso": _rel_s_to_iso(tr["t_end_s"]),
                "t_start_s": s_ts,
                "t_end_s": e_ts,
                "duration_s": tr.get("duration_s", tr["t_end_s"] - tr["t_start_s"]),
                "transition_type": tr.get("transition_type", "—"),
                "label_before": tr.get("label_before", ""),
                "label_after": tr.get("label_after", ""),
                "signature_match": sig_match,
                "micro_event": tr.get("micro_event", False),
                "invalid_topology": tr.get("invalid_topology", False),
                "day": _rel_s_to_iso(tr["t_start_s"])[:10],
            }
        )

    # Count signature types
    typed_with_checker = [t for t in out if t["signature_match"] is not None]
    n_verified = sum(1 for t in typed_with_checker if t["signature_match"] is True)

    return {
        "transitions": out,
        "n_total": len(out),
        "n_verified": n_verified,
        "n_typed": len(typed_with_checker),
    }


@app.get("/api/illwerke/pipeline/validation")
async def pipeline_validation() -> dict[str, Any]:
    """L1 physics oracle validation report + signal thresholds."""
    vr_path = _PIPELINE_DIR / "validation_report_L1.json"
    thr_path = _PIPELINE_DIR / "signal_thresholds.json"
    exp_path = _PIPELINE_DIR / "exploration_report.json"

    vr: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    exploration: dict[str, Any] = {}

    if vr_path.exists():
        try:
            vr = json.loads(vr_path.read_text())
        except Exception:
            pass
    if thr_path.exists():
        try:
            thresholds = json.loads(thr_path.read_text())
        except Exception:
            pass
    if exp_path.exists():
        try:
            exploration = json.loads(exp_path.read_text())
        except Exception:
            pass

    return {
        "available": vr_path.exists(),
        "dwell_ratios": vr.get("dwell_ratios", {}),
        "steady_coverage_pct": vr.get("steady_coverage_pct"),
        "sensor_freeze": vr.get("sensor_freeze", {}),
        "head_independence": vr.get("head_independence", {}),
        "hand_label_agreement": {},  # loaded from L1_full_exploration.json if needed
        "signal_thresholds": thresholds,
        "exploration_summary": (
            {
                ch: {
                    "mean": round(float(stats.get("mean", 0)), 3),
                    "std": round(float(stats.get("std", 0)), 3),
                    "min": round(float(stats.get("min", 0)), 3),
                    "max": round(float(stats.get("max", 0)), 3),
                }
                for ch, stats in (exploration.get("channel_stats") or {}).items()
            }
            if exploration
            else {}
        ),
    }


@app.get("/api/illwerke/pipeline/process_vars")
async def pipeline_process_vars(
    date: str = Query(..., description="Date string YYYY-MM-DD, e.g. '2026-04-16'"),
) -> dict[str, Any]:
    """Return minute-averaged RPM, Power, and Flow for a given campaign day.

    Served from the in-memory process-variable cache (built on startup from
    Synchronized_Campaign_*.json files).  If the cache is not yet ready the
    response will be empty and the frontend should retry.
    """
    from datetime import datetime, timezone as _tz

    # Parse requested day
    try:
        day_dt = datetime.fromisoformat(date + "T00:00:00").replace(tzinfo=_tz.utc)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {date!r}")

    day_start = day_dt.timestamp()
    day_end = day_start + 86400.0

    cache = _proc_cache_get()
    if cache is None:
        return {
            "ready": False,
            "date": date,
            "ts_iso": [],
            "rpm": [],
            "power": [],
            "flow_tu": [],
            "flow_pu": [],
            "head": [],
        }

    ts_arr = cache["ts"]
    mask = (ts_arr >= day_start) & (ts_arr < day_end)
    n = int(mask.sum())
    if n == 0:
        return {
            "ready": True,
            "date": date,
            "ts_iso": [],
            "rpm": [],
            "power": [],
            "flow_tu": [],
            "flow_pu": [],
            "head": [],
        }

    ts_sel = ts_arr[mask]
    rpm_sel = cache["rpm"][mask]
    power_sel = cache["power"][mask]
    flow_tu_sel = cache["flow_tu"][mask]
    flow_pu_sel = cache["flow_pu"][mask]
    head_sel = cache["head"][mask]

    # Decimate to at most 1440 points for the response (already 1-minute resolution)
    max_pts = 1440
    if n > max_pts:
        step = n // max_pts
        idx = np.arange(0, n, step)
        ts_sel = ts_sel[idx]
        rpm_sel = rpm_sel[idx]
        power_sel = power_sel[idx]
        flow_tu_sel = flow_tu_sel[idx]
        flow_pu_sel = flow_pu_sel[idx]
        head_sel = head_sel[idx]

    return {
        "ready": True,
        "date": date,
        "ts_iso": [
            datetime.fromtimestamp(float(t), tz=_tz.utc).strftime("%H:%M")
            for t in ts_sel
        ],
        "rpm": [round(float(v), 1) for v in rpm_sel],
        "power": [round(float(v), 1) for v in power_sel],
        "flow_tu": [round(float(v), 1) for v in flow_tu_sel],
        "flow_pu": [round(float(v), 1) for v in flow_pu_sel],
        "head": [round(float(v), 2) for v in head_sel],
    }
