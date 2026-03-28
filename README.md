# CMSC_Net_Hydro

The pipeline processes synchronized acoustic and vibration recordings from a reversible Francis pump-turbine, extracts multimodal features, builds latent representations, and trains anomaly detection models conditioned on the operational context (Turbine / Pump / Standstill).

---

## Thesis Context

**Facility:** Rodundwerk II (ROW II), Vorarlberg, Austria — single reversible Francis unit, 295 MW turbine / 286 MW pump, 375 rpm nominal.

**Sensor array (13 sensors, NTP-synchronized):**

- Level 1 (generator): 4 microphones at 0°, 90°, 180°, 270°
- Level 2 (turbine): 5 microphones at 72° spacing + 4 accelerometers at 90° spacing
- Audio: mono WAV, 16 kHz, 16-bit — Vibration: CSV with FFT peak amplitude and dominant frequency

**Operating modes:** Standstill, Turbine, Pump — each with distinct acoustic/vibration signatures that the anomaly model must condition on.

---

## Pipeline Overview

```
WAV / CSV recordings
  → ingestion          (scan and load recordings by layout)
  → preprocessing      (calibrate → DC remove → bandpass → z-score)
  → feature extraction (time-domain, frequency-domain, envelope, cross-channel, acoustic)
  → latent cache build (z feature vector, c context vector → .npz)
  → model training     (CNF anomaly + mode classifier + baselines)
  → inference & report (anomaly events, mode labels, consolidated report)
```

---

## Repository Layout

```text
configs/                          # YAML training/inference configs
data/                             # Input recordings (gitignored)
  All/                            # All modes combined (used for latent build)
  Pump/ Turbine/ Standstill/ ...  # Per-mode splits
artifacts/latents/                # Built latent .npz caches (gitignored)
results/                          # Model artifacts, inference outputs, reports (gitignored)
src/
  config/                         # Global constants (sensor geometry, machine params)
  data/                           # DataSegment contract
  features/                       # Multimodal feature extractors
  ingestion/                      # Recording scanning and loading
  preprocessing/                  # Signal preprocessing blocks
  modeling/
    core/                         # Shared contracts, runtime utils, run manifest
    latent/                       # Latent cache builder and preprocessing helpers
    models/                       # Model architecture definitions (CNF, AEs, MLP, OC-SVM)
    flow/                         # Conditional normalizing flow train/infer pipeline
    mode/                         # Operating-mode classifier pipeline
    baselines/                    # OC-SVM, LSTM-AE, CNN-AE baseline pipelines
    orchestration/                # End-to-end train-all orchestrator
    reporting/                    # Consolidated model report generation
    cli/                          # YAML config loader
tests/unit/                       # Unit test suite
```

---

## Models

| Model                   | Type                         | Description                                                                                                                       |
| ----------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **CNF**                 | Conditional normalizing flow | Primary anomaly model. Learns `p(z \| c)`; score = −log p. Context `c` adapts the density per operating mode via FiLM modulation. |
| **OC-SVM**              | One-class SVM                | Baseline. Trained on healthy latent vectors `z`.                                                                                  |
| **LSTM-AE**             | LSTM autoencoder             | Baseline. Reconstruction error as anomaly score.                                                                                  |
| **CNN-AE**              | CNN autoencoder              | Baseline. Reconstruction error as anomaly score.                                                                                  |
| **ModeMLP / ModeCNN2D** | Mode classifier              | Labels each window Pump / Turbine / Standstill; gates anomaly flags at transitions.                                               |

---

## Setup

**Requirements:** Python ≥ 3.11

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,dl]"
```

Core dependencies: `numpy`, `scipy`, `librosa`, `pywavelets`, `pyyaml`, `pydantic`.
Deep learning extras (`dl`): `torch`, `scikit-learn`, `umap-learn`.

---

## Usage

### Build Latent Caches

Build `.npz` latent files from raw recordings before any model training.

```python
from pathlib import Path
from src.modeling.latent.builder import build_latent_cache

summaries = build_latent_cache(
    data_root=Path("data/All"),
    output_dir=Path("artifacts/latents"),
    mode="mic_vibration",
)
```

### Train All Models

Runs the full sequence: anomaly training → mode training → RandomFault inference → report.

```powershell
python -m train_all_models
```

Or with explicit config overrides:

```powershell
python -m train_all_models --config-dir configs --artifacts-root results
```

### Individual Pipelines

| Task                  | Entry point                                            |
| --------------------- | ------------------------------------------------------ |
| CNF train/calibrate   | `src.modeling.flow.train.train_and_calibrate_flow`     |
| CNF inference         | `src.modeling.flow.infer.score_with_context_smoothing` |
| Mode classifier train | `src.modeling.mode.train.train_mode_classifier`        |
| Baseline train        | `src.modeling.baselines.train.train_baseline_model`    |
| Baseline inference    | `src.modeling.baselines.train.infer_baseline_model`    |

---

## Configuration

YAML configs live in `configs/`. Keys are normalized to snake_case on load.

| File                                      | Purpose                                 |
| ----------------------------------------- | --------------------------------------- |
| `anomaly_train.yaml`                      | CNF anomaly model training              |
| `anomaly_infer_randomfault.yaml`          | CNF inference on RandomFault recordings |
| `anomaly_baseline_train.yaml`             | Baseline model training                 |
| `anomaly_baseline_infer_randomfault.yaml` | Baseline inference                      |
| `mode_train.yaml`                         | Mode classifier training                |

---

## Outputs

| Path                                      | Contents                                                       |
| ----------------------------------------- | -------------------------------------------------------------- |
| `artifacts/latents/*.npz`                 | Latent arrays `z`, `c`, `recording_id`, `is_transition_window` |
| `results/<model>/anomaly/`                | Trained model artifacts and inference JSON                     |
| `results/<model>/mode/`                   | Mode classifier artifacts                                      |
| `results/reports/model_report.json`       | Consolidated comparison across all models                      |
| `results/reports/train_all_manifest.json` | Per-job run manifest with signatures and status                |

---

## Tests

```powershell
python -m pytest tests/unit/ -q
```
