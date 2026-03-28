# CMSC_Net_Hydro

Multimodal acoustic-vibration pipeline for hydropower operating-mode recognition and anomaly detection.

## What This Repo Contains

- Data ingestion for grouped and flat recording layouts.
- Preprocessing and handcrafted feature extraction.
- Latent cache generation (`z`, `c`) for model training.
- Baseline anomaly models (OC-SVM, LSTM-AE, CNN-AE).
- Conditional normalizing flow (CNF) anomaly pipeline.
- Mode classifier training over latent features.
- End-to-end training orchestrator and report generation.

## Repository Layout

```text
configs/                # Training/inference configuration files
src/
  features/             # Acoustic/vibration feature extractors
  ingestion/            # Dataset scanning and loading
  preprocessing/        # Signal preprocessing blocks
  modeling/             # Latent building, model train/infer, reporting
tests/unit/             # Unit tests
```

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,dl]"
```

## Main Workflows

### 1) Build Latent Caches

```python
from pathlib import Path
from src.modeling.latent.builder import build_latent_cache

summaries = build_latent_cache(
    data_root=Path("data/All"),
    output_dir=Path("artifacts/latents"),
    mode="mic_vibration",
)
print(f"Built {len(summaries)} latent file(s)")
```

### 2) Train All Models (Recommended)

```python
from src.modeling.orchestration.train_all import main as train_all_models_main

train_all_models_main([
    "--config-dir", "configs",
    "--artifacts-root", "results",
])
```

This runs anomaly training, per-family mode training, RandomFault inference, and consolidated report generation.

### 3) Train Individual Pipelines (Function API)

- CNF anomaly: `src.modeling.flow.train.train_and_calibrate_flow`
- Mode classifier: `src.modeling.mode.train.train_mode_classifier`
- Baseline anomaly: `src.modeling.baselines.train.train_baseline_model`
- Baseline inference: `src.modeling.baselines.train.infer_baseline_model`
- CNF inference: `src.modeling.flow.infer.score_with_context_smoothing`

## Test Suite

```powershell
c:/DEV/CMSC_Net_Hydro/.venv/Scripts/python.exe -m pytest tests/unit -q
```

## Configuration Convention

- Orchestration loads YAML configs from `configs/` through `src/modeling/cli/config.py` (`load_yaml_config`).
- YAML keys are normalized to snake_case when loaded.
- Individual model APIs are function-first and receive typed kwargs directly.

## Outputs

- `artifacts/latents/`: cached latent arrays.
- `results/`: serialized model artifacts, inference outputs, and model reports.

These output directories are intentionally ignored by git.
