# Architecture Map

This document explains how data flows through the repository and where to make changes safely.

## 1) End-to-End Pipeline

Data roots
-> ingestion (recording scanning + loading)
-> preprocessing
-> feature extraction
-> latent cache building (z, c)
-> model training or inference
-> summaries and reports

## 2) Main Source Areas

- src/ingestion
  - Input discovery and file loading for grouped/flat layouts.
- src/preprocessing
  - Signal calibration, filtering, normalization, segmentation.
- src/features
  - Acoustic and vibration handcrafted feature extractors.
- src/modeling
  - Latent build, anomaly models, mode models, flow model, orchestration, reporting.

## 3) Modeling Entry Points

- src/modeling/latent/builder.py
  - Builds .npz latent files with arrays z, c and metadata arrays.
- src/modeling/flow/train_core.py
  - Owns reusable flow training-loop, scoring, and clustering helper routines.
- src/modeling/flow/train.py
  - Owns end-to-end flow training/calibration orchestration and artifact serialization.
- src/modeling/flow/trainer.py
  - Function-first flow training facade plus optional latent auto-build.
- src/modeling/flow/infer.py
  - Runs CNF inference and event extraction over latent windows.
- src/modeling/mode/train.py
  - Trains latent-window mode classifier and exports mode artifacts.
- src/modeling/baselines/train.py
  - Trains/runs OC-SVM, LSTM-AE, CNN-AE baselines.
- src/modeling/orchestration/train_all.py
  - Orchestrates full training, resume/retry/fail-fast controls, report generation,
    and run-manifest export with config/data/job signatures.
- src/modeling/reporting/report.py
  - Aggregates outputs into comparable model reports; prefers run-manifest-driven
    deterministic resolution when manifest is provided.

Canonical package layout:

- src/modeling/core
  - Shared contracts and cross-domain runtime helpers.
- src/modeling/latent
  - Latent cache building and latent-specific preprocessing helpers.
- src/modeling/models
  - Model architecture definitions/factories and contrastive-pretraining utilities.
- src/modeling/flow
  - Flow data/train/infer/artifacts/detection-head modules.
- src/modeling/baselines
  - Baseline data/metrics/train modules.
- src/modeling/mode
  - Mode train/artifact/eval/data modules.
- src/modeling/orchestration
  - Train-all orchestration entrypoint.
- src/modeling/reporting
  - Consolidated reporting entrypoint.
- src/modeling/cli
  - Shared YAML config loader utilities.

## 4) Shared Modeling Utilities

- src/modeling/cli/config.py
  - Shared YAML config loading and key normalization used by orchestration.
- src/modeling/core/artifact_contracts.py
  - Artifact metadata schema stamping and validation helpers.
- src/modeling/core/runtime_utils.py
  - Shared runtime helpers for structured event logs, latent file path resolution,
    recording-id health/mode naming rules, window-step computations,
    post-processing label smoothing/run lengths, and feature standardization.
- src/modeling/core/run_manifest.py
  - Typed contracts for orchestration job results and stage rollups.

## 5) Config Conventions

The orchestration pipeline loads YAML files under configs/.
YAML keys are normalized to snake_case by the config loader.

Important config groups:

- configs/anomaly_train.yaml
- configs/anomaly_infer_randomfault.yaml
- configs/anomaly_baseline_train.yaml
- configs/anomaly_baseline_infer_randomfault.yaml
- configs/mode_train.yaml

## 6) Output Contracts

- Latent caches: artifacts/latents/\*.npz
  - required arrays: z, c
  - optional arrays: recording_id, is_transition_window
- Unified artifacts root (default: models/):
- Unified artifacts root (default: results/):
  - <artifacts-root>/<model>/anomaly/\*
  - <artifacts-root>/<model>/mode/\*
  - <artifacts-root>/reports/model_report.json
  - <artifacts-root>/reports/train_all_manifest.json

## 7) Change Safety Guide

Low-risk changes:

- Add metrics/summary fields.
- Add new config options with defaults.
- Add model-report columns.

Medium-risk changes:

- Split logic, threshold calibration, smoothing policy.
- Feature filtering and normalization changes.

High-risk changes:

- Latent schema changes (z/c shape or meaning).
- Artifact format changes without backward compatibility guards.

## 8) Recommended Handover Checklist

1. Run tests: python -m pytest tests/unit -q
2. Run full orchestrator once: python -m train_all_models
3. Inspect report and manifest outputs under <artifacts-root>/reports.
4. Confirm config files match intended experiment defaults.
