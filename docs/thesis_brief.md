# Master Thesis Brief

## Metadata

- Title: Multimodal Sensing for Acoustic-Vibration Based Fault Localization in Hydropower
- Student: Zakaria Omarar
- Type: Master thesis

## Motivation

Hydropower plants run electromechanical assets for long daily operating periods. Unexpected halts for inspections and repairs affect both grid continuity and operating cost. Predictive maintenance methods are useful but typically react once degradation trends are already visible. This motivates earlier diagnostic capabilities for anomaly detection and source localization.

## Problem Statement

The target scenario is difficult because anomaly localization is needed around operating transitions (turbine/pump), while multiple machines and reflective plant structures create mixed and reverberant acoustic fields. Fluid-borne and structure-borne effects overlap in the same observation windows, reducing separability when using one modality only.

## Research Question

To what extent can latent operational context be extracted from multimodal acoustic and vibration streams, and does context-conditioned anomaly detection and source localization improve robustness under domain shift?

## Objectives

1. Build a synchronized multimodal pipeline for audio and vibration streams.
2. Extract robust features for downstream context learning and detection.
3. Prepare localization-relevant spatial and cross-modal descriptors.
4. Keep the early pipeline modular so later model variants can be tested cleanly.

## Sensor Setup Modeled in This Starter

- Level 1: 4 microphones at 0, 90, 180, 270 degrees.
- Level 2: 5 microphones at 0, 72, 144, 216, 288 degrees.
- Level 2: 4 accelerometers at 0, 90, 180, 270 degrees.
- Audio input: mono WAV, 16 kHz, 16-bit.
- Vibration input: CSV with timestamped FFT peak amplitude and dominant frequency.

## What This Starter Intentionally Includes

- Core data contract (`DataSegment`)
- Ingestion and format adapters
- Preprocessing stack
- Feature contract (`FeatureFrame`) and multimodal extractors

## What Is Intentionally Left for Later

- Operating-state classifiers
- Anomaly detector models
- Source-localization decision modules
- Alerting and production dashboarding
