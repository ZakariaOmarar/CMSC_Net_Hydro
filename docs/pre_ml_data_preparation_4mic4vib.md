# Pre-ML Data Preparation Runbook (4 Microphones + 4 Vibration Channels)

## 1) Purpose and Constraints

This document defines all preparation steps that must be completed before any machine-learning model training/inference.

Scope:

- Early thesis dataset with paired files in one folder:
  - `recorded_<NODE>_<MODE>.wav`
  - `vibration_<NODE>_<MODE>.csv`
- Nodes: `B`, `C`, `D`, `E`
- Modes: `Pump`, `RandomFault`, `StandStill`, `Turbine`

Hard constraints from project requirements:

- Do not modify, overwrite, or delete raw files.
- Do not fabricate data.
- Do not fill missing sensor information with synthetic values.
- Any exclusion must be traceable and justified in metadata.

A "prepared" dataset is acceptable only if every gate in this runbook is passed or explicitly waived with a documented reason.

---

## 2) Observed Dataset Profile (Ground Truth from Current Files)

From the current `All` folder inventory and signal audit:

- Total files: `32`
- WAV files: `16`
- CSV files: `16`
- Pairing completeness:
  - Missing CSV for WAV key: none
  - Missing WAV for CSV key: none

WAV properties (all files):

- Channels: mono (`1`)
- Sample rate: `16000 Hz`
- Bit depth: `16-bit`
- Duration range: `557.344 s` to `633.824 s`
- Mean duration: `598.814 s`

Vibration CSV properties (all files):

- Rows per file: min `2143`, max `2438`, mean `2299.562`
- Timestamp monotonicity: all files monotonic (no non-positive step)
- Global timestamp step stats (`esp_time_us` diffs):
  - min `259969 us`
  - max `1260000 us`
  - mean `260357.017 us`
  - median `260000 us`
- Effective nominal vibration sampling cadence:
  - approximately `260 ms` per sample
  - approximately `3.846 Hz` nominal

Important timestamp discontinuity findings (large gaps):

- `vibration_D_Turbine.csv`: 3 large gaps (`1250000`/`1260000 us`)
- `vibration_E_RandomFault.csv`: 8 large gaps (`1250000`/`1260000 us`)
- `vibration_E_Turbine.csv`: 2 large gaps (`1250000`/`1260000 us`)

Vibration feature value ranges by mode (for baseline QC expectations):

- Pump:
  - amplitude min/max/mean/std: `3805.30 / 20850.32 / 6345.955 / 1439.607`
  - frequency range: `125.00` to `257.81` (5 unique)
- RandomFault:
  - amplitude min/max/mean/std: `192.89 / 51542.70 / 4289.463 / 5050.717`
  - frequency range: `7.81` to `492.19` (57 unique)
- StandStill:
  - amplitude min/max/mean/std: `186.75 / 615.97 / 287.126 / 45.830`
  - frequency range: `7.81` to `492.19` (63 unique)
- Turbine:
  - amplitude min/max/mean/std: `635.10 / 16473.52 / 1538.997 / 1067.739`
  - frequency range: `117.19` to `289.06` (7 unique)

Use these values as empirical reference bands for QC alarms, not as hard clipping bounds.

---

## 3) Non-Destructive Data Governance

### 3.1 Raw data freeze

- Create read-only snapshot manifest before processing.
- Store for each raw file:
  - relative path
  - byte size
  - SHA-256 hash
  - created timestamp (filesystem)
  - modified timestamp (filesystem)

Acceptance gate:

- Hashes remain unchanged for all raw files throughout project lifecycle.

### 3.2 Derived-data policy

- All processed artifacts must be written under a separate derived root (example: `derived/` or `artifacts/`).
- Never write into the raw folder.
- Include provenance metadata with each artifact:
  - source raw file hashes
  - code version/commit id
  - config id
  - creation timestamp

### 3.3 Reproducibility manifest

For each run, generate a run manifest containing:

- run id
- environment (python version, package versions)
- preprocessing parameters
- segmentation parameters
- feature extraction parameters
- exclusion decisions and rationale

---

## 4) Canonical Pairing and Recording Identity

### 4.1 Recording key

Use a strict key:

- `recording_key = <NODE>_<MODE>`
- Example: `B_Pump`

### 4.2 Pairing rules

For each key, exactly one WAV and one CSV are expected:

- WAV: `recorded_<NODE>_<MODE>.wav`
- CSV: `vibration_<NODE>_<MODE>.csv`

Acceptance gate:

- No duplicate keys.
- No missing pair member.
- Exactly 16 keys for this dataset.

### 4.3 Channel identity mapping

Treat node letters as channel identity for this early dataset:

- Node `B` -> channel 0
- Node `C` -> channel 1
- Node `D` -> channel 2
- Node `E` -> channel 3

Store this mapping in metadata and keep it fixed across all runs.

---

## 5) Stage A: File-Level Integrity Validation

### 5.1 WAV validation

For every WAV file, verify:

- extension is `.wav`
- readable RIFF/WAVE container
- mono channel only
- sample rate `16000 Hz`
- sample width `16-bit` integer
- finite duration and non-zero frame count

Reject/flag conditions:

- unreadable file
- sample rate mismatch
- multi-channel data
- zero-length audio

### 5.2 CSV schema validation

Required columns:

- timestamp: `esp_time_us` (or accepted alias)
- amplitude column
- frequency column

Validate:

- header exists
- at least one data row
- numeric parse success for required columns
- no all-NaN column after parse

Reject/flag conditions:

- missing required semantic column
- non-numeric parse failures over threshold
- empty file

### 5.3 Timestamp integrity checks

For each CSV:

- `dt = diff(esp_time_us)`
- monotonicity check: all `dt > 0`
- gap check bands:
  - nominal band: `259000 us` to `261000 us`
  - warning band: outside nominal but below `500000 us`
  - discontinuity band: `>= 500000 us`

Acceptance gate:

- Monotonicity must pass.
- Discontinuities are allowed only if they are logged and handled downstream by gap-aware segmentation rules.

---

## 6) Stage B: Timebase Construction Without Raw Mutation

Goal: construct analysis-time axes without editing source CSV rows.

### 6.1 Build per-file vibration time axis

For each vibration file:

- Parse integer `esp_time_us` as-is.
- Convert to seconds relative to first timestamp:
  - `t_rel_s = (t_us - t_us[0]) / 1e6`

### 6.2 Gap annotation

Create a per-sample boolean mask from `dt`:

- `gap_after[i] = True` if `dt[i] >= 500000 us`

Derived segment boundaries:

- split vibration stream into contiguous spans between gaps
- keep original rows unchanged; only annotate boundaries

### 6.3 Audio time axis

For WAV:

- `t_audio[n] = n / 16000`

### 6.4 Cross-modal overlap window

For each recording key:

- `T_common = min(duration_wav, duration_vibration)`
- analysis is restricted to `[0, T_common]`

Acceptance gate:

- `T_common > 0`
- duration mismatch ratio stays within expected band (currently around 1.000 to 1.001 observed)

---

## 7) Stage C: Gap-Aware Resampling for Vibration Features

Do not impute fake physics values across large timestamp gaps.

### 7.1 Why this matters

The current adapter computes average sampling rate from full series and linearly interpolates. With discontinuity gaps, naive interpolation smears values across missing intervals.

### 7.2 Required strategy

Resample amplitude/frequency piecewise per contiguous span:

- For each contiguous span (no `dt >= 500000 us` inside):
  - perform interpolation only within that span
- For missing intervals between spans:
  - emit `NaN` in derived resampled stream
  - mark these regions in quality mask

### 7.3 Target rate

For early pipeline compatibility:

- keep vibration target resample at `4 Hz`
- this is close to observed nominal `~3.846 Hz`

Acceptance gate:

- No interpolation across discontinuity boundaries.
- Gap regions are explicitly represented as invalid/masked.

---

## 8) Stage D: Segmentation Policy Before Feature Extraction

### 8.1 Window design constraints

Given vibration cadence (`~0.26 s/sample`), very short windows are unstable for vibration statistics.

Recommended initial windows:

- window length: `5 s` to `10 s`
- overlap: `50%`

Rationale:

- 5 s window contains about `19` vibration samples at nominal cadence.
- Supports robust median/percentile and low-frequency trend metrics.

### 8.2 Valid-window criteria

A window is valid only if:

- audio slice exists and contains finite values
- vibration slice has minimum count threshold (example: >= 75% expected samples)
- no discontinuity gap crossing unless masked handling is explicit
- NaN fraction in vibration derived channels below threshold (example: <= 10%)

### 8.3 Window metadata

Attach to every window:

- `recording_key`
- node and mode
- `t_start`, `t_end`
- validity flag
- reason codes for invalid windows (if any)

---

## 9) Stage E: Sensor Normalization and Calibration (Derived Only)

### 9.1 Audio preprocessing (derived stream)

Per channel:

- remove DC offset
- optional bandpass (domain-configured)
- normalization strategy fixed across train/val/test

### 9.2 Vibration preprocessing (derived stream)

Per channel:

- robust scaling recommended (median/IQR) due to heavy-tailed RandomFault amplitude
- keep both raw amplitude-derived and log-transformed feature variants in derived space
- preserve frequency channel as measured; avoid arbitrary clipping to mode-specific ranges

### 9.3 Leakage prevention

Fit normalization parameters only on training partition.
Apply frozen parameters to validation/test.

Acceptance gate:

- no fitting on global dataset before split
- scaler artifacts versioned and reproducible

---

## 10) Stage F: Labeling and Split Strategy

### 10.1 Label source of truth

Mode labels from filename suffix:

- `Pump`, `RandomFault`, `StandStill`, `Turbine`

### 10.2 Split policy for tiny dataset

Avoid random window-level split across same recording.
Use grouped split by recording identity to prevent leakage:

- group by `recording_key`
- optionally leave-one-node-out or leave-one-mode-instance-out depending experiment

### 10.3 Class balance report

Before model training, produce:

- windows per class (valid only)
- windows per node
- valid/invalid ratio per class and per node

Acceptance gate:

- documented imbalance and mitigation plan (class weighting, stratified grouped folds, etc.)

---

## 11) Stage G: Feature Readiness Checks

### 11.1 Audio feature checks

For each feature family (time/frequency/time-frequency):

- finite value ratio
- per-feature variance > epsilon
- no constant columns across all windows

### 11.2 Cross-modal feature checks

For cross-channel and audio-vibration relations:

- verify channel mapping consistency (`B,C,D,E` order)
- verify expected shape and ordering of feature vectors
- verify stable behavior in gap-masked windows

### 11.3 Drift baseline checks

Build baseline summary tables from current dataset:

- per-mode feature mean/std
- per-node feature mean/std

Use baseline only as reference alarms in future data intake.

---

## 12) Stage H: Final Model-Input Packaging

### 12.1 Output schema

Produce one tabular dataset (example: Parquet) with:

- row = one valid window
- columns:
  - identifiers (`recording_key`, node, mode, window index)
  - timing (`t_start`, `t_end`)
  - quality metrics/masks
  - extracted features
  - label

### 12.2 Quality sidecar

Create companion quality report file including:

- excluded windows count and reasons
- gap statistics per recording
- NaN rates before and after filtering
- class/node coverage after filtering

### 12.3 Traceability

Each packaged file must include references to:

- raw file hashes used
- run manifest id
- feature config id

Acceptance gate:

- packaged dataset reproducible from raw data + manifest + code commit.

---

## 13) Mandatory Pre-ML Checklist (Pass/Fail)

1. Raw hash manifest created and validated unchanged.
2. 16/16 WAV-CSV pairs found by key `<NODE>_<MODE>`.
3. WAV technical checks pass (`mono`, `16 kHz`, `16-bit`).
4. CSV schema checks pass (time/amplitude/frequency parseable).
5. Timestamp monotonicity passes for all files.
6. Discontinuity gaps are detected and logged.
7. Resampling is piecewise and gap-aware (no across-gap interpolation).
8. Window validity policy applied with reason-coded exclusions.
9. Train/val/test split is grouped to avoid leakage.
10. Normalization fit only on training partition.
11. Feature matrices contain finite, non-constant, correctly ordered fields.
12. Final packaged dataset has full provenance metadata.

No model training should start until all checklist items are passed or explicitly waived in a signed experiment note.

---

## 14) Project-Specific Notes for Current CSMC_Net Starter

Current implementation status relevant to this dataset:

- Scanner default pattern already matches current WAV naming: `recorded_*.wav`.
- Ingestion now accepts both 4-mic and 9-mic recordings by default.
- Flat grouped layout (`recorded_<sensor>_<recording>.wav`) is now supported and grouped automatically by `<recording>`.
- Current vibration resampling path estimates rate from mean timestamp delta and interpolates directly.

Before production experiments on this 4-mic dataset, align code/config with this runbook:

- enforce node-to-channel ordering (`B,C,D,E`)
- add gap-aware vibration resampling + quality mask propagation
- emit run manifest and quality sidecar for every feature build

---

## 15) Suggested Minimal Deliverables Before First Real Model Run

1. `intake_report.json`

- pairing table, wav/csv stats, timestamp gap summary

2. `quality_report.json`

- valid vs invalid windows, exclusion reasons, per-mode coverage

3. `features.parquet`

- final model-ready feature table with labels and identifiers

4. `run_manifest.json`

- exact parameters, data hashes, code version, environment

5. `experiment_note.md`

- explicit waivers (if any), assumptions, and known risks

These five artifacts are sufficient for an auditable first ML training baseline without violating the raw-data immutability requirement.
