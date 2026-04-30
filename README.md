# CMSC_Net_Hydro

Multimodal acoustic + vibration anomaly detection and source localization for
multi-mode reversible Francis pump-turbines.  Master's thesis codebase
implementing the V0 → V5 pipeline described in
[`.claude/plans/because-i-have-not-replicated-acorn.md`](.claude/plans/because-i-have-not-replicated-acorn.md).

## Thesis claim

> A single label-free model in which an unsupervised operational-context
> vector, learned by self-supervised pretraining on fused acoustic-vibration
> streams, simultaneously conditions an anomaly detection head and an
> anomaly-gated source-localization head.

The chained system (mode → anomaly → gated-localization) is the contribution.
Chapter 6 reports four severing ablations against the chained system, not a
bake-off of published architectures.

## Datasets in scope

| Dataset | Sensors | Folder labels | Spatial labels | Role |
|---|---|---|---|---|
| `data/first_test_dataset` | 4 mics + 4 vibration | Pump / Standstill / Turbine / RandomFault | none (synthetic geometry) | RQ1 + RQ2 |
| `data/second_test_dataset` | 5 mics + 5 vibration (`node_position.txt`) | …same + `pos_(x,y,z)_*` subfolders | YES, 5 positions | RQ1 + RQ2 + RQ3 |
| `data/third_test_dataset` | 9 mics (stereo pairs) + 4 accel (`position.json`) | speed1 / speed2 / speed3 + `hit_between_Fl_Gr_speed1` | YES, 1 hit | RQ2 (speed shift) + RQ3 + RQ4a |
| `data/illwerke_raw_stub` | future: 9 mics + 4 accel | TBD | TBD | drop-in via config when delivered |

Adding a new dataset is a YAML edit at `configs/datasets/<id>.yaml` — see
[`docs/ideal_prototype_dataset.md`](docs/ideal_prototype_dataset.md) for the
full collection spec.

## Architecture (target end-state)

```
   WAV / vibration  →  per-modality CNN encoders + Set-Transformer pool
                           (sensor-pos + modality + dataset embeddings)
                                       │
                                       ▼
                       Bidirectional cross-attention (1 block)
                                       │
                                       ▼
                  fused tokens z_t  →  c_t = PMA(z_t)        ── continuous mode label (cluster/Hungarian)
                                       │             │
                                       │      FiLM(c) ┐
                                       ▼              │
                       Conditional Normalizing Flow ◀─┘   ── continuous anomaly score s_t
                                       │
                                       ▼  (gate: s_t > per-cluster 99 % threshold)
                                       │
                       Cross3D 3-D CNN on SRP-PHAT + accel-TDOA, FiLM(c [+ s])
                                       ▼
                                    (x, y, z)            ── only on alert windows
```

## Iteration ladder (current state)

| Iter | What it delivers | Status |
|---|---|---|
| **V0**  | Reference baselines: LSTM-AE on log-mel (RQ2), LightGBM on hand-engineered features (RQ1 upper-bound), classical SRP-PHAT (RQ3) | LSTM-AE done · LightGBM done (smoke pending) · SRP-PHAT entry point pending |
| **V1**  | Per-modality SSL warmup (contrastive only) + cluster-purity sanity gate. Label-free. | pending |
| **V2**  | Bidirectional cross-attention fusion + multimodal SSL (contrastive + Latent Masked Modeling), inherits V1 weights | pending |
| **V3**  | Conditional Normalizing Flow anomaly head + per-cluster percentile thresholds + synthetic transition stress-test + A2 ablation | pending |
| **V4**  | Anomaly-gated Cross3D localization head + accel-TDOA + FiLM conditioning + A3 ablation | pending |
| **V5**  | RQ4a D3 speed conditioning + RQ4b Illwerke Allg_M1 MI ranking | pending |
| **streaming** | Gated runtime pipeline emitting `(mode, anomaly_score, alert_flag, (x,y,z) | None)` per window | pending |

## Layout

```
src/
├── config/               physical constants (sensor geometry, speeds)
├── data/                 DataSegment universal data contract
├── ingestion/
│   ├── loader.py / adapters.py / scanner.py   generic WAV+CSV reader
│   ├── positions.py                           per-dataset 3-D position registry
│   ├── test_dataset_loader.py                 unified loader (D1/D2/D3/illwerke)
│   ├── illwerke_loader.py / udbf_reader.py    Allg_M1 SCADA mining (V5.2)
├── features/
│   ├── audio_spectral.py                      log-mel + CWT V1 encoder input
│   ├── vibration_temporal.py                  amplitude + envelope + kurtosis
│   └── acoustic_representations.py            CWT / MFCC / STFT primitives
└── modeling/
    ├── localization/localization_head.py      GCC-PHAT, SRP-PHAT, Cross3D primitives (V4)
    └── anomaly_baselines/
        ├── lstm_ae.py                         V0 LSTM-AE on log-mel
        └── mode_lgbm.py                       V0 LightGBM mode classifier

configs/
├── datasets/{d1, d2, d3, illwerke_raw_stub}.yaml   per-dataset registration
└── test_datasets/v0_lstm_ae.yaml                   V0 LSTM-AE config

results/illwerke/    frozen Illwerke 5-layer pipeline outputs (V5.2 inputs)
docs/                Thesis.md + ideal_prototype_dataset.md
tests/               smoke tests
```

## Run the V0 smoke tests

```bash
python -m pytest tests/test_test_dataset_loader.py tests/test_v0_lstm_ae.py -v
```

## Notes on the previous repo state

A prior iteration of this repo implemented an Illwerke-specific 5-layer
physics pipeline + Plotly.js dashboard.  That code is preserved in commit
`51b77db` (`git checkout 51b77db -- <path>` to recover any file) and was
removed when the thesis architecture pivoted to the V0–V5 chained label-free
system above.  The Illwerke chapter results in `results/illwerke/` are kept
intact as frozen evidence and feed the V5.2 SCADA-mining analysis.
