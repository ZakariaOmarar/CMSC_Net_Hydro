# `scripts/` — runnable experiment drivers

Everything here is a **command-line entry point**, not importable library code
(the library lives under [`src/`](../src)). Each script either drives a
multi-stage experiment, sweeps a hyperparameter, produces a diagnostic, or is a
one-off utility. They are run from the repository root, e.g.

```bash
python -m scripts.run_thesis_campaign        # module form (preferred)
python scripts/derive_dataset_sampling_rate.py configs/datasets/d5.yaml
```

All of them import shared per-stage configuration from
[`src/modeling/orchestration/full_run.py`](../src/modeling/orchestration/full_run.py)
(`_v1_cfg`, `_v2_cfg`, …) so that scripts and the canonical pipeline never drift.

> **Reproducing the headline results does not require any script here.**
> The canonical run is `python -m src.modeling.orchestration.full_run`
> (smoke: `--quick`); `python -m src.modeling.orchestration.multi_seed`
> produces the mean ± std numbers in the thesis tables. The scripts below are
> the supporting investigations, sweeps, and diagnostics around that run.

---

## Campaign drivers (multi-stage)

Sequence many training runs into one end-to-end experiment.

| Script | Purpose |
|---|---|
| `run_thesis_campaign.py` | Master campaign: acoustic-improvement sweep → full pipeline → multi-seed verdict → report. One command, produces every result. |
| `run_ablation_campaign.py` | End-to-end ablation campaign (the 57-cell plan): baseline → per-phase sweeps → top-K promotion → multi-seed → conditional follow-ups. |
| `run_deep_v3v4_campaign.py` | V3-first deep campaign: deep V3 sweep → V3-gated deep V4 sweep → multi-seed verdict on the winners. |
| `ablation_full_pipeline.py` | Single-cell runner used by the campaigns: one parameter combination through the full pipeline. |
| `run_v1_v2_only.py` | Phase-B helper: retrain only V1+V2 (+A1+modality probe) under one named intervention (~50 min CPU vs ~6 h full). |
| `analyze_ablation.py` | Aggregates campaign cell outputs (`results/runs/*__ablation_*/`) into a Markdown report. |

## Paradigm comparisons (RQ2 / RQ3)

| Script | Purpose |
|---|---|
| `run_v3_three_paradigms.py` | Trains V3 acoustic / vibration / fusion CNF heads from saved V1+V2 weights; emits comparable per-pipeline metrics. |
| `run_v4_three_paradigms.py` | Trains the V4 localization paradigms (SRP-PHAT, accel-multilateration, learned heads) for the RQ3 comparison. |

## Hyperparameter sweeps

| Script | Purpose |
|---|---|
| `v3_deep_sweep.py` | Deep V3 anomaly sweep against a frozen V1/V2 encoder; selects by gap-guarded real-anomaly F1. |
| `v4_deep_sweep.py` | Deep V4 localization sweep against a frozen V2 encoder; evaluates on held-out positions, V3-gated. |
| `v4_aug_sweep.py` | V4-only augmentation sweep (target-position noise × SRP-volume noise) reusing one cached set of V4 samples. |

## Diagnostics

One-off investigations. Relocated here from `src/modeling/orchestration/`
during the thesis-submission cleanup — they are run, never imported.

| Script | Purpose |
|---|---|
| `probe_v2.py` | Probes V2's internal representations to locate where the acoustic↔vibration fusion loses cluster purity. |
| `reeval_k3.py` | Re-evaluates trained V2 encoders under the corrected K=3 (Pump/Standstill/Turbine) held-out setup, no retraining. |
| `train_v2_cma.py` | Trains V2 with the cross-modal-alignment loss and compares RQ1 purity against vanilla V2. |
| `v2_sweep.py` | V2 architectural sweep: CMA-weight grid × context-aggregation variants. |
| `v3_diagnostic.py` | Dumps per-cohort V3 anomaly-score distributions for the Chapter 6 histograms. |

## Utilities

| Script | Purpose |
|---|---|
| `derive_dataset_sampling_rate.py` | Canonical way to fill a new dataset's `accel_target_sr` from raw CSV timestamps. `--apply` writes it back into the YAML. |
| `visualize_sensor_knock_positions.py` | Plots sensor + knock positions for a dataset (figure helper). |

## Concluded studies (historical)

The acoustic-feature grid search is **settled**: it selected
`n_fft=4096, hop_length=2048, n_mels=96`, now hard-wired in
[`ACOUSTIC_FEATURES`](../src/config/architecture.py) (chapter 3 §3.4.2). These
scripts are retained for provenance and are not part of the live pipeline.
`analyze_hop_length_full_grid.py` is the canonical run that produced the
decision; the rest are earlier or narrower probes of the same question.

| Script | Role |
|---|---|
| `analyze_hop_length_full_grid.py` | **Canonical** full-grid (n_fft × hop × n_mels) sweep on all 5 datasets → `results/hop_grid_sweep.json`. |
| `analyze_hop_length_empirical.py` | Earlier evidence-driven hop analysis on real data (mode separability + knock SNR). |
| `ablation_hop_length.py` | Pre-registered hop-length ablation runner (synthetic-AUC protocol). |
| `analyze_hop_ablation.py` | Hypothesis test + effect sizes on the `ablation_hop_length.py` output. |
| `full_run_hop43.py` | Re-runs the full pipeline at the alternative `hop=43` condition for head-to-head comparison. |
| `compare_hop43_vs_baseline.py` | Side-by-side `hop=43` vs `hop=512` metric table. |
| `hop_comparison_d4.py` | Quick single-recording hop=512 vs hop=43 spot check on a D4 knock. |
