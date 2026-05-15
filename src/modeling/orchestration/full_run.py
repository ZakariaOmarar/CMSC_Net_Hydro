"""End-to-end orchestrator: V1 -> V2 -> V3 -> V4 -> V5 on real D1+D2+D3 data.

Designed for a single CPU machine, so the default hyperparameters trade
exhaustive convergence for tractable wall-clock runtime:
  - acoustic features: log-mel only (no CWT), 48 mels
  - epochs: V1=6, V2=6, V3=15, V4=30, V5.1=30
  - batch size: 16
  - window: 2 s with 1 s stride

Total expected runtime on a modern desktop CPU: ~30–60 minutes.

Persists everything under `results/full_run/`:
  - `v1/{acoustic,vibration}.pt`        — encoder state_dicts
  - `v2/{encoder,projection}.pt`         — fusion encoder + projection head
  - `v2/cluster_to_label.json`           — Hungarian mapping for streaming
  - `v3/{flow,thresholds}.{pt,npz}`      — CNF + per-cluster percentiles
  - `v4/head.pt`                         — V4 localization head
  - `v5_1/head_speed.pt`                 — V4 head with D3 speed SCADA slot
  - `metrics.json`                       — RQ1/RQ2/RQ3/RQ4 headline numbers

Run with:
    python -m src.modeling.orchestration.full_run [--quick]

`--quick` halves the epoch counts again for a smoke-level real-data run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader, TestDatasetSegment
from ...modeling.localization import localization_head as lh
from ..anomaly import (
    PerClusterThresholds,
    V3Config,
    gate_samples_by_alert,
    make_transition_segment,
    train_v3_cnf,
    transition_fpr,
)
from ..anomaly.v3_trainer import encoder_level_transition_fpr
from ..anomaly.synthetic_eval import evaluate_synthetic_anomaly_auc
from ..anomaly.threshold import per_cluster_alert_breakdown
from ..context.cluster_metric import cluster_purity_per_dataset
from ..context.modality_probe import run_modality_balance_probe
from ..eval import paired_bootstrap_test, percentile_bootstrap_ci
from ..anomaly_baselines import (
    SRPConfig,
    V0Config,
    V0ModeConfig,
    evaluate_srp_phat,
    summarise,
    train_v0_lstm_ae,
    train_v0_mode_lgbm,
)
from ..anomaly.v3_trainer import precompute_paired
from ..context.v1_ssl import V1SSLConfig, train_v1_per_modality
from ..context.v2_fusion import V2FusionEncoder
from ..context.v2_ssl import V2SSLConfig, _gather_paired_segments, train_v2_fusion
from ..localization import (
    GridSpec,
    V4Config,
    V4Sample,
    precompute_v4_samples,
    train_v4_localization,
)
from ..scada import d3_speed_lookup


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _resolved_loader(yaml_name: str) -> TestDatasetLoader:
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / yaml_name)
    spec = DatasetSpec(
        id=spec.id,
        root=REPO_ROOT / spec.root,
        n_mics=spec.n_mics,
        n_vibrations=spec.n_vibrations,
        accel_target_sr=spec.accel_target_sr,
        position_source=spec.position_source,
        label_scheme=spec.label_scheme,
        extra=spec.extra,
    )
    return TestDatasetLoader(spec)


def _all_segments(loaders: list[TestDatasetLoader]) -> list[TestDatasetSegment]:
    out: list[TestDatasetSegment] = []
    for L in loaders:
        out.extend(L.list_segments())
    return out


# ---------------------------------------------------------------------------
# Hyperparam profiles
# ---------------------------------------------------------------------------


def _v1_cfg(quick: bool) -> V1SSLConfig:
    return V1SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        feature_dim=64,
        embed_dim=64,
        n_heads=4,
        proj_dim=32,
        # Bumped 6→12 epochs for the full profile so c_t reaches a stable
        # mode-discriminative state — a stronger c_t directly improves
        # V3's per-cluster CNF density estimates and consequently the
        # quality of the unsupervised p95 threshold.  CPU wall-clock cost
        # ~ 25 min on V1 acoustic, ~ 20 min on V1 vibration.
        epochs=3 if quick else 12,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-5,
        temperature=0.1,
        val_ratio=0.3,
        n_mels=48,
        n_fft=1024,
        hop_length=512,
        cwt_n_scales=32,
        # CWT scalogram re-enabled for the publication run.  The plan's
        # smart-decisions table treats CWT as primary for non-stationary
        # mode-transition energy; it was disabled in the prior run only
        # for a CPU runtime budget.  Re-enabling it is the simplest single
        # unimodal lift available given V1-acoustic already leads at 0.727
        # purity on log-mel-only.  Wall-clock impact: V1 / V2 epoch ~ 1.7×.
        use_cwt=True,
        vib_kurtosis_window=5,
        gain_jitter_db=6.0,
        channel_dropout_p=0.2,
        spec_augment_freq_mask=6,
        spec_augment_time_mask=8,
        seed=42,
    )


def _v2_cfg(quick: bool) -> V2SSLConfig:
    return V2SSLConfig(
        window_seconds=2.0,
        window_stride_seconds=1.0,
        feature_dim=64,
        embed_dim=64,
        n_heads=4,
        proj_dim=32,
        # Bumped 6→12 epochs (matches V1).  With asymmetric modality dropout
        # (vibration_dropout_p=0.5), the fusion block sees only ~ 50 % of
        # the vibration stream per batch and needs the extra epochs to
        # converge.
        epochs=3 if quick else 12,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-5,
        temperature=0.1,
        val_ratio=0.3,
        n_mels=48,
        n_fft=1024,
        hop_length=512,
        cwt_n_scales=32,
        # CWT enabled — see _v1_cfg note.
        use_cwt=True,
        vib_kurtosis_window=5,
        gain_jitter_db=6.0,
        channel_dropout_p=0.2,
        spec_augment_freq_mask=6,
        spec_augment_time_mask=8,
        lmm_mask_p=0.3,
        lmm_weight=1.0,
        # Asymmetric modality dropout: acoustic is the strong mode-
        # discriminator (V1-acoustic purity 0.727 vs V1-vib 0.572 in the
        # prior run).  Dropping vibration twice as often as acoustic stops
        # the fusion block from diluting the acoustic signal — the
        # mechanism that produced V2 purity 0.612 < V1-acoustic 0.727.
        modality_dropout_p=0.0,  # legacy fallback off
        acoustic_dropout_p=0.0,
        vibration_dropout_p=0.5,
        # Two PMA seeds in the context pool — one summary is bottlenecked
        # for a 9–14-token fused sequence.
        num_context_seeds=2,
        seed=42,
    )


def _v3_cfg(quick: bool) -> V3Config:
    return V3Config(
        n_layers=6,
        hidden_dim=64,
        n_hidden_per_net=2,
        epochs=8 if quick else 15,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-5,
        val_ratio=0.3,
        unconditional=False,
        # K = 3 matches the 3-mode hypothesis (Pump / Standstill / Turbine);
        # see REVIEW.md fix (A).
        n_threshold_clusters=3,
        # p95 = lower-FPR-tolerant operating point that's still defensible
        # vs healthy variance; see REVIEW.md fix (C) and §3.4.6.
        threshold_percentile=95,
        seed=42,
    )


def _v4_cfg(quick: bool, scada_dim: int = 0, unconditional: bool = False) -> V4Config:
    return V4Config(
        cnn_feature_dim=64,
        tdoa_feature_dim=32,
        hidden_dim=64,
        n_heads_tdoa=2,
        scada_dim=scada_dim,
        unconditional=unconditional,
        # Soft-argmax + FiLM-residual head: most of the work is in the
        # 3-D CNN trunk, which is unchanged from the original Cross3D.
        # 30 epochs is enough for the residual to stabilise.
        epochs=15 if quick else 30,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-5,
        val_ratio=0.3,
        seed=42,
        # Soft-argmax / residual / loss / augmentation: see V4Config defaults.
        # Residual half-range = 20 cm (was 5 cm).  The 2026-05-11 retrain
        # showed soft-argmax centre-bias of 10–15 cm on corner-of-grid
        # ground-truth positions (e.g. D4 `(-20, 0, 0)`); the prior cap
        # could not correct it, producing 0.192 m MAE vs the prior dense
        # regressor's 0.160 m.  See REVIEW.md fourth-pass audit.
        residual_scale_m=0.20,
        soft_argmax_temperature=1.0,
        train_in_centimetres=True,
        smooth_l1_beta=1.0,  # = 1 cm in the post-scale unit
        target_pos_noise_m=0.002,
        srp_volume_noise_std=0.02,
        tdoa_jitter_m=0.001,
        augment=True,
    )


# ---------------------------------------------------------------------------
# Spatial-label derivation for D3 hits
# ---------------------------------------------------------------------------


def _d3_spatial_overrides(d3_segments: list[TestDatasetSegment]) -> dict[str, tuple[float, float, float]]:
    """Derive spatial labels for D3 `hit_between_*_speed*` recordings.

    Uses the ROW II reference constant `S3_HIT_FL_GR_APPROX_M` for the single
    `hit_between_Fl_Gr_*` family.  For other hit pairs we fall back to the
    midpoint of named accelerometer positions (`SENSOR_XYZ`) if available,
    otherwise skip.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for s in d3_segments:
        if "hit_between" not in s.recording_id and "hit_between" not in str(s.source_dir).lower():
            continue
        rec_lower = s.recording_id.lower()
        src_lower = str(s.source_dir).lower()
        if "fl" in rec_lower and "gr" in rec_lower:
            xyz = lh.S3_HIT_FL_GR_APPROX_M
            out[s.recording_id] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        elif "fl" in src_lower and "gr" in src_lower:
            xyz = lh.S3_HIT_FL_GR_APPROX_M
            out[s.recording_id] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    quick: bool = False,
    dataset_ids: tuple[str, ...] | None = None,
) -> dict:
    # Pin Python / NumPy / PyTorch RNGs at the top of `main` so the same
    # invocation produces the same V1 / V2 / V3 / V4 weights regardless of
    # whether the orchestrator was preceded by other RNG-consuming work in
    # the same Python process (e.g. the multi-seed driver, smoke tests).
    # PyTorch's `use_deterministic_algorithms(warn_only=True)` flips on
    # deterministic kernels where available; this still allows multi-
    # threaded BLAS (so per-epoch wall-clock is unchanged), which means
    # run-to-run variance from BLAS thread scheduling is NOT eliminated —
    # it is bounded.  For publication-grade numbers the
    # `multi_seed.py` driver remains the canonical mean ± std reporter.
    import os
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # older torch versions

    out_dir = REPO_ROOT / "results" / "full_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v1").mkdir(exist_ok=True)
    (out_dir / "v2").mkdir(exist_ok=True)
    (out_dir / "v3").mkdir(exist_ok=True)
    (out_dir / "v4").mkdir(exist_ok=True)
    (out_dir / "v5_1").mkdir(exist_ok=True)

    log_path = out_dir / "run_log.txt"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # Defend against Windows cp1252 stdout: drop characters it can't encode.
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        log_lines.append(line)

    log(f"REPO_ROOT = {REPO_ROOT}")
    log(f"quick mode: {quick}")

    metrics: dict = {"quick": quick, "stages": {}}

    # ------------------------------------------------------------------ data
    log("Loading D1, D2, D3, D4 loaders ...")
    D1 = _resolved_loader("d1.yaml")
    D2 = _resolved_loader("d2.yaml")
    D3 = _resolved_loader("d3.yaml")
    D4 = _resolved_loader("d4.yaml")
    _all_loaders = {"d1": D1, "d2": D2, "d3": D3, "d4": D4}
    if dataset_ids is None:
        SSL_LOADERS = [D1, D2, D3, D4]
        ANOM_LOADERS = [D1, D2, D3, D4]
    else:
        # Ablation: V1 / V2 / V3 SSL training restricted to these dataset IDs.
        # V0 baselines, V4 cohort, V5.1 SCADA still see the full set so the
        # downstream comparison rows are unaffected.
        log(f"Ablation: restricting SSL pool to {sorted(dataset_ids)}")
        SSL_LOADERS = [_all_loaders[did] for did in dataset_ids if did in _all_loaders]
        ANOM_LOADERS = SSL_LOADERS  # V3 healthy training inherits the same restriction
        if not SSL_LOADERS:
            raise RuntimeError(f"No valid datasets from {dataset_ids}")

    # =================================================================== V0
    # V0 baselines produce the reference numbers each subsequent iteration
    # must beat (per the plan).  One per RQ:
    #   - LSTM-AE on log-mel windows: RQ2 (anomaly) reference
    #   - LightGBM mode classifier:    RQ1 supervised upper-bound reference
    #   - Classical SRP-PHAT:          RQ3 (localization) reference
    # All three are trained per-dataset (no cross-campaign training); the
    # orchestrator collects the headline numbers into `metrics["stages"]`.
    # ──────────── Cross-modal sync verification AND correction ──────────
    # The current ingestion path aligns the WAV and the vibration CSV
    # by sample index ("WAV sample 0 ↔ vibration sample 0 at the
    # same wall-clock instant").  The bench-top prototype rig has
    # no NTP discipline, so this implicit alignment is **not
    # guaranteed by the hardware**.  We therefore:
    #
    #   1. Estimate the per-recording cross-modal offset via
    #      envelope cross-correlation with parabolic sub-sample
    #      refinement (Jacovitti & Scarano 1993).
    #   2. Verify the offset is **time-stable** across K = 5
    #      disjoint sub-segments of the recording (clock-drift
    #      check; Bregler & Konig 1998).
    #   3. **Apply** an integer-+-fractional-sample correction to
    #      whichever stream is lagging IFF confidence ≥ 1.5 AND
    #      offset spread ≤ 10 ms AND |offset| ≥ 1 ms.  Both gates
    #      protect against introducing error rather than reducing it.
    #   4. Persist per-recording corrections to `metrics.json` and
    #      apply the correction **in-place on every
    #      `DataSegment.mic_data` / `accel_data`** so every
    #      downstream stage (V0 through V5) consumes aligned data.
    log("Cross-modal sync verification + correction "
        "(parabolic-refined xcorr + stability gate; corrects in place) ...")
    try:
        from ...ingestion.sync_verification import auto_sync_paired_recording

        sync_metrics: dict = {}
        for L in SSL_LOADERS:
            ds_name = L.spec.id
            offsets_s: list[float] = []
            confidences: list[float] = []
            env_kurtoses: list[float] = []
            n_applied = 0
            n_rejected_low_conf = 0
            n_rejected_drift = 0
            n_rejected_below_floor = 0
            n_rejected_flat_envelope = 0
            n_skipped = 0
            for s in L.list_segments():
                # Audit / correct every healthy recording.  Anomaly
                # cohorts are corrected too — the cross-correlation
                # works whenever the two modalities share any
                # broadband event in the recording.
                try:
                    mic_corr, accel_corr, report = auto_sync_paired_recording(
                        s.segment.mic_data,
                        s.segment.accel_data,
                        mic_fs=float(s.segment.mic_sample_rate),
                        accel_fs=float(s.segment.accel_sample_rate),
                        max_offset_s=0.5,
                        n_sub_segments=5,
                        confidence_floor=1.5,
                        drift_tolerance_s=0.010,
                        min_offset_to_correct_s=0.001,
                        use_fractional_shift=True,
                    )
                    if not np.isnan(report.audit.offset_s):
                        offsets_s.append(float(report.audit.offset_s))
                        confidences.append(float(report.audit.confidence))
                    if not np.isnan(report.acoustic_envelope_kurtosis):
                        env_kurtoses.append(float(report.acoustic_envelope_kurtosis))
                    if report.applied:
                        # Mutate the segment so every downstream
                        # stage consumes the corrected data.  After
                        # the drop, the two streams may have slightly
                        # different durations; the loader's existing
                        # `common_duration` truncation downstream
                        # rounds both to the shorter one.
                        s.segment.mic_data = mic_corr
                        s.segment.accel_data = accel_corr
                        n_applied += 1
                    elif "stability" in report.reason.lower() or "drift" in report.reason.lower():
                        n_rejected_drift += 1
                    elif "near-gaussian" in report.reason.lower():
                        # F3 Gate 0 — genuinely flat acoustic envelope.
                        n_rejected_flat_envelope += 1
                    elif "confidence" in report.reason.lower() or "uninformative" in report.reason.lower():
                        n_rejected_low_conf += 1
                    elif "below" in report.reason.lower() or "already aligned" in report.reason.lower():
                        n_rejected_below_floor += 1
                except Exception as ee:
                    n_skipped += 1
            if offsets_s:
                arr = np.asarray(offsets_s)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                mean_conf = float(np.mean(confidences))
                med_kurt = float(np.median(env_kurtoses)) if env_kurtoses else float("nan")
                n_total = (
                    n_applied + n_rejected_low_conf + n_rejected_drift
                    + n_rejected_below_floor + n_rejected_flat_envelope + n_skipped
                )
                log(
                    f"  {ds_name}: n={n_total} recordings  "
                    f"median offset = {med * 1e3:+.1f} ms (MAD ± {mad * 1e3:.1f} ms)  "
                    f"mean confidence = {mean_conf:.2f}  "
                    f"median env-kurtosis = {med_kurt:.2f}  "
                    f"applied={n_applied}  drift={n_rejected_drift}  "
                    f"low_conf={n_rejected_low_conf}  flat_env={n_rejected_flat_envelope}  "
                    f"below_floor={n_rejected_below_floor}  skipped={n_skipped}"
                )
                sync_metrics[ds_name] = {
                    "n_recordings_total": n_total,
                    "n_corrections_applied": n_applied,
                    "n_rejected_drift": n_rejected_drift,
                    "n_rejected_low_confidence": n_rejected_low_conf,
                    "n_rejected_flat_envelope": n_rejected_flat_envelope,
                    "n_rejected_below_floor": n_rejected_below_floor,
                    "n_skipped": n_skipped,
                    "median_offset_s": med,
                    "median_absolute_deviation_s": mad,
                    "min_offset_s": float(np.min(arr)),
                    "max_offset_s": float(np.max(arr)),
                    "mean_confidence": mean_conf,
                    # F3 diagnostic — distinguishes "genuinely steady-state"
                    # (low kurtosis) from "transient content that doesn't
                    # cross-correlate" (high kurtosis, low confidence).
                    "median_acoustic_envelope_kurtosis": med_kurt,
                    "per_recording_offsets_s": offsets_s,
                    "per_recording_confidences": confidences,
                    "per_recording_envelope_kurtosis": env_kurtoses,
                }
            else:
                log(f"  {ds_name}: no auditable recordings (n_skipped={n_skipped})")
                sync_metrics[ds_name] = {
                    "n_recordings_total": 0,
                    "n_skipped": n_skipped,
                }
        metrics["stages"]["sync_correction"] = sync_metrics
    except Exception as e:
        log(f"Sync verification + correction skipped: {type(e).__name__}: {e}")
        metrics["stages"]["sync_correction"] = {
            "skipped_reason": f"{type(e).__name__}: {e}"
        }

    log("V0 — running baseline references ...")
    v0_metrics: dict = {}
    for L in SSL_LOADERS:
        ds_name = L.spec.id
        # V0 LightGBM mode classifier: only datasets with explicit mode
        # labels (D1, D2).  D3 / D4 have unrecorded modes per protocol.
        if ds_name in ("d1", "d2"):
            try:
                t0 = time.time()
                lgbm_result = train_v0_mode_lgbm(L, V0ModeConfig())
                log(
                    f"  V0 LGBM mode ({ds_name}) done in {time.time() - t0:.1f}s — "
                    f"val macro-F1 = {lgbm_result.val_macro_f1:.3f}"
                )
                v0_metrics[f"v0_lgbm_{ds_name}"] = {
                    "val_macro_f1": float(lgbm_result.val_macro_f1),
                    "val_per_class_f1": {
                        str(k): float(v) for k, v in lgbm_result.val_per_class_f1.items()
                    },
                    "n_train_recordings": len(lgbm_result.train_recording_ids),
                    "n_val_recordings": len(lgbm_result.val_recording_ids),
                }
            except Exception as e:
                log(f"  V0 LGBM mode ({ds_name}) skipped: {e}")
                v0_metrics[f"v0_lgbm_{ds_name}"] = {"skipped": str(e)}
        # V0 LSTM-AE anomaly baseline: trained on healthy windows only,
        # scored on anomaly recordings (each campaign separately).
        try:
            t0 = time.time()
            v0_ae = train_v0_lstm_ae(L, V0Config())
            log(
                f"  V0 LSTM-AE ({ds_name}) done in {time.time() - t0:.1f}s — "
                f"val recon MSE = {v0_ae.val_loss_history[-1]:.4f}"
            )
            v0_metrics[f"v0_lstm_ae_{ds_name}"] = {
                "val_loss_final": float(v0_ae.val_loss_history[-1]),
                "n_train_recordings": len(v0_ae.healthy_train_recordings),
                "n_val_recordings": len(v0_ae.healthy_val_recordings),
            }
        except Exception as e:
            log(f"  V0 LSTM-AE ({ds_name}) skipped: {e}")
            v0_metrics[f"v0_lstm_ae_{ds_name}"] = {"skipped": str(e)}
        # V0 classical SRP-PHAT: only datasets that have spatial labels.
        if ds_name in ("d2", "d3", "d4"):
            try:
                records = evaluate_srp_phat(L, SRPConfig())
                summary = summarise(records)
                log(
                    f"  V0 SRP-PHAT ({ds_name}): {summary['n_recordings']} "
                    f"recordings, mean MAE = {summary['mean_error_m']:.3f} m"
                )
                v0_metrics[f"v0_srp_phat_{ds_name}"] = summary
            except Exception as e:
                log(f"  V0 SRP-PHAT ({ds_name}) skipped: {e}")
                v0_metrics[f"v0_srp_phat_{ds_name}"] = {"skipped": str(e)}
    metrics["stages"]["v0"] = v0_metrics

    # =================================================================== V1
    v1_cfg = _v1_cfg(quick)
    log(f"V1 config: epochs={v1_cfg.epochs}, n_mels={v1_cfg.n_mels}, use_cwt={v1_cfg.use_cwt}")

    def _fmt_gate(g: dict) -> str:
        # NMI is the Chapter 6 headline; ARI is the chance-corrected
        # secondary; purity is retained for direct Khamaisi-style
        # comparison.  See `cluster_metric.py` module docstring.
        nmi = g.get("nmi", 0.0)
        ari = g.get("ari", 0.0)
        purity = g.get("purity", 0.0)
        ls = g.get("label_set", ())
        n_w = g.get("n_windows", 0)
        return (
            f"NMI={nmi:.3f} ARI={ari:.3f} purity={purity:.3f} "
            f"(K={g.get('n_clusters', 0)} clusters, "
            f"labels={ls}, n_val_windows={n_w})"
        )

    log("V1 acoustic — training on D1+D2+D3+D4 healthy windows ...")
    t0 = time.time()
    v1_acoustic = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    log(f"V1 acoustic done in {time.time() - t0:.1f}s — sanity gate: {_fmt_gate(v1_acoustic.sanity_gate)}")
    torch.save(v1_acoustic.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_acoustic.train_loss_history[-1],
        "val_loss_final": v1_acoustic.val_loss_history[-1],
        "sanity_nmi": v1_acoustic.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_acoustic.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_acoustic.sanity_gate.get("purity", 0.0),
        "sanity_n_windows": v1_acoustic.sanity_gate.get("n_windows", 0),
        "sanity_label_set": list(v1_acoustic.sanity_gate.get("label_set", ())),
        "n_train_recordings": len(v1_acoustic.train_recording_ids),
        "n_val_recordings": len(v1_acoustic.val_recording_ids),
    }

    log("V1 vibration — training on D1+D2+D3+D4 healthy windows ...")
    t0 = time.time()
    v1_vibration = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    log(f"V1 vibration done in {time.time() - t0:.1f}s — sanity gate: {_fmt_gate(v1_vibration.sanity_gate)}")
    torch.save(v1_vibration.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_vibration.train_loss_history[-1],
        "val_loss_final": v1_vibration.val_loss_history[-1],
        "sanity_nmi": v1_vibration.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_vibration.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_vibration.sanity_gate.get("purity", 0.0),
        "sanity_label_set": list(v1_vibration.sanity_gate.get("label_set", ())),
    }

    # =================================================================== V2
    v2_cfg = _v2_cfg(quick)
    log("V2 — training fusion (inherits V1 weights) on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v2 = train_v2_fusion(
        SSL_LOADERS,
        cfg=v2_cfg,
        v1_acoustic_state_dict=v1_acoustic.encoder.state_dict(),
        v1_vibration_state_dict=v1_vibration.encoder.state_dict(),
    )
    log(f"V2 done in {time.time() - t0:.1f}s — RQ1 sanity: {_fmt_gate(v2.rq1)}")
    torch.save(v2.encoder.state_dict(), out_dir / "v2" / "encoder.pt")
    torch.save(v2.projection.state_dict(), out_dir / "v2" / "projection.pt")
    metrics["stages"]["v2"] = {
        "epochs": v2_cfg.epochs,
        "train_loss_final": v2.train_loss_history[-1],
        "val_loss_final": v2.val_loss_history[-1],
        "train_simclr_final": v2.train_simclr_history[-1],
        "train_lmm_final": v2.train_lmm_history[-1],
        "rq1_nmi": v2.rq1.get("nmi", 0.0),
        "rq1_ari": v2.rq1.get("ari", 0.0),
        "rq1_purity": v2.rq1.get("purity", 0.0),
        "rq1_n_windows": v2.rq1.get("n_windows", 0),
        "rq1_label_set": list(v2.rq1.get("label_set", ())),
    }

    # ------- A1 ablation: drop vibration ----------------------------------
    log("V2 — A1 ablation (drop_vibration=True) ...")
    a1_cfg = V2SSLConfig(**{**asdict(v2_cfg), "drop_vibration": True})
    t0 = time.time()
    v2_a1 = train_v2_fusion(
        SSL_LOADERS,
        cfg=a1_cfg,
        v1_acoustic_state_dict=v1_acoustic.encoder.state_dict(),
        v1_vibration_state_dict=v1_vibration.encoder.state_dict(),
    )
    log(f"V2 A1 done in {time.time() - t0:.1f}s — sanity: {_fmt_gate(v2_a1.rq1)}")

    # ------- RQ1 depth: modality-balance probe (inference-time) -----------
    # Defends the asymmetric-modality-dropout design choice by measuring
    # how V2's cluster-purity degrades when one modality is masked at
    # inference.  Cited as RQ1 depth in REVIEW.md sixth-pass audit.
    try:
        from ..context.v2_ssl import _gather_labeled_segments
        labeled_segs = _gather_labeled_segments(SSL_LOADERS, v2_cfg)
        probe = run_modality_balance_probe(
            v2.encoder, labeled_segs,
            v2_cfg=v2_cfg,
            n_clusters=v3_cfg.n_threshold_clusters if False else 3,
            seed=v2_cfg.seed,
        )
        log(
            f"V2 modality-balance probe: "
            f"both NMI={probe.both.get('nmi', 0.0):.3f}, "
            f"acoustic-only NMI={probe.acoustic_only.get('nmi', 0.0):.3f}, "
            f"vibration-only NMI={probe.vibration_only.get('nmi', 0.0):.3f} "
            f"(n_windows={probe.both.get('n_windows', 0)})"
        )
        metrics["stages"]["v2_modality_probe"] = {
            "both": {k: v for k, v in probe.both.items() if k != "confusion" and k != "cluster_idx"},
            "acoustic_only": {k: v for k, v in probe.acoustic_only.items() if k != "confusion" and k != "cluster_idx"},
            "vibration_only": {k: v for k, v in probe.vibration_only.items() if k != "confusion" and k != "cluster_idx"},
            "n_segments": len(probe.healthy_segments_used),
        }
    except Exception as e:
        log(f"V2 modality-balance probe skipped: {e}")

    metrics["stages"]["v2_a1_drop_vibration"] = {
        "rq1_nmi": v2_a1.rq1.get("nmi", 0.0),
        "rq1_ari": v2_a1.rq1.get("ari", 0.0),
        "rq1_purity": v2_a1.rq1.get("purity", 0.0),
    }

    # =================================================================== V3
    v3_cfg = _v3_cfg(quick)
    log("V3 — training conditional CNF on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v3 = train_v3_cnf(v2.encoder, ANOM_LOADERS, v2_cfg=v2_cfg, v3_cfg=v3_cfg)
    log(f"V3 done in {time.time() - t0:.1f}s — final NLL: {v3.val_nll[-1]:.3f}")
    torch.save(v3.flow.state_dict(), out_dir / "v3" / "flow.pt")
    np.savez(
        out_dir / "v3" / "thresholds.npz",
        centroids=v3.thresholds.centroids,
        p95=v3.thresholds.p95,
        p99=v3.thresholds.p99,
        n_per_cluster=v3.thresholds.n_per_cluster,
    )
    metrics["stages"]["v3"] = {
        "epochs": v3_cfg.epochs,
        "train_nll_final": v3.train_nll[-1],
        "val_nll_final": v3.val_nll[-1],
        # F6 — outlier-batch diagnostics.  A healthy training curve has
        # `train_nll_max ≲ 3 × train_nll_final` per epoch; sustained spread
        # above that ratio indicates the gradient clip is firing routinely
        # and the affine couplings are saturating somewhere.
        "train_nll_min_final": v3.train_nll_min[-1],
        "train_nll_max_final": v3.train_nll_max[-1],
        "val_nll_min_final": v3.val_nll_min[-1],
        "val_nll_max_final": v3.val_nll_max[-1],
        "n_clusters_fit": int(v3.thresholds.centroids.shape[0]),
        "p99_per_cluster": v3.thresholds.p99.tolist(),
        "n_val_windows_healthy": int(v3.val_scores.shape[0]),
        "n_threshold_fit_recordings": len(v3.threshold_fit_recording_ids),
        "n_val_eval_recordings": len(v3.val_recording_ids),
    }

    # ------- V3 RQ2: synthetic transition stress-test ---------------------
    # Two cohorts: (a) within-D1 mode pairs (Pump↔Turbine) — the original
    # narrow stress-test; (b) cross-dataset same-mode pairs (D1 Pump → D2
    # Pump, D1 Turbine → D2 Turbine) — exercises the c_t coherence claim
    # across acquisition campaigns, since the unit is in the same operating
    # mode in both halves but the recording rig is different.  Both are
    # healthy-by-construction crossfades; alert FPR should sit at or below
    # the threshold percentile (≈ 5 % at p95).
    log("V3 — synthetic transition stress-test (within-D1 + cross-dataset same-mode) ...")
    # Restrict the crossfade source pool to V3's reportable val cohort
    # (`v3.val_recording_ids`, i.e. the val_eval split).  This cohort is
    # disjoint from both V3 flow training and V3 threshold-fitting, so the
    # transition FPR is on truly held-out data.  Fall back to any healthy
    # recording only if val_eval contains no recording for a given (dataset,
    # mode) pair — in that fallback case we log a leakage warning.
    _val_eval_set = set(v3.val_recording_ids)

    def _qualifier(seg) -> str:
        return f"{Path(seg.source_dir).name}/{seg.recording_id}"

    paired_by_dataset_and_mode: dict[tuple[str, str], list] = {}
    paired_by_dataset_and_mode_fallback: dict[tuple[str, str], list] = {}
    for L in (D1, D2):
        for s in L.list_segments():
            if s.mode_label is None or s.is_anomaly:
                continue
            p = precompute_paired(s, v2_cfg)
            if p is None:
                continue
            key = (s.dataset_id, s.mode_label)
            if _qualifier(s) in _val_eval_set:
                paired_by_dataset_and_mode.setdefault(key, []).append(p)
            else:
                paired_by_dataset_and_mode_fallback.setdefault(key, []).append(p)
    # If a (dataset, mode) pair has no val_eval recording at all, fall back to
    # the broader healthy pool with a logged caveat — this is rare on D1/D2.
    for key, segs in paired_by_dataset_and_mode_fallback.items():
        if key not in paired_by_dataset_and_mode:
            log(
                f"  transition picker: no val_eval recording for {key}; "
                f"falling back to training-pool recording (NOT fully held-out)"
            )
            paired_by_dataset_and_mode[key] = segs

    # Two transition modes — within-dataset uses raw-feature crossfade (the
    # original `make_transition_segment`); cross-dataset uses encoder-level
    # crossfade in c_t / x space (channel-count agnostic).
    pairs: list[tuple[str, tuple[str, str], tuple[str, str], str]] = [
        # within-D1 mode transitions (mode-changes the model has never seen)
        ("d1_pump_to_turbine", ("d1", "Pump"),    ("d1", "Turbine"), "raw"),
        ("d1_turbine_to_pump", ("d1", "Turbine"), ("d1", "Pump"),    "raw"),
        # cross-dataset same-mode (test c_t coherence across acquisition rigs)
        ("d1_to_d2_pump",      ("d1", "Pump"),    ("d2", "Pump"),    "encoder"),
        ("d1_to_d2_turbine",   ("d1", "Turbine"), ("d2", "Turbine"), "encoder"),
    ]
    transition_results: dict[str, float] = {}
    for label, key_a, key_b, level in pairs:
        if key_a not in paired_by_dataset_and_mode or key_b not in paired_by_dataset_and_mode:
            log(f"  transition {label}: skipped (missing source recordings)")
            continue
        try:
            seg_a = paired_by_dataset_and_mode[key_a][0]
            seg_b = paired_by_dataset_and_mode[key_b][0]
            if level == "raw":
                fpr_out = transition_fpr(
                    v2.encoder, v3.flow, v3.thresholds,
                    seg_a, seg_b,
                    v2_cfg=v2_cfg, crossfade_seconds=1.0,
                    percentile=v3_cfg.threshold_percentile,
                )
            else:
                fpr_out = encoder_level_transition_fpr(
                    v2.encoder, v3.flow, v3.thresholds,
                    seg_a, seg_b,
                    v2_cfg=v2_cfg, n_crossfade_windows=8,
                    percentile=v3_cfg.threshold_percentile,
                )
            transition_results[label] = fpr_out["fpr"]
            log(
                f"  transition {label} ({level}): fpr={fpr_out['fpr']:.3f} "
                f"({fpr_out['n_alerts']}/{fpr_out['n_windows']})"
            )
        except Exception as e:
            log(f"  transition {label} skipped: {e}")
    metrics["stages"]["v3_rq2_transition_fpr"] = transition_results

    # ------- A2 ablation: unconditional flow ------------------------------
    log("V3 — A2 ablation (unconditional flow) ...")
    a2_cfg = V3Config(**{**asdict(v3_cfg), "unconditional": True})
    t0 = time.time()
    v3_a2 = train_v3_cnf(v2.encoder, ANOM_LOADERS, v2_cfg=v2_cfg, v3_cfg=a2_cfg)
    log(f"V3 A2 done in {time.time() - t0:.1f}s — final NLL: {v3_a2.val_nll[-1]:.3f}")
    metrics["stages"]["v3_a2_unconditional"] = {
        "val_nll_final": v3_a2.val_nll[-1],
        "p99_per_cluster": v3_a2.thresholds.p99.tolist(),
    }

    # ------- RQ2 statistical significance: paired bootstrap V3 vs A2 -----
    # Per-window NLL is computed on the *same* held-out healthy windows
    # under both flows, so a paired bootstrap on Δ_NLL is the rigorous
    # answer to "is the conditional flow significantly sharper than the
    # unconditional?".  Cite: Efron & Tibshirani 1993 §16 (paired bootstrap).
    try:
        if v3.val_scores.shape[0] == v3_a2.val_scores.shape[0] and v3.val_scores.shape[0] >= 4:
            paired_v3 = paired_bootstrap_test(
                v3.val_scores, v3_a2.val_scores,
                lower_is_better=True, n_boot=1000, seed=v3_cfg.seed,
            )
            log(
                f"V3 vs A2 (paired bootstrap on per-window NLL): "
                f"Δ={paired_v3.delta_point:.3f} "
                f"[95% CI {paired_v3.delta_ci_low:.3f}, {paired_v3.delta_ci_high:.3f}], "
                f"p={paired_v3.p_value_two_sided:.4f}, "
                f"direction={paired_v3.direction}"
            )
            metrics["stages"]["v3_vs_a2_paired_test"] = {
                "delta_point": paired_v3.delta_point,
                "delta_ci95_low": paired_v3.delta_ci_low,
                "delta_ci95_high": paired_v3.delta_ci_high,
                "p_value_two_sided": paired_v3.p_value_two_sided,
                "direction": paired_v3.direction,
                "n_paired": int(v3.val_scores.shape[0]),
                "method": "paired_percentile_bootstrap_1000",
            }
    except Exception as e:
        log(f"V3 vs A2 paired test skipped: {e}")

    # ------- RQ2 depth: synthetic-anomaly ROC-AUC -------------------------
    # Real ROC curve at calibrated SNRs.  Defends the V3 RQ2 claim with a
    # classical AUC number on a controlled cohort, complementing the
    # alert-rate-only validation (which is limited by the absence of
    # per-window anomaly labels in the field-collection protocol).
    # Cite: Kawaguchi et al., DCASE 2021; Hanley & McNeil 1982 for AUC.
    try:
        if v3.val_contexts.shape[0] >= 4:
            # V3Result carries val_scores + val_contexts; the synthetic-AUC
            # path also needs val_x (the mean-pool feature V3 takes as
            # input), which is not cached on V3Result.  Recompute via a
            # single encoder pass over the same val split V3 used.
            from ..anomaly.v3_trainer import _extract_xc
            from ..context.v2_ssl import (
                _PairedGroupedBatchSampler, _PairedWindowedDataset, _collate,
                _split_segments_by_recording, _gather_paired_segments,
            )
            import torch.utils.data as _tud
            segs_all = _gather_paired_segments(ANOM_LOADERS, v2_cfg)
            # Match V3's nested split so the synthetic-AUC cohort is exactly
            # the same val_eval recordings V3 reports on — disjoint from
            # both flow training and threshold fitting.
            _train_segs, _val_segs_full = _split_segments_by_recording(
                segs_all, v3_cfg.val_ratio, v3_cfg.seed
            )
            _val_fit_segs, val_segs_for_auc = _split_segments_by_recording(
                _val_segs_full, v3_cfg.threshold_fit_val_ratio, v3_cfg.seed + 1
            )
            val_ds = _PairedWindowedDataset(val_segs_for_auc, v2_cfg)
            if len(val_ds) > 0:
                val_loader = _tud.DataLoader(
                    val_ds,
                    batch_sampler=_PairedGroupedBatchSampler(
                        val_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed
                    ),
                    collate_fn=_collate,
                )
                x_val, c_val, _ = _extract_xc(v2.encoder, val_loader, torch.device("cpu"))
                auc_result = evaluate_synthetic_anomaly_auc(
                    v3.flow, x_val.numpy(), c_val.numpy(),
                    snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                    n_boot=500, seed=v3_cfg.seed,
                )
                auc_a2_result = evaluate_synthetic_anomaly_auc(
                    v3_a2.flow, x_val.numpy(), c_val.numpy(),
                    snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                    n_boot=500, seed=v3_cfg.seed,
                )
                log("V3 synthetic-anomaly ROC-AUC (V3 conditional vs A2 unconditional):")
                for snr in sorted(auc_result.snr_db_to_auc):
                    a_v3 = auc_result.snr_db_to_auc[snr]
                    a_a2 = auc_a2_result.snr_db_to_auc[snr]
                    log(f"  SNR={snr:+.1f} dB: V3={a_v3:.3f}, A2={a_a2:.3f}, Δ={a_v3 - a_a2:+.3f}")
                metrics["stages"]["v3_synthetic_anomaly_auc"] = {
                    "snr_db_list": list(auc_result.snr_db_to_auc.keys()),
                    "auc_conditional": auc_result.snr_db_to_auc,
                    "auc_conditional_ci_low": auc_result.snr_db_to_auc_ci_low,
                    "auc_conditional_ci_high": auc_result.snr_db_to_auc_ci_high,
                    "auc_unconditional": auc_a2_result.snr_db_to_auc,
                    "auc_unconditional_ci_low": auc_a2_result.snr_db_to_auc_ci_low,
                    "auc_unconditional_ci_high": auc_a2_result.snr_db_to_auc_ci_high,
                    "n_clean": auc_result.snr_db_to_n_clean,
                    "method": "additive_isotropic_gaussian_latent_perturbation",
                }
    except Exception as e:
        log(f"V3 synthetic-anomaly AUC skipped: {e}")

    # =================================================================== V4
    log("V4 — gathering labeled anomaly segments ...")
    # Prototype-scale SRP-PHAT grid: bounded by the max extent of the labeled
    # positions across D2 + D3 + D4 (roughly -20 cm to 40 cm on the longest
    # axis).  Tightened from 24×24×12 → 32×32×16 over the labeled bounding
    # box for ~ 1.9 cm voxel resolution (was ~ 3 cm); the 10-cm prototype
    # casing demands sub-cm resolution to make the SRP-PHAT peak useful.
    grid = GridSpec(lo=(-0.22, -0.22, -0.02), hi=(0.40, 0.42, 0.30), n=(32, 32, 16))
    # D2 supervision: only single-mode RandomFault recordings (the loader's
    # `mode_label is not None` filter drops the multi-mode `_turbine_pump`
    # folders the user flagged as too noisy — see Q2 in the design discussion).
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    log(f"  D2 labeled segments (single-mode RandomFault): {len(d2_labeled)}")
    log(f"  D3 labeled segments (via approx midpoint): {len(d3_labeled)}")
    log(f"  D4 labeled segments (raw RandomFault positions): {len(d4_labeled)}")

    log("V4 — precomputing candidate samples (burst-aware SRP-PHAT volumes + accel-TDOA tokens + c_t) ...")
    # Burst-aware SRP: cropping to the highest-energy ~100 ms sub-window
    # before GCC-PHAT sharpens the SRP peak on D4 sparse knocks (where the
    # knock occupies < 50 ms inside a 2 s window) and is no worse on D2 /
    # D3 continuously-anomalous recordings — see REVIEW.md third-pass audit.
    t0 = time.time()
    v4_samples_all = precompute_v4_samples(
        v2.encoder,
        d2_labeled + d3_labeled + d4_labeled,
        v2_cfg=v2_cfg,
        grid=grid,
        spatial_label_overrides=overrides,
        burst_aware_srp=True,
        burst_seconds=0.10,
    )
    log(f"  {len(v4_samples_all)} candidate V4 samples in {time.time() - t0:.1f}s")

    # ── V3 threshold validation: per-cohort alert rates ───────────────
    # Calibration of V3 thresholds is unsupervised on healthy data alone
    # (per-cluster p95 / p99), because the field-collection protocol does
    # not provide per-window anomaly labels.  We validate the threshold
    # quality post-hoc by reporting alert rates on each cohort:
    #
    #   - Healthy hold-out (D1+D2 mode folders + D3/D4 speed buckets):
    #     alert rate should sit at ≈ 5 % (the construction of p95).
    #   - D2 RandomFault / D3 hit (anomaly-containing recordings, density
    #     unknown but the user's protocol notes they are mostly anomalous):
    #     alert rate should sit substantially above 5 %.
    #   - D4 RandomFault (sparsely anomalous): alert rate is necessarily
    #     bounded below 100 % and reflects per-window event density rather
    #     than recording-level anomaly status.
    #
    # We report these rates without making absolute precision/recall
    # claims — anomaly density is unknown, so only the *ranking* of alert
    # rates across cohorts is interpretable.
    v3_cohort_validation: dict = {}

    def _alert_rate_for_samples(samples: list, label: str) -> dict:
        if not samples:
            v3_cohort_validation[label] = {"n": 0, "alert_rate": float("nan")}
            return v3_cohort_validation[label]
        xs = torch.from_numpy(np.stack([s.x_for_v3 for s in samples], axis=0))
        cs_np = np.stack([s.context for s in samples], axis=0)
        with torch.no_grad():
            scores = v3.flow.anomaly_score(xs, torch.from_numpy(cs_np)).numpy()
        alerts, _ = v3.thresholds.alert(
            cs_np, scores, percentile=v3_cfg.threshold_percentile
        )
        out = {
            "n": int(len(samples)),
            "alert_rate": float(alerts.mean()),
            "score_mean": float(scores.mean()),
            "score_p50": float(np.percentile(scores, 50)),
            "score_p95": float(np.percentile(scores, 95)),
        }
        v3_cohort_validation[label] = out
        return out

    log(
        f"V3 threshold validation (p={v3_cfg.threshold_percentile} unsupervised, "
        "per-cohort alert rates):"
    )

    # Held-out healthy alert rate: by construction of the per-cluster p95
    # threshold, this should be ≈ (100 − threshold_percentile) / 100 ≈ 5 %.
    # Departure from that target indicates either a held-out healthy
    # distribution that does not match the training healthy distribution
    # (unrepresentative split) or a CNF that is overfitting to its
    # training subset.  Either way the cohort-ranking rows below are
    # interpretable only when this row sits at its target.  Plan §V3
    # verification calls this "the canonical healthy-baseline row".
    if v3.val_scores.size > 0:
        healthy_alerts, _ = v3.thresholds.alert(
            v3.val_contexts, v3.val_scores, percentile=v3_cfg.threshold_percentile
        )
        healthy_rate = float(healthy_alerts.mean())
        healthy_n = int(v3.val_scores.size)
        target_rate = (100 - int(v3_cfg.threshold_percentile)) / 100.0
        delta = healthy_rate - target_rate
        v3_cohort_validation["healthy_holdout"] = {
            "n": healthy_n,
            "alert_rate": healthy_rate,
            "target_rate": target_rate,
            "delta_from_target": delta,
            "score_p50": float(np.percentile(v3.val_scores, 50)),
            "score_p95": float(np.percentile(v3.val_scores, 95)),
        }
        log(
            f"  {'healthy_holdout':<22} n={healthy_n:>5} alert_rate={healthy_rate:.3f} "
            f"(target={target_rate:.3f}, delta={delta:+.3f}, "
            f"score p50={np.percentile(v3.val_scores, 50):.2f}, "
            f"p95={np.percentile(v3.val_scores, 95):.2f})"
        )

    cohort_buckets = {
        "d2_random_fault": [s for s in v4_samples_all if s.dataset_id == "d2"],
        "d3_hit": [s for s in v4_samples_all if s.dataset_id == "d3"],
        "d4_random_fault": [s for s in v4_samples_all if s.dataset_id == "d4"],
    }
    for label, samples in cohort_buckets.items():
        out = _alert_rate_for_samples(samples, label)
        log(
            f"  {label:<22} n={out['n']:>5} alert_rate={out['alert_rate']:.3f} "
            f"(score p50={out.get('score_p50', float('nan')):.2f}, "
            f"p95={out.get('score_p95', float('nan')):.2f})"
        )
    metrics["stages"]["v3_cohort_validation"] = v3_cohort_validation

    # ── Sliding-window event extraction (supervisor request, 2026-05-13) ─
    # The cohort alert rates above are *per-window* (training-stride =
    # 1 s).  At deployment we want to know:
    #   * **when** an anomaly starts and ends (event-level annotation),
    #   * **how long** it lasts (duration distribution per cohort),
    #   * **how many** distinct events occur per recording.
    # We answer all three by re-running V3 at a finer 250 ms stride on
    # each anomaly-cohort recording and extracting events with
    # hysteresis-thresholded run-length detection (high = p95,
    # low = p90).  See `src/modeling/anomaly/event_detection.py`.
    try:
        from ..anomaly.event_detection import (
            detect_events_from_score_timeline,
            sliding_window_v3_inference,
            summarise_events,
        )

        log(
            "V3 sliding-window event extraction "
            "(stride=0.25 s, hysteresis: high=p95, low=p90, min_duration=0.10 s):"
        )
        # Per-cluster high / low thresholds.  We use the V3 trainer's
        # `p95` as `high`; the corresponding `low = p90` is derived
        # from the same healthy `val_scores` per cluster.  When
        # `val_scores` is sparse the percentile is taken across the
        # whole pool (cluster-agnostic) as a fallback.
        n_clusters = int(v3.thresholds.centroids.shape[0])
        per_cluster_p95 = v3.thresholds.p95.tolist()
        per_cluster_p90: list[float] = []
        if v3.val_scores.size > 0 and v3.val_contexts.shape[0] > 0:
            cluster_assign = v3.thresholds.assign(v3.val_contexts)
            for k in range(n_clusters):
                mask = cluster_assign == k
                if int(mask.sum()) > 4:
                    per_cluster_p90.append(float(np.percentile(v3.val_scores[mask], 90)))
                else:
                    per_cluster_p90.append(float(np.percentile(v3.val_scores, 90)))
        else:
            per_cluster_p90 = list(v3.thresholds.p95)

        cohort_event_summary: dict = {}
        for cohort_label, anomaly_loader, dataset_id_filter in (
            ("d2_random_fault", D2, "d2"),
            ("d3_hit", D3, "d3"),
            ("d4_random_fault", D4, "d4"),
        ):
            events_this_cohort: list = []
            n_recordings = 0
            for s in anomaly_loader.list_segments():
                if not s.is_anomaly:
                    continue
                seg = precompute_paired(s, v2_cfg)
                if seg is None:
                    continue
                try:
                    times_s, scores, contexts = sliding_window_v3_inference(
                        v2.encoder, v3.flow, seg,
                        v2_cfg=v2_cfg, inference_stride_s=0.25, device="cpu",
                    )
                except Exception as inner:
                    log(f"    {cohort_label}/{s.recording_id} skipped: {inner}")
                    continue
                if scores.size == 0:
                    continue
                # Per-window cluster assignment → per-window high/low
                # threshold.  When a single recording's windows span
                # multiple clusters we use the *median* threshold over
                # that recording's windows so the event extractor sees
                # a scalar pair of bounds — the right trade-off for an
                # event-localisation step where we want a single
                # decision boundary per recording.
                w_clusters = v3.thresholds.assign(contexts)
                rec_high = float(np.median([per_cluster_p95[int(k)] for k in w_clusters]))
                rec_low = float(np.median([per_cluster_p90[int(k)] for k in w_clusters]))
                # Hysteresis sanity: low must be ≤ high.
                if rec_low > rec_high:
                    rec_low = rec_high
                events = detect_events_from_score_timeline(
                    scores, times_s,
                    high_threshold=rec_high,
                    low_threshold=rec_low,
                    min_duration_s=0.10,
                    max_gap_windows=0,
                    recording_id=s.recording_id,
                    dataset_id=dataset_id_filter,
                    window_seconds=v2_cfg.window_seconds,
                )
                events_this_cohort.extend(events)
                n_recordings += 1
            summary = summarise_events(events_this_cohort)
            summary["n_recordings_audited"] = n_recordings
            cohort_event_summary[cohort_label] = summary
            log(
                f"  {cohort_label:<22} n_recordings={n_recordings:>3} "
                f"n_events={summary['n_events']:>4}  "
                f"median_duration={summary['median_duration_s']*1000 if not np.isnan(summary['median_duration_s']) else float('nan'):.0f} ms  "
                f"max_duration={summary['max_duration_s']:.2f} s"
            )
        metrics["stages"]["v3_sliding_window_events"] = cohort_event_summary
    except Exception as e:
        log(f"V3 sliding-window event extraction skipped: {type(e).__name__}: {e}")

    # ── V3 per-cluster threshold breakdown (RQ2 depth) ────────────────
    # The plan §V3 calls for a per-mode FPR breakdown.  Since the K-means
    # cluster IDs are arbitrary integers (label-free fit), we report the
    # alert rate per *predicted cluster* — interpretable in isolation per
    # cluster (does each cluster's healthy alert rate hit its 5 % target?
    # are the D2 RF alerts concentrated in the cluster K-means matched to
    # Turbine?) and complemented by the aggregate rows above.
    try:
        per_cluster_validation: dict = {}
        if v3.val_scores.size > 0:
            per_cluster_validation["healthy_holdout"] = per_cluster_alert_breakdown(
                v3.thresholds, v3.val_contexts, v3.val_scores,
                percentile=v3_cfg.threshold_percentile,
            )
        for label, samples in cohort_buckets.items():
            if not samples:
                continue
            cs_np = np.stack([s.context for s in samples], axis=0)
            xs = torch.from_numpy(np.stack([s.x_for_v3 for s in samples], axis=0))
            with torch.no_grad():
                sc = v3.flow.anomaly_score(xs, torch.from_numpy(cs_np)).numpy()
            per_cluster_validation[label] = per_cluster_alert_breakdown(
                v3.thresholds, cs_np, sc,
                percentile=v3_cfg.threshold_percentile,
            )
        log("V3 per-cluster threshold breakdown (RQ2 depth):")
        for label, row in per_cluster_validation.items():
            pieces = [
                f"k{ck}={r['n_alerts']}/{r['n']}({r['alert_rate']:.2f})"
                for ck, r in row["per_cluster"].items() if r["n"] > 0
            ]
            log(f"  {label:<22} {' | '.join(pieces) if pieces else '(empty)'}")
        metrics["stages"]["v3_per_cluster_breakdown"] = per_cluster_validation
    except Exception as e:
        log(f"V3 per-cluster breakdown skipped: {e}")

    # V3-gated cohort assembly:
    #   - D2 RandomFault + D3 hit: passthrough (recordings-level anomaly
    #     status already known; per-window labels not required).
    #   - D4 RandomFault: V3-gated at the unsupervised threshold percentile.
    gate_percentile = v3_cfg.threshold_percentile
    log(f"V4 — V3-gating D4 windows (gate_percentile={gate_percentile!r}; "
        "D2/D3 windows pass-through) ...")
    v4_samples, gate_stats = gate_samples_by_alert(
        v4_samples_all,
        v3.flow,
        v3.thresholds,
        percentile=gate_percentile,
        keep_dataset_ids=("d2", "d3"),
    )
    log(
        f"  gate stats: in={gate_stats['n_in']}, kept={gate_stats['n_kept']}, "
        f"by_dataset={gate_stats['by_dataset']}"
    )
    metrics["stages"]["v4_gating"] = {
        **gate_stats,
        "gate_percentile": str(gate_percentile),
    }

    if len(v4_samples) >= 4:
        v4_cfg = _v4_cfg(quick)
        log("V4 — training localization head ...")
        t0 = time.time()
        v4 = train_v4_localization(v4_samples, cfg=v4_cfg, grid=grid)
        log(
            f"V4 done in {time.time() - t0:.1f}s — val MAE: {v4.val_mae_3d:.3f} m "
            f"[95% CI {v4.val_mae_ci_low:.3f}, {v4.val_mae_ci_high:.3f}], "
            f"p95: {v4.val_p95_3d:.3f} m"
        )
        torch.save(v4.head.state_dict(), out_dir / "v4" / "head.pt")
        # Decompose val MAE into the soft-argmax prior + FiLM residual.  If
        # ‖init−target‖ ≈ ‖pred−target‖ the residual is not helping; if the
        # residual reduces the error, conditioning is pulling its weight.
        init_err_mean = (
            float(np.linalg.norm(v4.val_init_xyz - v4.val_targets, axis=-1).mean())
            if v4.val_init_xyz.size else float("nan")
        )
        delta_norm_mean = (
            float(np.linalg.norm(v4.val_residuals, axis=-1).mean())
            if v4.val_residuals.size else float("nan")
        )
        log(
            f"V4 decomposition: init_xyz (soft-argmax only) MAE={init_err_mean:.3f} m, "
            f"residual ||δ||={delta_norm_mean:.3f} m, final MAE={v4.val_mae_3d:.3f} m"
        )
        log("V4 per-recording breakdown:")
        for k, row in sorted(v4.val_recording_breakdown.items()):
            log(
                f"  {k:<48} n={row['n']:>3} mae={row['mae_3d']:.3f}m "
                f"target={tuple(round(v, 3) for v in row['target_xyz'])} "
                f"pred={tuple(round(v, 3) for v in row['pred_xyz_mean'])}"
            )
        metrics["stages"]["v4"] = {
            "epochs": v4_cfg.epochs,
            "n_samples_total": len(v4_samples),
            "n_train_recordings": len(v4.train_recording_ids),
            "n_val_recordings": len(v4.val_recording_ids),
            "val_mae_3d": v4.val_mae_3d,
            "val_mae_ci95_low": v4.val_mae_ci_low,
            "val_mae_ci95_high": v4.val_mae_ci_high,
            "val_mae_ci_method": v4.val_mae_ci_method,
            "val_p95_3d": v4.val_p95_3d,
            "val_init_mae_3d": init_err_mean,
            "val_residual_norm_mean": delta_norm_mean,
            "val_recording_breakdown": v4.val_recording_breakdown,
            "train_loss_final": v4.train_loss_history[-1],
            "val_loss_final": v4.val_loss_history[-1],
            "burst_aware_srp": True,
        }

        # A3 ablation: unconditional
        log("V4 — A3 ablation (unconditional=True) ...")
        v4_a3_cfg = _v4_cfg(quick, unconditional=True)
        t0 = time.time()
        v4_a3 = train_v4_localization(v4_samples, cfg=v4_a3_cfg, grid=grid)
        log(
            f"V4 A3 done in {time.time() - t0:.1f}s — val MAE: {v4_a3.val_mae_3d:.3f} m "
            f"[95% CI {v4_a3.val_mae_ci_low:.3f}, {v4_a3.val_mae_ci_high:.3f}]"
        )
        metrics["stages"]["v4_a3_unconditional"] = {
            "val_mae_3d": v4_a3.val_mae_3d,
            "val_mae_ci95_low": v4_a3.val_mae_ci_low,
            "val_mae_ci95_high": v4_a3.val_mae_ci_high,
            "val_p95_3d": v4_a3.val_p95_3d,
        }

        # ── RQ3 statistical significance: paired V4 vs A3 ─────────────
        # Per-window Euclidean errors are paired (same val windows scored
        # by both the conditional V4 head and the unconditional A3 head).
        # The paired bootstrap on Δ_MAE is the rigorous answer to "does
        # FiLM(c) conditioning significantly improve localisation?".
        try:
            if (
                v4.val_predictions.shape[0] == v4_a3.val_predictions.shape[0]
                and v4.val_predictions.shape[0] >= 4
            ):
                err_v4 = np.linalg.norm(
                    v4.val_predictions - v4.val_targets, axis=-1
                ).astype(np.float64)
                err_a3 = np.linalg.norm(
                    v4_a3.val_predictions - v4_a3.val_targets, axis=-1
                ).astype(np.float64)
                paired_v4 = paired_bootstrap_test(
                    err_v4, err_a3,
                    lower_is_better=True, n_boot=1000, seed=v4_cfg.seed,
                )
                log(
                    f"V4 vs A3 (paired bootstrap on per-window 3-D error): "
                    f"Δ_MAE={paired_v4.delta_point*1000:.1f} mm "
                    f"[95% CI {paired_v4.delta_ci_low*1000:.1f}, "
                    f"{paired_v4.delta_ci_high*1000:.1f} mm], "
                    f"p={paired_v4.p_value_two_sided:.4f}, "
                    f"direction={paired_v4.direction}"
                )
                metrics["stages"]["v4_vs_a3_paired_test"] = {
                    "delta_mae_m": paired_v4.delta_point,
                    "delta_mae_ci95_low_m": paired_v4.delta_ci_low,
                    "delta_mae_ci95_high_m": paired_v4.delta_ci_high,
                    "p_value_two_sided": paired_v4.p_value_two_sided,
                    "direction": paired_v4.direction,
                    "n_paired_windows": int(err_v4.shape[0]),
                    "method": "paired_percentile_bootstrap_1000",
                }
        except Exception as e:
            log(f"V4 vs A3 paired test skipped: {e}")

        # ── RQ3 depth: V4 channel ablation (A5) ───────────────────────
        # Severs each V4 front-end channel in turn to isolate its
        # marginal contribution.  The A3 ablation severs FiLM(c) but
        # keeps both front-end channels.  A5 instead keeps FiLM(c) and
        # severs SRP / TDOA / neither.  Reading: if "srp_only" matches
        # "both" closely, the structure-borne TDOA pathway is doing
        # little additional spatial work; if "tdoa_only" is far behind
        # "both", the acoustic SRP is the dominant signal — and so on.
        # Defends the multimodal-localization claim at the front-end
        # level rather than just at the FiLM-conditioning level.
        from dataclasses import asdict as _asdict
        for channel_mode in ("srp_only", "tdoa_only"):
            try:
                a5_cfg = V4Config(**{**_asdict(v4_cfg), "channel_mode": channel_mode})
                t0 = time.time()
                v4_a5 = train_v4_localization(v4_samples, cfg=a5_cfg, grid=grid)
                log(
                    f"V4 A5 ({channel_mode}) done in {time.time() - t0:.1f}s — "
                    f"val MAE: {v4_a5.val_mae_3d:.3f} m "
                    f"[95% CI {v4_a5.val_mae_ci_low:.3f}, {v4_a5.val_mae_ci_high:.3f}]"
                )
                metrics["stages"][f"v4_a5_{channel_mode}"] = {
                    "channel_mode": channel_mode,
                    "val_mae_3d": v4_a5.val_mae_3d,
                    "val_mae_ci95_low": v4_a5.val_mae_ci_low,
                    "val_mae_ci95_high": v4_a5.val_mae_ci_high,
                    "val_p95_3d": v4_a5.val_p95_3d,
                }
            except Exception as e:
                log(f"V4 A5 ({channel_mode}) skipped: {e}")
    else:
        log(f"V4 SKIPPED — only {len(v4_samples)} labeled samples (need ≥4 for split)")
        metrics["stages"]["v4"] = {"skipped": True, "n_samples_total": len(v4_samples)}

    # ================================================================ V5.1
    # V5.1 — fan-noise robustness conditioning (NOT a SCADA enhancement).
    # The D3 / D4 `speed{N}` token is a level of an external background-
    # noise fan added to the recording rig to make both mode discovery
    # and anomaly localisation harder; it is *not* an operational SCADA
    # variable.  RQ4's V5.1 row therefore tests whether explicitly
    # informing the localisation head of the noise level helps it stay
    # accurate in noisier conditions — a robustness ablation rather than
    # an enhancement-from-additional-side-channel ablation.
    log("V5.1 — fan-noise robustness conditioning (D3+D4 speed-level one-hot) ...")
    speed_lookup = {**d3_speed_lookup(d3_segments), **d3_speed_lookup(D4.list_segments())}
    if speed_lookup and len(v4_samples) >= 4:
        log(f"  speed lookup ({len(speed_lookup)} recordings): "
            f"{ {k: v.tolist() for k, v in list(speed_lookup.items())[:3]} } ...")
        # Reuse the gated cohort — V5.1 only differs from V4 by injecting
        # the speed one-hot into the FiLM conditioner.  Attach scada per
        # recording; D2 recordings (no speed bucket) get a zero vector so
        # the head's `s_dim` stays consistent across the batch.
        scada_dim = next(iter(speed_lookup.values())).shape[0]
        v5_1_samples = []
        for s in v4_samples:
            scada = speed_lookup.get(s.recording_id)
            if scada is None:
                scada = np.zeros(scada_dim, dtype=np.float32)
            v5_1_samples.append(
                V4Sample(
                    srp_volume=s.srp_volume,
                    tdoa_tokens=s.tdoa_tokens,
                    context=s.context,
                    x_for_v3=s.x_for_v3,
                    target_xyz=s.target_xyz,
                    scada=scada,
                    mode_label=s.mode_label,
                    recording_id=s.recording_id,
                    source_dir=s.source_dir,
                    dataset_id=s.dataset_id,
                )
            )
        v5_1_cfg = _v4_cfg(quick, scada_dim=scada_dim)
        t0 = time.time()
        v5_1 = train_v4_localization(v5_1_samples, cfg=v5_1_cfg, grid=grid)
        log(
            f"V5.1 done in {time.time() - t0:.1f}s — val MAE: {v5_1.val_mae_3d:.3f} m "
            f"[95% CI {v5_1.val_mae_ci_low:.3f}, {v5_1.val_mae_ci_high:.3f}]"
        )
        torch.save(v5_1.head.state_dict(), out_dir / "v5_1" / "head_speed.pt")
        metrics["stages"]["v5_1"] = {
            "scada_dim": scada_dim,
            "val_mae_3d": v5_1.val_mae_3d,
            "val_mae_ci95_low": v5_1.val_mae_ci_low,
            "val_mae_ci95_high": v5_1.val_mae_ci_high,
            "val_p95_3d": v5_1.val_p95_3d,
        }
    else:
        log("V5.1 SKIPPED — no speed segments or insufficient V4 samples")
        metrics["stages"]["v5_1"] = {"skipped": True}

    # ================================================================ V5.2
    # Illwerke SCADA mutual-information ranking.
    #
    # Method (per plan §V5.2): take the legacy 5-layer pipeline anomaly
    # events as the binary anomaly indicator on a 1 Hz grid; load Allg_M1
    # plant process variables decimated to 1 Hz; compute mutual
    # information per channel; rank top-K.  No injection on D2/D3 —
    # this is the offline analysis that produces the **deployment
    # recommendation** the thesis claims for RQ4 alongside the V5.1
    # categorical-conditioning null result.
    #
    # Critically defensible methodologically: even though the company
    # data we have access to is RMS-only with too few mics to support
    # localization, the Allg_M1 process variables we *do* have are
    # sufficient to rank candidate SCADA channels by their statistical
    # association with anomaly events.  This converts the data
    # constraint into a real RQ4 contribution rather than a null row.
    log("V5.2 — Allg_M1 MI ranking (offline; uses legacy 5-layer anomaly events) ...")
    try:
        from ..scada.channel_mining import (
            anomaly_indicator,
            load_anomaly_events,
            rank_channels_by_mi,
        )
        from ...ingestion.illwerke_loader import load_allg_campaign

        events_path = REPO_ROOT / "results" / "illwerke" / "pipeline" / "anomaly_events.json"
        illwerke_root = REPO_ROOT / "data" / "illwerke"
        if not events_path.exists():
            log(f"V5.2 SKIPPED — anomaly events file not present at {events_path}")
            metrics["stages"]["v5_2"] = {
                "skipped_reason": f"anomaly events not found at {events_path}"
            }
        elif not illwerke_root.exists() or not (illwerke_root / "ROWII_Allg_M1").exists():
            log(f"V5.2 SKIPPED — Allg_M1 raw data not present under {illwerke_root}")
            metrics["stages"]["v5_2"] = {
                "skipped_reason": f"Illwerke Allg_M1 directory not found at {illwerke_root}"
            }
        else:
            events = load_anomaly_events(events_path)
            log(f"  loaded {len(events)} legacy anomaly events from {events_path.name}")
            ts_ns, allg_data, channel_names = load_allg_campaign(illwerke_root)
            log(
                f"  Allg_M1 loaded: T={allg_data.shape[0]} samples @ 1 Hz, "
                f"{allg_data.shape[1]} channels"
            )
            indicator = anomaly_indicator(ts_ns, events)
            n_anom = int(indicator.sum())
            if n_anom == 0:
                log("V5.2 SKIPPED — anomaly indicator is all-zero on this campaign window")
                metrics["stages"]["v5_2"] = {
                    "skipped_reason": "no anomaly events overlap with Allg_M1 campaign window",
                    "n_total_samples": int(indicator.shape[0]),
                }
            else:
                ranking = rank_channels_by_mi(
                    allg_data, channel_names, indicator, seed=42,
                )
                top_k = ranking.top_k(8)
                log(f"V5.2 top-8 Allg_M1 channels by MI with anomaly events:")
                for name, mi_val, family in top_k:
                    log(f"  MI={mi_val:.4f}  [{family:<10}]  {name}")
                metrics["stages"]["v5_2"] = ranking.to_dict(k=20)  # persist top-20
    except Exception as e:
        log(f"V5.2 SKIPPED — {type(e).__name__}: {e}")
        metrics["stages"]["v5_2"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    # ============================================================== persist
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"Metrics written to {metrics_path}")
    # Explicit UTF-8 + replacement: the run log contains diagnostic
    # symbols (Greek δ, ‖·‖, etc.) that Windows cp1252 (the default
    # `write_text` encoding on this platform) cannot represent.  Writing
    # as UTF-8 with `errors="replace"` makes the log archival robust
    # without losing any metric values.
    (out_dir / "run_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8", errors="replace"
    )

    # ============================================================== archive
    # Auto-archive the run into `results/runs/<timestamp>__<label>/` so
    # every full-run accumulates as a permanent record for the master's-
    # thesis Chapter 6 tables.  The archive carries `metrics.json`, the
    # full set of trained checkpoints, and a manifest with git commit +
    # config so the run can be reproduced or re-evaluated later.
    try:
        from .archive import archive_run

        label = "quick" if quick else "full"
        if dataset_ids is not None:
            label += "_only_" + "+".join(sorted(dataset_ids))
        archive_dir = archive_run(
            out_dir,
            run_label=f"{label}_seed{v1_cfg.seed}",
            extra_manifest={
                "v1_cfg": v1_cfg,
                "v2_cfg": v2_cfg,
                "v3_cfg": v3_cfg,
                "quick": quick,
                "dataset_ids": dataset_ids,
            },
        )
        log(f"Run archived to {archive_dir}")
    except Exception as e:
        log(f"Archival failed (non-fatal): {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Halve epoch counts for a smoke-level real-data run",
    )
    parser.add_argument(
        "--only-datasets", type=str, nargs="+", default=None,
        choices=["d1", "d2", "d3", "d4"],
        help="Restrict V1/V2/V3 SSL training pools to only these dataset IDs. "
             "Useful for the D4-only V1 vibration ablation that tests whether "
             "vibration's mode-discrimination weakness is an acquisition (D1/D2/D3 "
             "peak-amplitude stream) limitation rather than an encoder defect.",
    )
    args = parser.parse_args()
    main(quick=args.quick, dataset_ids=tuple(args.only_datasets) if args.only_datasets else None)
