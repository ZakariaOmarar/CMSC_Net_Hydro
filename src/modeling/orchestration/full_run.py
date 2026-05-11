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
        # Bound the FiLM-conditioned residual at ±5 cm so a poorly-trained
        # residual cannot blow up the soft-argmax prior on the 10 cm prototype.
        residual_scale_m=0.05,
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

    log("V1 acoustic — training on D1+D2+D3+D4 healthy windows ...")
    t0 = time.time()
    v1_acoustic = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    log(f"V1 acoustic done in {time.time() - t0:.1f}s — sanity gate: {v1_acoustic.sanity_gate}")
    torch.save(v1_acoustic.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_acoustic.train_loss_history[-1],
        "val_loss_final": v1_acoustic.val_loss_history[-1],
        "sanity_purity": v1_acoustic.sanity_gate.get("purity", 0.0),
        "sanity_nmi": v1_acoustic.sanity_gate.get("nmi", 0.0),
        "sanity_n_windows": v1_acoustic.sanity_gate.get("n_windows", 0),
        "n_train_recordings": len(v1_acoustic.train_recording_ids),
        "n_val_recordings": len(v1_acoustic.val_recording_ids),
    }

    log("V1 vibration — training on D1+D2+D3+D4 healthy windows ...")
    t0 = time.time()
    v1_vibration = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    log(f"V1 vibration done in {time.time() - t0:.1f}s — sanity gate: {v1_vibration.sanity_gate}")
    torch.save(v1_vibration.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_vibration.train_loss_history[-1],
        "val_loss_final": v1_vibration.val_loss_history[-1],
        "sanity_purity": v1_vibration.sanity_gate.get("purity", 0.0),
        "sanity_nmi": v1_vibration.sanity_gate.get("nmi", 0.0),
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
    log(f"V2 done in {time.time() - t0:.1f}s — RQ1 purity: {v2.rq1.get('purity', 0.0):.3f}")
    torch.save(v2.encoder.state_dict(), out_dir / "v2" / "encoder.pt")
    torch.save(v2.projection.state_dict(), out_dir / "v2" / "projection.pt")
    metrics["stages"]["v2"] = {
        "epochs": v2_cfg.epochs,
        "train_loss_final": v2.train_loss_history[-1],
        "val_loss_final": v2.val_loss_history[-1],
        "train_simclr_final": v2.train_simclr_history[-1],
        "train_lmm_final": v2.train_lmm_history[-1],
        "rq1_purity": v2.rq1.get("purity", 0.0),
        "rq1_nmi": v2.rq1.get("nmi", 0.0),
        "rq1_n_windows": v2.rq1.get("n_windows", 0),
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
    log(f"V2 A1 done in {time.time() - t0:.1f}s — purity: {v2_a1.rq1.get('purity', 0.0):.3f}")
    metrics["stages"]["v2_a1_drop_vibration"] = {
        "rq1_purity": v2_a1.rq1.get("purity", 0.0),
        "rq1_nmi": v2_a1.rq1.get("nmi", 0.0),
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
        "n_clusters_fit": int(v3.thresholds.centroids.shape[0]),
        "p99_per_cluster": v3.thresholds.p99.tolist(),
        "n_val_windows_healthy": int(v3.val_scores.shape[0]),
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
    paired_by_dataset_and_mode: dict[tuple[str, str], list] = {}
    for L in (D1, D2):
        for s in L.list_segments():
            if s.mode_label is None or s.is_anomaly:
                continue
            p = precompute_paired(s, v2_cfg)
            if p is not None:
                paired_by_dataset_and_mode.setdefault(
                    (s.dataset_id, s.mode_label), []
                ).append(p)

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
        log(f"V4 done in {time.time() - t0:.1f}s — val MAE: {v4.val_mae_3d:.3f} m, p95: {v4.val_p95_3d:.3f} m")
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
        log(f"V4 A3 done in {time.time() - t0:.1f}s — val MAE: {v4_a3.val_mae_3d:.3f} m")
        metrics["stages"]["v4_a3_unconditional"] = {
            "val_mae_3d": v4_a3.val_mae_3d,
            "val_p95_3d": v4_a3.val_p95_3d,
        }
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
        log(f"V5.1 done in {time.time() - t0:.1f}s — val MAE: {v5_1.val_mae_3d:.3f} m")
        torch.save(v5_1.head.state_dict(), out_dir / "v5_1" / "head_speed.pt")
        metrics["stages"]["v5_1"] = {
            "scada_dim": scada_dim,
            "val_mae_3d": v5_1.val_mae_3d,
            "val_p95_3d": v5_1.val_p95_3d,
        }
    else:
        log("V5.1 SKIPPED — no speed segments or insufficient V4 samples")
        metrics["stages"]["v5_1"] = {"skipped": True}

    # ================================================================ V5.2
    log("V5.2 — Allg_M1 MI ranking SKIPPED (no Illwerke raw data in this checkout)")
    metrics["stages"]["v5_2"] = {"skipped_reason": "Illwerke raw data not present"}

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
