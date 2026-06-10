"""End-to-end thesis orchestrator (b5_cma + three/four paradigms).

Single-process pipeline that produces every artefact and metric the
Chapter 6 paradigm-comparison tables need under one timestamped output
directory ``results/runs/<ts>__full_pipeline_b5_cma/``:

  Stage 0  cross-modal sync verification + correction audit
  Stage 1  V0 baselines: RQ2 anomaly reference (Khamaisi trio + KDE, pooled,
             acoustic + vibration) plus per-dataset LightGBM mode + SRP-PHAT
  Stage 2  V1 + V2 trained with the ``b5_cma`` intervention
             (cma_weight=0.5, cma_temperature=0.1)
             + V2 A1 ablation (drop_vibration) + modality-balance probe
  Stage 3  V3 three paradigms (V3-acoustic / V3-vibration / V3-fusion)
  Stage 4  V3 fusion depth — A2 unconditional + paired bootstrap V3 vs A2,
             synthetic anomaly ROC-AUC, transition FPR, per-cluster
             threshold breakdown, sliding-window event extraction
  Stage 5  V4 four paradigms (acoustic / vibration / tdoa_legacy / fusion)
             + V0 SRP-PHAT and accel-multilateration per dataset
  Stage 6  V4 fusion depth — A3 unconditional + paired bootstrap V4 vs A3
  Stage 7  V5.1 fan-noise robustness conditioning (speed one-hot SCADA)
  Stage 8  Inline late-fusion eval (LF AND / OR / score-weighted / MAX
             rows via ``rq2_three_paradigm_eval`` on this run dir)
  Stage 9  Inline RQ3 localisation paradigm eval (LF confidence-gated +
             LORO cross-validation via ``rq3_three_paradigm_eval``)

The module-level config builders (`resolved_loader`, `v1_config`, `v2_config`,
`v3_config`, `v4_config`, `_d3_spatial_overrides`) are the canonical source of
truth shared with the sibling orchestrators in this package and the
ablation scripts under ``scripts/`` — change configs here, not at the
caller.

Run::

    python -m src.modeling.orchestration.full_run           # full (~2 h CPU)
    python -m src.modeling.orchestration.full_run --quick   # smoke (~25 min)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ...config import resolve_device
from ...config.architecture import SYNC
from ...config.dataset_registry import REGISTRY
from ...ingestion.test_dataset_loader import DatasetSpec, TestDatasetLoader, TestDatasetSegment
from ..anomaly import (
    V3Config,
    train_v3_cnf,
    transition_fpr,
)
from ..anomaly.event_detection import (
    detect_events_from_score_timeline,
    sliding_window_v3_inference,
    summarise_events,
)
from ..anomaly.synthetic_eval import evaluate_synthetic_anomaly_auc
from ..anomaly.threshold import per_cluster_alert_breakdown
from ..anomaly.v3_per_modality import V3AcousticOnlyAdapter, V3VibrationOnlyAdapter
from ..anomaly.v3_trainer import encoder_level_transition_fpr, precompute_paired
from ..anomaly_baselines import (
    ALL_MODELS,
    MODALITIES,
    SRPConfig,
    V0Config,
    V0ModeConfig,
    cluster_mode_floor,
    evaluate_srp_phat,
    evaluate_v0_anomaly,
    summarise,
    train_v0_lstm_ae,
    train_v0_mode_lgbm,
)
from ..context.modality_probe import run_modality_balance_probe
from ..context.v1_ssl import train_v1_per_modality
from ..context.v2_ssl import V2SSLConfig, train_v2_fusion
from ..eval import paired_bootstrap_test
from ..localization import (
    V4_CANDIDATE_GRID,
    V4Config,
    V4Sample,
    precompute_v4_samples,
    train_v4_localization,
)
from ..localization.multilateration import accel_tdoa_multilateration_v0
from ..scada import d3_speed_lookup

# V1-V4 hyperparameter builders live in `stage_configs`; they are imported (and
# re-exported) here so `main()` resolves them from this module's namespace,
# which keeps the multi-seed / hop-length drivers' monkeypatching of
# `full_run.vN_config` working.
from .stage_configs import v1_config, v2_config, v3_config, v4_config

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def resolved_loader(yaml_name: str) -> TestDatasetLoader:
    """Build a sync-corrected loader for one dataset spec.

    Two things matter here that the legacy implementation missed:

      1. ``vibration_format`` MUST be propagated when reconstructing
         the spec — D4's spec sets ``vibration_format="raw"`` and the
         default-"peak" fallback would silently pick the wrong CSV
         family (and either error on missing files or, worse, load
         the peak-decimated stream instead of the 376 Hz raw waveform).

      2. ``sync_correct=True`` is set at the loader level so the
         WavVibrationAdapter applies the four-gate cross-modal sync
         correction at load time, BEFORE the frozen ``DataSegment`` is
         built.  The legacy orchestrator pattern — load, then mutate
         ``s.segment.mic_data = mic_corr`` after a separate auto-sync
         call — was a silent no-op: the assignment raised
         ``FrozenInstanceError`` on every recording and the bare
         ``except Exception`` in the audit loop swallowed it as
         ``n_skipped += 1``.  Configuring the loader's flag is the
         only working entry point and guarantees every downstream
         stage (V0 through V5) consumes sync-aligned segments.

    The four sync gate thresholds match the orchestrator's historical
    values so the audit-table semantics in chapter 6 carry over.
    """
    # `DatasetSpec.from_yaml` now resolves all paths (root, position_path) to
    # absolute REPO_ROOT-prefixed values, so the legacy reconstruction is
    # unnecessary.
    spec = DatasetSpec.from_yaml(REPO_ROOT / "configs" / "datasets" / yaml_name)
    return TestDatasetLoader(
        spec,
        sync_correct=True,
        # Sync gating thresholds sourced from `SYNC` in
        # `src/config/architecture.py` — change there, not here.
        sync_correct_kwargs=dict(
            max_offset_s=SYNC.max_offset_s,
            n_sub_segments=SYNC.n_sub_segments,
            confidence_floor=SYNC.confidence_floor,
            drift_tolerance_s=SYNC.drift_tolerance_s,
            min_offset_to_correct_s=SYNC.min_offset_to_correct_s,
            use_fractional_shift=SYNC.use_fractional_shift,
        ),
    )


def _all_segments(loaders: list[TestDatasetLoader]) -> list[TestDatasetSegment]:
    out: list[TestDatasetSegment] = []
    for L in loaders:
        out.extend(L.list_segments())
    return out


# ---------------------------------------------------------------------------
# V4 spatial holdout — positions reserved as the localization generalisation
# test (folder coords in cm → metres).  Held out of V4 training so the
# reported MAE measures localise-an-unseen-position, not within-position
# interpolation.
# ---------------------------------------------------------------------------

V4_HOLDOUT_POSITIONS_M: list[tuple[float, float, float]] = [
    (0.22, 0.0, 0.0),     # D5 knock (22, 0, 0)
    (0.03, -0.03, 0.08),  # D5 knock (3, -3, 8)
    (0.06, -0.15, 0.0),   # D5 knock (6, -15, 0)
    (0.02, 0.04, 0.08),   # D4 RandomFault_knock (2, 4, 8)
    (0.0, -0.20, 0.0),    # D4 RandomFault_knock (0, -20, 0)
]


def _v3_event_intervals_for_recordings(
    holdout_samples,
    loaders_by_id: dict,
    v2_encoder,
    v3,
    v2_cfg,
    v3_cfg,
) -> dict:
    """Return {recording_id: [(t_start_s, t_end_s), ...]} of V3-detected events.

    Runs V3 (the trained fusion flow + thresholds) at a fine stride on each
    held-out recording and extracts alert events.  Used to GATE the V4
    holdout: V4 only "fires" on windows V3 flags anomalous in deployment, so
    the gated MAE is the deployment-faithful localization number.
    """
    from ..anomaly.event_detection import (
        detect_events_from_score_timeline,
        sliding_window_v3_inference,
    )
    from ..anomaly.v3_trainer import precompute_paired

    bar = v3.thresholds.p99 if int(v3_cfg.threshold_percentile) >= 99 else v3.thresholds.p95
    # Map each holdout sample's (dataset_id, recording_id) back to a segment.
    wanted: dict[str, str] = {s.recording_id: s.dataset_id for s in holdout_samples}
    out: dict[str, list[tuple[float, float]]] = {}
    for dsid in sorted(set(wanted.values())):
        loader = loaders_by_id.get(dsid)
        if loader is None:
            continue
        for s in loader.list_segments():
            if s.recording_id not in wanted or wanted[s.recording_id] != dsid:
                continue
            paired = precompute_paired(s, v2_cfg)
            if paired is None:
                continue
            try:
                times, scores, contexts = sliding_window_v3_inference(
                    v2_encoder, v3.flow, paired,
                    v2_cfg=v2_cfg, inference_stride_s=0.25,
                    xt_pool=getattr(v3, "xt_pool", None), device=v3_cfg.device,
                )
            except Exception:
                # Skip a recording whose V3 inference fails; it simply gets no
                # event-interval entry (callers read `out` with .get()).
                continue
            if scores.size == 0:
                continue
            clusters = v3.thresholds.assign(contexts)
            high = float(np.median([float(bar[int(k)]) for k in clusters]))
            # low <= high required; V3 scores are negative NLLs so 0.95*high
            # would invert (see event_detection.v3_real_anomaly_detection).
            low = high - abs(high) * 0.05
            if low > high:
                low = high
            evs = detect_events_from_score_timeline(
                scores, times, high_threshold=high, low_threshold=low,
                min_duration_s=0.10, max_gap_windows=0,
                recording_id=s.recording_id, dataset_id=dsid,
                window_seconds=v2_cfg.window_seconds,
            )
            out[s.recording_id] = [(e.t_start_s, e.t_end_s) for e in evs]
    return out


# ---------------------------------------------------------------------------
# Spatial-label derivation for D3 hits
# ---------------------------------------------------------------------------


# Ground-truth spatial label for D3's `hit_between_Fl_Gr_speed1` family.
# Sensors Fl=(6, -5, 8) cm and Gr=(11, 0, 8) cm both sit at z=8 cm, so the
# knock is constrained to that height and approximated by their centroid
# (cm converted to metres). z is the reliable constraint; x, y carry larger
# uncertainty. This is the only D3 hit family with a usable spatial label.
_D3_HIT_FL_GR_XYZ_M: tuple[float, float, float] = (0.085, -0.025, 0.080)


def _d3_spatial_overrides(d3_segments: list[TestDatasetSegment]) -> dict[str, tuple[float, float, float]]:
    """Derive spatial labels for D3 `hit_between_*_speed*` recordings.

    Only the `hit_between_Fl_Gr_*` family has a usable ground-truth position
    (`_D3_HIT_FL_GR_XYZ_M`); it is matched on either the recording id or the
    source folder name. Other hit pairs lack a reliable label and are skipped.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for s in d3_segments:
        if "hit_between" not in s.recording_id and "hit_between" not in str(s.source_dir).lower():
            continue
        rec_lower = s.recording_id.lower()
        src_lower = str(s.source_dir).lower()
        is_fl_gr = ("fl" in rec_lower and "gr" in rec_lower) or (
            "fl" in src_lower and "gr" in src_lower
        )
        if is_fl_gr:
            out[s.recording_id] = _D3_HIT_FL_GR_XYZ_M
    return out


# ---------------------------------------------------------------------------
# Logging + git provenance
# ---------------------------------------------------------------------------


def _make_logger(out_dir: Path) -> Callable[[str], None]:
    """Return a `log(msg)` that prints and appends to ``run_log.txt``."""
    log_path = out_dir / "run_log.txt"

    def log(msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")

    return log


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True, capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage 0 — sync verification + correction audit
# ---------------------------------------------------------------------------


def _audit_sync(loaders: list, log: Callable[[str], None]) -> dict:
    out: dict = {}
    for L in loaders:
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
            report = s.segment.metadata.get("sync_correction")
            if report is None:
                n_skipped += 1
                continue
            audit_offset = float(report.get("audit_offset_s", float("nan")))
            audit_conf = float(report.get("audit_confidence", float("nan")))
            env_kurt = float(report.get("acoustic_envelope_kurtosis", float("nan")))
            if not np.isnan(audit_offset):
                offsets_s.append(audit_offset)
                confidences.append(audit_conf)
            if not np.isnan(env_kurt):
                env_kurtoses.append(env_kurt)
            reason = str(report.get("reason") or "").lower()
            if report.get("applied"):
                n_applied += 1
            elif "stability" in reason or "drift" in reason:
                n_rejected_drift += 1
            elif "near-gaussian" in reason:
                n_rejected_flat_envelope += 1
            elif "confidence" in reason or "uninformative" in reason:
                n_rejected_low_conf += 1
            elif "below" in reason or "already aligned" in reason:
                n_rejected_below_floor += 1
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
                f"  {ds_name}: n={n_total} median_offset={med*1e3:+.1f} ms "
                f"(MAD ±{mad*1e3:.1f} ms) conf={mean_conf:.2f} "
                f"applied={n_applied} drift={n_rejected_drift} "
                f"low_conf={n_rejected_low_conf} flat={n_rejected_flat_envelope}"
            )
            out[ds_name] = {
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
                "median_acoustic_envelope_kurtosis": med_kurt,
            }
        else:
            log(f"  {ds_name}: no auditable recordings (n_skipped={n_skipped})")
            out[ds_name] = {"n_recordings_total": 0, "n_skipped": n_skipped}
    return out


# ---------------------------------------------------------------------------
# Stage 1 — V0 baselines per dataset
# ---------------------------------------------------------------------------


def _run_v0(loaders: list, log: Callable[[str], None], anom_loaders: list | None = None) -> dict:
    out: dict = {}
    anom_loaders = anom_loaders if anom_loaders is not None else loaders
    for L in loaders:
        ds_name = L.spec.id
        if ds_name in ("d1", "d2"):
            try:
                t0 = time.time()
                r = train_v0_mode_lgbm(L, V0ModeConfig())
                log(f"  V0 LGBM mode ({ds_name}) {time.time()-t0:.0f}s — "
                    f"val macro-F1={r.val_macro_f1:.3f}")
                out[f"v0_lgbm_{ds_name}"] = {
                    "val_macro_f1": float(r.val_macro_f1),
                    "val_per_class_f1": {str(k): float(v) for k, v in r.val_per_class_f1.items()},
                    "n_train_recordings": len(r.train_recording_ids),
                    "n_val_recordings": len(r.val_recording_ids),
                }
            except Exception as e:
                log(f"  V0 LGBM mode ({ds_name}) skipped: {type(e).__name__}: {e}")
                out[f"v0_lgbm_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}
        try:
            t0 = time.time()
            ae = train_v0_lstm_ae(L, V0Config())
            log(f"  V0 LSTM-AE ({ds_name}) {time.time()-t0:.0f}s — "
                f"val recon MSE={ae.val_loss_history[-1]:.4f}")
            out[f"v0_lstm_ae_{ds_name}"] = {
                "val_loss_final": float(ae.val_loss_history[-1]),
                "n_train_recordings": len(ae.healthy_train_recordings),
                "n_val_recordings": len(ae.healthy_val_recordings),
            }
        except Exception as e:
            log(f"  V0 LSTM-AE ({ds_name}) skipped: {type(e).__name__}: {e}")
            out[f"v0_lstm_ae_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}
        if ds_name in ("d2", "d3", "d4"):
            try:
                recs = evaluate_srp_phat(L, SRPConfig())
                s = summarise(recs)
                log(f"  V0 SRP-PHAT ({ds_name}): {s.get('n_recordings', 0)} recordings, "
                    f"mean MAE={s.get('mean_error_m', float('nan')):.3f} m")
                out[f"v0_srp_phat_{ds_name}"] = s
            except Exception as e:
                log(f"  V0 SRP-PHAT ({ds_name}) skipped: {type(e).__name__}: {e}")
                out[f"v0_srp_phat_{ds_name}"] = {"skipped": f"{type(e).__name__}: {e}"}

    # RQ1 context floor — unsupervised K-means on hand-engineered features,
    # scored against the mode label (NMI / ARI / purity).  The lower bound the
    # label-free encoder must beat; the LightGBM rows above are the supervised
    # upper bound it approaches from below.
    try:
        floor = cluster_mode_floor(loaders, V0ModeConfig())
        log(f"  V0 RQ1 mode-floor (K-means/handcrafted): NMI={floor.nmi:.3f} "
            f"ARI={floor.ari:.3f} purity={floor.purity:.3f} "
            f"({floor.n_windows} win / {floor.n_recordings} rec)")
        out["v0_mode_floor"] = {
            "nmi": floor.nmi, "ari": floor.ari, "purity": floor.purity,
            "n_windows": floor.n_windows, "n_recordings": floor.n_recordings,
            "label_set": list(floor.label_set), "n_clusters": floor.n_clusters,
        }
    except Exception as e:
        log(f"  V0 RQ1 mode-floor skipped: {type(e).__name__}: {e}")
        out["v0_mode_floor"] = {"skipped": f"{type(e).__name__}: {e}"}

    # RQ2 anomaly reference — the full Khamaisi trio + KDE, pooled across all
    # campaigns (like the V3 training cohort) and scored on the same protocol the
    # conditional head reports: within-campaign healthy-vs-anomaly ROC-AUC plus
    # the in-distribution-vs-domain-shift false-positive-rate contrast.  This is
    # the credible prior-work reference V3 must improve on for RQ2.
    out["v0_anomaly_rq2"] = {}
    for modality in MODALITIES:
        for model in ALL_MODELS:
            cell = f"{modality}/{model}"
            try:
                t0 = time.time()
                res = evaluate_v0_anomaly(anom_loaders, model, modality, V0Config())
                out["v0_anomaly_rq2"][cell] = res.to_dict()
                log(f"  V0 RQ2 {cell} {time.time()-t0:.0f}s — "
                    f"ROC-AUC={res.roc_auc:.3f} FPR(in-dist={res.fpr_in_distribution:.3f} "
                    f"shift={res.fpr_domain_shift:.3f})")
            except Exception as e:
                log(f"  V0 RQ2 {cell} skipped: {type(e).__name__}: {e}")
                out["v0_anomaly_rq2"][cell] = {"skipped": f"{type(e).__name__}: {e}"}
    return out


# ---------------------------------------------------------------------------
# Stage 5 helper — V0 accel multilateration per dataset (vibration classical
# localisation; mirrors run_v4_three_paradigms.py).
# ---------------------------------------------------------------------------


def _run_v0_multilateration(
    loaders: list, overrides: dict, log: Callable[[str], None]
) -> dict:
    out: dict = {}
    for L in loaders:
        ds_name = L.spec.id
        if ds_name not in ("d2", "d3", "d4"):
            continue
        per_rec: list[dict] = []
        for s in L.list_segments():
            if not s.is_anomaly or s.spatial_label is None:
                continue
            try:
                if s.segment.accel_data.shape[0] < 4:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "n_accel < 4"})
                    continue
                pos, residual = accel_tdoa_multilateration_v0(
                    s.segment.accel_data, s.vib_positions,
                    fs=float(s.segment.accel_sample_rate),
                )
                target = overrides.get(s.recording_id) if ds_name == "d3" else s.spatial_label
                if target is None:
                    per_rec.append({"recording_id": s.recording_id, "skipped": "no spatial label"})
                    continue
                err = float(np.linalg.norm(pos - np.asarray(target, dtype=np.float64)))
                per_rec.append({
                    "recording_id": s.recording_id,
                    "target": list(map(float, target)),
                    "pred": list(map(float, pos)),
                    "residual": float(residual),
                    "error_m": err,
                })
            except Exception as e:
                per_rec.append({"recording_id": s.recording_id, "error": f"{type(e).__name__}: {e}"})
        errs = [r["error_m"] for r in per_rec if "error_m" in r]
        out[ds_name] = {
            "n_recordings": len(per_rec),
            "n_successful": len(errs),
            "mean_error_m": float(np.mean(errs)) if errs else float("nan"),
            "median_error_m": float(np.median(errs)) if errs else float("nan"),
            "p95_error_m": float(np.percentile(errs, 95)) if errs else float("nan"),
            "per_recording": per_rec,
        }
        log(f"  V0 multilat ({ds_name}): {len(errs)}/{len(per_rec)} resolved, "
            f"mean MAE={out[ds_name]['mean_error_m']:.3f} m")
    return out


# ---------------------------------------------------------------------------
# Stage 3 helper — train one V3 instance and persist artefacts the
# rq2_three_paradigm_eval CLI consumes (flow.pt + thresholds.npz + val_eval.npz).
# ---------------------------------------------------------------------------


def _train_one_v3(
    name: str,
    encoder: torch.nn.Module,
    loaders: list,
    v2_cfg: V2SSLConfig,
    v3_cfg: V3Config,
    out_dir: Path,
    log: Callable[[str], None],
):
    log(f"V3-{name} — training conditional CNF ...")
    t0 = time.time()
    # `encoder` may be the V2FusionEncoder (fusion paradigm) or one of the
    # V3{Acoustic,Vibration}OnlyAdapter wrappers (unimodal paradigms).  All
    # three implement the same `forward(...) -> (paired, c_t, x_t_per_w)`
    # contract train_v3_cnf consumes; the static type of the parameter is
    # narrower than the runtime contract.
    res = train_v3_cnf(encoder, loaders, v2_cfg=v2_cfg, v3_cfg=v3_cfg)  # type: ignore[arg-type]
    log(f"  V3-{name} {time.time()-t0:.0f}s — val NLL={res.val_nll[-1]:.3f}")
    pipe_dir = out_dir / f"v3_{name}"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res.flow.state_dict(), pipe_dir / "flow.pt")
    np.savez(
        pipe_dir / "thresholds.npz",
        centroids=res.thresholds.centroids,
        p95=res.thresholds.p95,
        p99=res.thresholds.p99,
        n_per_cluster=res.thresholds.n_per_cluster,
    )
    np.savez(
        pipe_dir / "val_eval.npz",
        scores=res.val_scores,
        contexts=res.val_contexts,
        labels=np.asarray(res.val_labels, dtype="U64"),
    )
    return res


# ---------------------------------------------------------------------------
# Stage 5 helper — train one V4 instance and persist artefacts the
# rq3_three_paradigm_eval CLI consumes (head.pt + val_predictions.npz).
# ---------------------------------------------------------------------------


def _train_one_v4(
    name: str,
    channel_mode: str,
    samples: list,
    grid,
    base_cfg: V4Config,
    out_dir: Path,
    log: Callable[[str], None],
):
    log(f"V4-{name} (channel_mode={channel_mode}) — training localisation head ...")
    cfg = replace(base_cfg, channel_mode=channel_mode)
    t0 = time.time()
    res = train_v4_localization(samples, cfg=cfg, grid=grid)
    dt = time.time() - t0
    log(f"  V4-{name} {dt:.0f}s — val MAE={res.val_mae_3d:.4f} m, p95={res.val_p95_3d:.4f} m")
    pipe_dir = out_dir / f"v4_{name}"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res.head.state_dict(), pipe_dir / "head.pt")
    val_set = set(res.val_recording_ids)
    val_keys: list[str] = []
    for s in samples:
        key = f"{Path(s.source_dir).name}/{s.recording_id}"
        if key in val_set:
            val_keys.append(key)
    np.savez(
        pipe_dir / "val_predictions.npz",
        predictions=res.val_predictions,
        targets=res.val_targets,
        init_xyz=res.val_init_xyz,
        residuals=res.val_residuals,
        recording_keys=np.asarray(val_keys, dtype="U64"),
    )
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    quick: bool = False,
    *,
    run_sync_audit: bool = False,
    run_v0_baselines: bool = False,
) -> dict:
    """Run the end-to-end V1-V5 pipeline and return the metrics dict.

    Args:
        quick: halve epoch counts at every stage for a smoke run.
        run_sync_audit: also run the opt-in cross-modal sync audit (Stage 0).
        run_v0_baselines: also run the opt-in V0 reference baselines (Stage 1).
    """
    # Determinism — Python / NumPy / PyTorch RNGs pinned, deterministic
    # algorithms enabled where available (warn_only so non-deterministic
    # kernels fall through rather than crash).  BLAS thread scheduling
    # variance is bounded, not eliminated — `multi_seed.py` remains the
    # canonical mean ± std reporter for publication numbers.
    import os
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.manual_seed(42)
    np.random.seed(42)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError):
        # Older torch lacks the API; a few ops have no deterministic kernel
        # even under warn_only. Determinism here is best-effort.
        pass

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "full_pipeline_b5_cma" + ("_quick" if quick else "")
    out_dir = REPO_ROOT / "results" / "runs" / f"{timestamp}__{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("v1", "v2", "v3", "v4", "v5_1"):
        (out_dir / sub).mkdir(exist_ok=True)

    log = _make_logger(out_dir)
    metrics: dict = {
        "quick": quick,
        "variant": "b5_cma",
        "timestamp": timestamp,
        "stages": {},
        "timings_s": {},
    }

    stage_t0_ref = [time.time()]

    def _stage_done(name: str) -> None:
        dt = time.time() - stage_t0_ref[0]
        metrics["timings_s"][name] = dt
        log(f"=== stage '{name}' complete in {dt:.0f}s ===\n")
        stage_t0_ref[0] = time.time()

    log(f"REPO_ROOT = {REPO_ROOT}")
    log(f"out_dir = {out_dir}")
    log(f"quick = {quick}, variant = b5_cma")

    # ----------------------------------------------------------------- data
    # Loaders are built dynamically from the registry — adding a future
    # dataset is a YAML edit (configs/datasets/dN.yaml).  Subsets per
    # downstream stage are chosen below, not by hardcoding loaders here.
    LOADERS_BY_ID: dict[str, TestDatasetLoader] = {}
    for meta in REGISTRY:
        if not meta.root.exists():
            log(f"  skipping {meta.id} (root does not exist: {meta.root})")
            continue
        log(f"  loading {meta.id} from {meta.root.relative_to(REPO_ROOT)} ...")
        LOADERS_BY_ID[meta.id] = resolved_loader(f"{meta.id}.yaml")

    # SSL stages (V1, V2) — user direction: D5 has no operating-mode label
    # and is reserved for V3/V4 only, so the SSL cohort stays D1..D4.
    SSL_IDS = [i for i in ("d1", "d2", "d3", "d4") if i in LOADERS_BY_ID]
    SSL_LOADERS = [LOADERS_BY_ID[i] for i in SSL_IDS]
    # Anomaly stage (V3) — D5 contributes both healthy density-fit data and
    # held-out knock anomalies (label_scheme=d5_healthy_or_knock).
    ANOM_IDS = [i for i in ("d1", "d2", "d3", "d4", "d5") if i in LOADERS_BY_ID]
    ANOM_LOADERS = [LOADERS_BY_ID[i] for i in ANOM_IDS]
    log(f"SSL cohort: {SSL_IDS} | Anomaly cohort: {ANOM_IDS}")

    # Backward-compat per-loader names used by stage-specific helpers below
    # (transition FPR, labeled-segment gathering, RQ3 evaluation, ...).
    # Stage-specific code paths still index loaders by their fixed role
    # (D2 = rectangular bench rig, D3/D4 = circular rig, ...), so the
    # aliases stay but are now dict-driven and trivially extend to D5.
    D1 = LOADERS_BY_ID.get("d1")
    D2 = LOADERS_BY_ID.get("d2")
    D3 = LOADERS_BY_ID.get("d3")
    D4 = LOADERS_BY_ID.get("d4")
    D5 = LOADERS_BY_ID.get("d5")

    # ===================================================== S0 / S1 (opt-in)
    # The cross-modal sync audit and the V0 reference baselines are expensive
    # and off by default; enable them with run_sync_audit / run_v0_baselines
    # (or the matching CLI flags) for an ad-hoc or full-provenance run.
    if run_sync_audit:
        log("\n=== Stage 0 — cross-modal sync verification + correction audit ===")
        try:
            metrics["stages"]["sync_correction"] = _audit_sync(SSL_LOADERS, log)
        except Exception as e:
            log(f"sync audit failed: {type(e).__name__}: {e}")
            metrics["stages"]["sync_correction"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
        _stage_done("stage_0_sync")

    if run_v0_baselines:
        log("=== Stage 1 — V0 baselines (RQ2 anomaly trio+KDE / LightGBM / SRP-PHAT) ===")
        metrics["stages"]["v0"] = _run_v0(SSL_LOADERS, log, anom_loaders=ANOM_LOADERS)
        _stage_done("stage_1_v0")

    # ================================================================= S2
    log("=== Stage 2 — V1 + V2 with b5_cma intervention ===")
    v1_cfg = v1_config(quick)
    v2_cfg_base = v2_config(quick)
    # b5_cma: CMA loss on with cma_weight=0.5 and tightened temperature.
    # Source of truth: `scripts/campaigns/run_v1_v2_only.py::_apply_variant("b5_cma")`.
    v2_cfg = replace(v2_cfg_base, cma_weight=0.5, cma_temperature=0.1)
    log(f"V1 config: epochs={v1_cfg.epochs}, n_mels={v1_cfg.n_mels}, use_cwt={v1_cfg.use_cwt}")
    log(f"V2 config: epochs={v2_cfg.epochs}, cma_weight={v2_cfg.cma_weight}, "
        f"cma_temperature={v2_cfg.cma_temperature}, "
        f"context_mode={v2_cfg.context_mode}, "
        f"acoustic_dropout_p={v2_cfg.acoustic_dropout_p}, "
        f"vibration_dropout_p={v2_cfg.vibration_dropout_p}")

    log("V1 acoustic — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_a = train_v1_per_modality(SSL_LOADERS, modality="acoustic", cfg=v1_cfg)
    log(f"  V1 acoustic {time.time()-t0:.0f}s — sanity NMI={v1_a.sanity_gate.get('nmi',0):.3f} "
        f"ARI={v1_a.sanity_gate.get('ari',0):.3f} purity={v1_a.sanity_gate.get('purity',0):.3f}")
    torch.save(v1_a.encoder.state_dict(), out_dir / "v1" / "acoustic.pt")
    metrics["stages"]["v1_acoustic"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_a.train_loss_history[-1],
        "val_loss_final": v1_a.val_loss_history[-1],
        "sanity_nmi": v1_a.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_a.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_a.sanity_gate.get("purity", 0.0),
        "sanity_n_windows": v1_a.sanity_gate.get("n_windows", 0),
        "sanity_label_set": list(v1_a.sanity_gate.get("label_set", ())),
        "n_train_recordings": len(v1_a.train_recording_ids),
        "n_val_recordings": len(v1_a.val_recording_ids),
    }

    log("V1 vibration — training on D1+D2+D3+D4 healthy ...")
    t0 = time.time()
    v1_v = train_v1_per_modality(SSL_LOADERS, modality="vibration", cfg=v1_cfg)
    log(f"  V1 vibration {time.time()-t0:.0f}s — sanity NMI={v1_v.sanity_gate.get('nmi',0):.3f} "
        f"ARI={v1_v.sanity_gate.get('ari',0):.3f} purity={v1_v.sanity_gate.get('purity',0):.3f}")
    torch.save(v1_v.encoder.state_dict(), out_dir / "v1" / "vibration.pt")
    metrics["stages"]["v1_vibration"] = {
        "epochs": v1_cfg.epochs,
        "train_loss_final": v1_v.train_loss_history[-1],
        "val_loss_final": v1_v.val_loss_history[-1],
        "sanity_nmi": v1_v.sanity_gate.get("nmi", 0.0),
        "sanity_ari": v1_v.sanity_gate.get("ari", 0.0),
        "sanity_purity": v1_v.sanity_gate.get("purity", 0.0),
        "sanity_label_set": list(v1_v.sanity_gate.get("label_set", ())),
    }

    log("V2 — training fusion (inherits V1 weights) with b5_cma ...")
    t0 = time.time()
    v2 = train_v2_fusion(
        SSL_LOADERS, cfg=v2_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    log(f"  V2 {time.time()-t0:.0f}s — RQ1 NMI={v2.rq1.get('nmi',0):.3f} "
        f"ARI={v2.rq1.get('ari',0):.3f} purity={v2.rq1.get('purity',0):.3f}")
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

    # V2 A1 ablation: drop vibration.
    log("V2 A1 ablation (drop_vibration=True) ...")
    t0 = time.time()
    a1_cfg = replace(v2_cfg, drop_vibration=True)
    v2_a1 = train_v2_fusion(
        SSL_LOADERS, cfg=a1_cfg,
        v1_acoustic_state_dict=v1_a.encoder.state_dict(),
        v1_vibration_state_dict=v1_v.encoder.state_dict(),
    )
    log(f"  V2 A1 {time.time()-t0:.0f}s — NMI={v2_a1.rq1.get('nmi',0):.3f}")
    metrics["stages"]["v2_a1_drop_vibration"] = {
        "rq1_nmi": v2_a1.rq1.get("nmi", 0.0),
        "rq1_ari": v2_a1.rq1.get("ari", 0.0),
        "rq1_purity": v2_a1.rq1.get("purity", 0.0),
    }

    # V2 modality-balance probe (the headline Phase-B metric).
    try:
        from ..context.v2_ssl import _gather_labeled_segments
        labeled_segs = _gather_labeled_segments(SSL_LOADERS, v2_cfg)
        probe = run_modality_balance_probe(
            v2.encoder, labeled_segs, v2_cfg=v2_cfg, n_clusters=3, seed=v2_cfg.seed,
        )
        log(f"V2 modality probe: both NMI={probe.both.get('nmi',0):.3f}, "
            f"acoustic_only={probe.acoustic_only.get('nmi',0):.3f}, "
            f"vibration_only={probe.vibration_only.get('nmi',0):.3f}")
        metrics["stages"]["v2_modality_probe"] = {
            "both": {k: v for k, v in probe.both.items() if k not in ("confusion", "cluster_idx")},
            "acoustic_only": {k: v for k, v in probe.acoustic_only.items() if k not in ("confusion", "cluster_idx")},
            "vibration_only": {k: v for k, v in probe.vibration_only.items() if k not in ("confusion", "cluster_idx")},
            "n_segments": len(probe.healthy_segments_used),
            "delta_nmi_both_minus_acoustic": float(
                probe.both.get("nmi", 0.0) - probe.acoustic_only.get("nmi", 0.0)
            ),
        }
    except Exception as e:
        log(f"V2 modality probe skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v2_modality_probe"] = {"skipped": f"{type(e).__name__}: {e}"}

    _stage_done("stage_2_v1_v2_b5_cma")

    # ================================================================= S3
    log("=== Stage 3 — V3 three paradigms (acoustic / vibration / fusion) ===")
    v3_cfg = v3_config(quick)
    log(f"V3 config: epochs={v3_cfg.epochs}, K={v3_cfg.n_threshold_clusters}, "
        f"percentile={v3_cfg.threshold_percentile}")

    v3_acoustic_adapter = V3AcousticOnlyAdapter(v1_a.encoder)
    v3_vibration_adapter = V3VibrationOnlyAdapter(v1_v.encoder)

    v3_results: dict = {}
    pipelines = [
        ("acoustic", v3_acoustic_adapter),
        ("vibration", v3_vibration_adapter),
        ("fusion", v2.encoder),
    ]
    metrics["stages"]["v3_three_paradigms"] = {}
    for name, enc in pipelines:
        try:
            res = _train_one_v3(name, enc, ANOM_LOADERS, v2_cfg, v3_cfg, out_dir, log)
            v3_results[name] = res
            metrics["stages"]["v3_three_paradigms"][name] = {
                "val_nll_final": float(res.val_nll[-1]),
                "val_nll_min_final": float(res.val_nll_min[-1]) if res.val_nll_min else float("nan"),
                "val_nll_max_final": float(res.val_nll_max[-1]) if res.val_nll_max else float("nan"),
                "p95_per_cluster": res.thresholds.p95.tolist(),
                "p99_per_cluster": res.thresholds.p99.tolist(),
                "n_val_windows": int(res.val_scores.shape[0]),
                "n_threshold_fit_recordings": len(res.threshold_fit_recording_ids),
                "n_val_eval_recordings": len(res.val_recording_ids),
                "n_clusters_fit": int(res.thresholds.centroids.shape[0]),
            }
        except Exception as e:
            log(f"V3-{name} FAILED: {type(e).__name__}: {e}")
            metrics["stages"]["v3_three_paradigms"][name] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()

    # Mirror v3-fusion artefacts into out_dir/v3/ so the legacy archive
    # layout used by `archive.py` and other downstream tools still finds
    # them at the expected path.
    if "fusion" in v3_results:
        import shutil
        for fname in ("flow.pt", "thresholds.npz", "val_eval.npz"):
            src = out_dir / "v3_fusion" / fname
            if src.exists():
                shutil.copy2(src, out_dir / "v3" / fname)

    _stage_done("stage_3_v3_three_paradigms")

    # ================================================================= S4
    log("=== Stage 4 — V3 fusion deeper diagnostics ===")
    if "fusion" not in v3_results:
        log("Skipped — V3 fusion failed in Stage 3")
        metrics["stages"]["v3_fusion_depth"] = {"skipped": "v3_fusion failed"}
    else:
        v3 = v3_results["fusion"]
        v3_depth: dict = {}
        v3_a2 = None

        # A2 unconditional flow ablation.
        try:
            log("V3 A2 ablation (unconditional flow) ...")
            t0 = time.time()
            a2_cfg = replace(v3_cfg, unconditional=True)
            v3_a2 = train_v3_cnf(v2.encoder, ANOM_LOADERS, v2_cfg=v2_cfg, v3_cfg=a2_cfg)
            log(f"  V3 A2 {time.time()-t0:.0f}s — val NLL={v3_a2.val_nll[-1]:.3f}")
            v3_depth["a2_unconditional"] = {
                "val_nll_final": float(v3_a2.val_nll[-1]),
                "p99_per_cluster": v3_a2.thresholds.p99.tolist(),
            }

            # Paired bootstrap V3 vs A2 on per-window NLL.
            if (
                v3.val_scores.shape[0] == v3_a2.val_scores.shape[0]
                and v3.val_scores.shape[0] >= 4
            ):
                pt = paired_bootstrap_test(
                    v3.val_scores, v3_a2.val_scores,
                    lower_is_better=True, n_boot=1000, seed=v3_cfg.seed,
                )
                log(f"  V3 vs A2 paired test: Δ={pt.delta_point:.3f} "
                    f"[{pt.delta_ci_low:.3f}, {pt.delta_ci_high:.3f}] "
                    f"p={pt.p_value_two_sided:.4f}")
                v3_depth["v3_vs_a2_paired_test"] = {
                    "delta_point": pt.delta_point,
                    "delta_ci95_low": pt.delta_ci_low,
                    "delta_ci95_high": pt.delta_ci_high,
                    "p_value_two_sided": pt.p_value_two_sided,
                    "direction": pt.direction,
                    "n_paired": int(v3.val_scores.shape[0]),
                    "method": "paired_percentile_bootstrap_1000",
                }
        except Exception as e:
            log(f"V3 A2 / paired test skipped: {type(e).__name__}: {e}")
            v3_depth["a2_unconditional"] = {"skipped": f"{type(e).__name__}: {e}"}

        # Synthetic anomaly ROC-AUC across SNR ladder.
        try:
            if v3.val_contexts.shape[0] >= 4:
                import torch.utils.data as _tud

                from ..anomaly.v3_trainer import _extract_xc
                from ..context.v2_ssl import (
                    _collate,
                    _gather_paired_segments,
                    _PairedGroupedBatchSampler,
                    _PairedWindowedDataset,
                    _split_segments_by_recording,
                )
                segs_all = _gather_paired_segments(ANOM_LOADERS, v2_cfg)
                _, _val_segs_full = _split_segments_by_recording(
                    segs_all, v3_cfg.val_ratio, v3_cfg.seed,
                )
                _, val_segs_for_auc = _split_segments_by_recording(
                    _val_segs_full, v3_cfg.threshold_fit_val_ratio, v3_cfg.seed + 1,
                )
                val_ds = _PairedWindowedDataset(val_segs_for_auc, v2_cfg)
                if len(val_ds) > 0:
                    val_loader = _tud.DataLoader(
                        val_ds,
                        batch_sampler=_PairedGroupedBatchSampler(
                            val_ds, v3_cfg.batch_size, shuffle=False, seed=v3_cfg.seed,
                        ),
                        collate_fn=_collate,
                    )
                    x_val, c_val, _ = _extract_xc(v2.encoder, val_loader, resolve_device(v3_cfg.device))
                    auc = evaluate_synthetic_anomaly_auc(
                        v3.flow, x_val.numpy(), c_val.numpy(),
                        snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                        n_boot=500, seed=v3_cfg.seed,
                    )
                    log("V3 synthetic-anomaly ROC-AUC:")
                    for snr in sorted(auc.snr_db_to_auc):
                        log(f"  SNR={snr:+.1f} dB: AUC={auc.snr_db_to_auc[snr]:.3f}")
                    v3_depth["synthetic_anomaly_auc"] = {
                        "auc_conditional": auc.snr_db_to_auc,
                        "auc_conditional_ci_low": auc.snr_db_to_auc_ci_low,
                        "auc_conditional_ci_high": auc.snr_db_to_auc_ci_high,
                        "n_clean": auc.snr_db_to_n_clean,
                    }
                    if v3_a2 is not None:
                        auc_a2 = evaluate_synthetic_anomaly_auc(
                            v3_a2.flow, x_val.numpy(), c_val.numpy(),
                            snr_db_list=(-10.0, -5.0, 0.0, 5.0, 10.0),
                            n_boot=500, seed=v3_cfg.seed,
                        )
                        v3_depth["synthetic_anomaly_auc"]["auc_unconditional"] = auc_a2.snr_db_to_auc
        except Exception as e:
            log(f"V3 synthetic AUC skipped: {type(e).__name__}: {e}")

        # Transition FPR (within-D1 + cross-dataset same-mode).
        try:
            _val_eval_set = set(v3.val_recording_ids)
            paired_by: dict[tuple[str, str], list] = {}
            paired_fb: dict[tuple[str, str], list] = {}
            for L in (D1, D2):
                for s in L.list_segments():
                    if s.mode_label is None or s.is_anomaly:
                        continue
                    p = precompute_paired(s, v2_cfg)
                    if p is None:
                        continue
                    key = (s.dataset_id, s.mode_label)
                    qual = f"{Path(s.source_dir).name}/{s.recording_id}"
                    (paired_by if qual in _val_eval_set else paired_fb).setdefault(key, []).append(p)
            for key, segs in paired_fb.items():
                if key not in paired_by:
                    log(f"  transition fallback for {key} (training-pool recording)")
                    paired_by[key] = segs

            pairs = [
                ("d1_pump_to_turbine", ("d1", "Pump"), ("d1", "Turbine"), "raw"),
                ("d1_turbine_to_pump", ("d1", "Turbine"), ("d1", "Pump"), "raw"),
                ("d1_to_d2_pump", ("d1", "Pump"), ("d2", "Pump"), "encoder"),
                ("d1_to_d2_turbine", ("d1", "Turbine"), ("d2", "Turbine"), "encoder"),
            ]
            transition_results: dict[str, float] = {}
            for plabel, ka, kb, level in pairs:
                if ka not in paired_by or kb not in paired_by:
                    log(f"  transition {plabel}: skipped (missing source)")
                    continue
                seg_a, seg_b = paired_by[ka][0], paired_by[kb][0]
                if level == "raw":
                    out_pair = transition_fpr(
                        v2.encoder, v3.flow, v3.thresholds, seg_a, seg_b,
                        v2_cfg=v2_cfg, crossfade_seconds=1.0,
                        percentile=v3_cfg.threshold_percentile,
                    )
                else:
                    out_pair = encoder_level_transition_fpr(
                        v2.encoder, v3.flow, v3.thresholds, seg_a, seg_b,
                        v2_cfg=v2_cfg, n_crossfade_windows=8,
                        percentile=v3_cfg.threshold_percentile,
                    )
                transition_results[plabel] = out_pair["fpr"]
                log(f"  transition {plabel} ({level}): fpr={out_pair['fpr']:.3f} "
                    f"({out_pair['n_alerts']}/{out_pair['n_windows']})")
            v3_depth["transition_fpr"] = transition_results
        except Exception as e:
            log(f"V3 transition FPR skipped: {type(e).__name__}: {e}")

        # Per-cluster threshold breakdown on healthy holdout.
        try:
            if v3.val_scores.size > 0:
                breakdown = per_cluster_alert_breakdown(
                    v3.thresholds, v3.val_contexts, v3.val_scores,
                    percentile=v3_cfg.threshold_percentile,
                )
                v3_depth["per_cluster_breakdown_healthy"] = breakdown
                log("V3 per-cluster healthy alert rates: "
                    + " | ".join(
                        f"k{k}={r['n_alerts']}/{r['n']}({r['alert_rate']:.2f})"
                        for k, r in breakdown["per_cluster"].items() if r["n"] > 0
                    ))
        except Exception as e:
            log(f"V3 per-cluster breakdown skipped: {type(e).__name__}: {e}")

        # Sliding-window event extraction per anomaly cohort.
        try:
            log("V3 sliding-window event extraction (stride=0.25s) ...")
            n_clusters = int(v3.thresholds.centroids.shape[0])
            per_cluster_p95 = v3.thresholds.p95.tolist()
            per_cluster_p90: list[float] = []
            if v3.val_scores.size > 0 and v3.val_contexts.shape[0] > 0:
                assign = v3.thresholds.assign(v3.val_contexts)
                for k in range(n_clusters):
                    mask = assign == k
                    if int(mask.sum()) > 4:
                        per_cluster_p90.append(float(np.percentile(v3.val_scores[mask], 90)))
                    else:
                        per_cluster_p90.append(float(np.percentile(v3.val_scores, 90)))
            else:
                per_cluster_p90 = list(v3.thresholds.p95)

            cohort_event_summary: dict = {}
            for cohort_label, loader, dsid in (
                ("d2_random_fault", D2, "d2"),
                ("d3_hit", D3, "d3"),
                ("d4_random_fault", D4, "d4"),
            ):
                events: list = []
                n_rec = 0
                for s in loader.list_segments():
                    if not s.is_anomaly:
                        continue
                    seg = precompute_paired(s, v2_cfg)
                    if seg is None:
                        continue
                    try:
                        times_s, scores, contexts = sliding_window_v3_inference(
                            v2.encoder, v3.flow, seg,
                            v2_cfg=v2_cfg, inference_stride_s=0.25,
                            xt_pool=v3.xt_pool,
                            device=resolve_device(v3_cfg.device),
                        )
                    except Exception as inner:
                        log(f"    {cohort_label}/{s.recording_id} skipped: {inner}")
                        continue
                    if scores.size == 0:
                        continue
                    w_clusters = v3.thresholds.assign(contexts)
                    rec_high = float(np.median([per_cluster_p95[int(k)] for k in w_clusters]))
                    rec_low = float(np.median([per_cluster_p90[int(k)] for k in w_clusters]))
                    if rec_low > rec_high:
                        rec_low = rec_high
                    evs = detect_events_from_score_timeline(
                        scores, times_s, high_threshold=rec_high, low_threshold=rec_low,
                        min_duration_s=0.10, max_gap_windows=0,
                        recording_id=s.recording_id, dataset_id=dsid,
                        window_seconds=v2_cfg.window_seconds,
                    )
                    events.extend(evs)
                    n_rec += 1
                summary = summarise_events(events)
                summary["n_recordings_audited"] = n_rec
                cohort_event_summary[cohort_label] = summary
                log(f"  {cohort_label}: n_rec={n_rec} n_events={summary['n_events']}")
            v3_depth["sliding_window_events"] = cohort_event_summary
        except Exception as e:
            log(f"V3 sliding-window events skipped: {type(e).__name__}: {e}")

        # Real-anomaly detection vs weak knock GT.  Scores V3's detected
        # events against impulse-derived knock intervals on the sparse-anomaly
        # cohorts (precision / recall / F1 / onset-timing).  This is a
        # prerequisite metric: V4 cannot be trusted until V3 detects the real
        # anomalies well.
        try:
            from ..anomaly.event_detection import v3_real_anomaly_detection
            rf_segments = []
            for loader, _dsid in ((D4, "d4"), (D2, "d2"), (D5, "d5")):
                if loader is None:
                    continue
                rf_segments += [s for s in loader.list_segments() if s.is_anomaly]
            real_det = v3_real_anomaly_detection(
                v2.encoder, v3.flow, v3.thresholds, rf_segments,
                v2_cfg=v2_cfg, percentile=v3_cfg.threshold_percentile,
                inference_stride_s=0.25, xt_pool=v3.xt_pool, device=v3_cfg.device,
            )
            v3_depth["real_anomaly_detection"] = real_det
            metrics["stages"]["v3_real_anomaly"] = real_det
            log(f"V3 real-anomaly: P={real_det['precision']:.3f} "
                f"R={real_det['recall']:.3f} F1={real_det['f1']:.3f} "
                f"onset_err={real_det['median_onset_error_s']:.3f}s "
                f"(scored {real_det['n_recordings_scored']} recs, "
                f"{real_det['n_recordings_no_weak_gt']} had no weak GT)")
        except Exception as e:
            log(f"V3 real-anomaly detection skipped: {type(e).__name__}: {e}")
            metrics["stages"]["v3_real_anomaly"] = {"skipped": f"{type(e).__name__}: {e}"}

        metrics["stages"]["v3_fusion_depth"] = v3_depth
    _stage_done("stage_4_v3_depth")

    # ================================================================= S5
    log("=== Stage 5 — V4 four paradigms + V0 classical localisation ===")
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segments = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segments)
    d3_labeled = [s for s in d3_segments if s.recording_id in overrides]
    d4_labeled = [
        s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None
    ]
    # B1 (2026-05-23) — D5 knock recordings carry parsed positions
    # (`d5_healthy_or_knock` scheme → spatial_label set, is_anomaly=True) and
    # `d5.yaml` explicitly lists them as V4 localisation labels, but the
    # cohort builder previously concatenated only D2/D3/D4 and silently
    # dropped D5.  Including D5 roughly doubles the position inventory.
    d5_labeled = [
        s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None
    ] if D5 is not None else []
    n_positions = len({
        tuple(np.round(s.spatial_label, 3)) for s in
        (d2_labeled + d3_labeled + d4_labeled + d5_labeled)
        if s.spatial_label is not None
    })
    log(f"Labelled segments: D2={len(d2_labeled)} D3={len(d3_labeled)} "
        f"D4={len(d4_labeled)} D5={len(d5_labeled)} | distinct positions={n_positions}")

    grid = V4_CANDIDATE_GRID

    log("Precomputing V4 samples (burst-aware SRP-PHAT + accel TDOA + V2 c_t) ...")
    t0 = time.time()
    v4_samples = precompute_v4_samples(
        v2.encoder, d2_labeled + d3_labeled + d4_labeled + d5_labeled,
        v2_cfg=v2_cfg, grid=grid,
        spatial_label_overrides=overrides,
        burst_aware_srp=True, burst_seconds=0.10,
        # Pool `x_for_v3` with V3's pooling so gating-time NLL matches the
        # manifold the flow was trained on (avoids the PMA-2/mean saturation).
        v3_xt_pool=getattr(v3, "xt_pool", None),
    )
    log(f"  {len(v4_samples)} V4 samples in {time.time()-t0:.0f}s")
    n_with_multilat = sum(1 for s in v4_samples if s.multilat_xyz is not None)
    log(f"  multilat init available on {n_with_multilat}/{len(v4_samples)} samples")
    metrics["stages"]["v4_samples"] = {
        "n_total": len(v4_samples),
        "n_with_multilat": n_with_multilat,
    }

    v4_cfg = v4_config(quick)
    v4_paradigms = [
        ("acoustic", "srp_only"),
        ("vibration", "vibration_only_learned"),
        ("vibration_tdoa_only_legacy", "tdoa_only"),
        ("fusion", "both"),
    ]
    v4_results: dict = {}
    metrics["stages"]["v4_four_paradigms"] = {}
    if len(v4_samples) < 4:
        log(f"V4 SKIPPED — only {len(v4_samples)} labelled samples (need ≥4)")
        metrics["stages"]["v4_four_paradigms"] = {"skipped": True}
    else:
        for name, mode in v4_paradigms:
            try:
                if mode == "vibration_only_learned" and n_with_multilat < len(v4_samples):
                    log(f"  filtering to {n_with_multilat} samples with multilat for {name}")
                    samples_in = [s for s in v4_samples if s.multilat_xyz is not None]
                else:
                    samples_in = v4_samples
                res = _train_one_v4(name, mode, samples_in, grid, v4_cfg, out_dir, log)
                v4_results[name] = res
                metrics["stages"]["v4_four_paradigms"][name] = {
                    "channel_mode": mode,
                    "val_mae_3d": float(res.val_mae_3d),
                    "val_mae_ci95_low": float(res.val_mae_ci_low),
                    "val_mae_ci95_high": float(res.val_mae_ci_high),
                    "val_p95_3d": float(res.val_p95_3d),
                    "n_val": int(res.val_predictions.shape[0]),
                    "n_train_recordings": len(res.train_recording_ids),
                    "n_val_recordings": len(res.val_recording_ids),
                    "train_loss_final": float(res.train_loss_history[-1]) if res.train_loss_history else float("nan"),
                    "val_loss_final": float(res.val_loss_history[-1]) if res.val_loss_history else float("nan"),
                }
            except Exception as e:
                log(f"V4-{name} FAILED: {type(e).__name__}: {e}")
                metrics["stages"]["v4_four_paradigms"][name] = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()

        if "fusion" in v4_results:
            import shutil
            for fname in ("head.pt", "val_predictions.npz"):
                src = out_dir / "v4_fusion" / fname
                if src.exists():
                    shutil.copy2(src, out_dir / "v4" / fname)

    log("V0 accel multilateration per dataset ...")
    metrics["stages"]["v0_multilateration"] = _run_v0_multilateration(
        [D2, D3, D4], overrides, log
    )

    # ============================================== Stage 5b — spatial holdout
    # Train fusion V4 on all positions EXCEPT the reserved held-out set,
    # then report holdout MAE (localise-an-unseen-position), the V3-GATED
    # holdout MAE (deployment-faithful: V4 only fires on V3-flagged windows),
    # and V0 multilateration on the same held-out samples.  Gated by
    # `gated_v4_eval` (CLI --ungated disables the gating column).
    log("=== Stage 5b — V4 spatial-holdout + V3-gated eval ===")
    try:
        from ..localization import split_samples_by_position
        train_pos, holdout_pos = split_samples_by_position(
            v4_samples, V4_HOLDOUT_POSITIONS_M,
        )
        n_hold_pos = len({tuple(np.round(s.target_xyz, 3)) for s in holdout_pos})
        log(f"  spatial split: {len(train_pos)} train / {len(holdout_pos)} holdout "
            f"samples across {n_hold_pos} held-out positions")
        if len(train_pos) >= 4 and len(holdout_pos) >= 1:
            sh: dict = {"n_train_samples": len(train_pos),
                        "n_holdout_samples": len(holdout_pos),
                        "n_holdout_positions": n_hold_pos,
                        "holdout_positions_m": [list(p) for p in V4_HOLDOUT_POSITIONS_M]}
            res_sh = train_v4_localization(
                v4_samples, cfg=replace(v4_cfg, channel_mode="both"), grid=grid,
                explicit_split=(train_pos, holdout_pos),
            )
            sh["holdout_mae_ungated_m"] = float(res_sh.val_mae_3d)
            sh["holdout_p95_ungated_m"] = float(res_sh.val_p95_3d)
            sh["holdout_train_val_gap_m"] = float(abs(
                (res_sh.val_loss_history[-1] if res_sh.val_loss_history else float("nan"))
                - (res_sh.train_loss_history[-1] if res_sh.train_loss_history else float("nan"))
            ))
            log(f"  holdout MAE (ungated) = {res_sh.val_mae_3d:.4f} m")

            # V3-gated holdout: keep holdout windows V3 flags as anomalous,
            # scored directly on each sample's cached x_for_v3 + context (the
            # direct-path gate).  Replaces the legacy interval-overlap matching
            # (`_v3_event_intervals_for_recordings` → `window_overlaps_any`),
            # whose recording-id collisions + timeline drift produced
            # n_holdout_gated=0.  See `..localization.v3_gating`.
            gated_v4_eval = True
            if gated_v4_eval and "fusion" in v3_results:
                try:
                    from ..localization.v3_gating import gate_samples_by_v3
                    v3f = v3_results["fusion"]
                    gres = gate_samples_by_v3(
                        v3f.flow, v3f.thresholds, holdout_pos,
                        percentile=(99 if int(v3_cfg.threshold_percentile) >= 99 else 95),
                    )
                    keep = gres.keep_mask
                    sh["v3_gating_diagnostic"] = gres.per_recording
                    if keep.shape[0] == res_sh.val_predictions.shape[0] and keep.any():
                        err = np.linalg.norm(
                            res_sh.val_predictions[keep] - res_sh.val_targets[keep], axis=-1)
                        sh["holdout_mae_v3gated_m"] = float(np.mean(err))
                        sh["n_holdout_gated"] = int(keep.sum())
                        log(f"  holdout MAE (V3-gated) = {np.mean(err):.4f} m "
                            f"on {int(keep.sum())}/{keep.shape[0]} V3-flagged windows")
                    else:
                        sh["holdout_mae_v3gated_m"] = None
                        sh["n_holdout_gated"] = int(keep.sum()) if keep.size else 0
                        log("  V3-gated holdout: no windows flagged (or shape mismatch)")
                except Exception as e:
                    log(f"  V3-gated holdout skipped: {type(e).__name__}: {e}")
                    sh["holdout_mae_v3gated_m"] = None

            # V0 multilateration on the same held-out samples (recording-level).
            try:
                hold_recs = {s.recording_id for s in holdout_pos}
                v0_errs: list[float] = []
                for payload in metrics["stages"]["v0_multilateration"].values():
                    for rec in payload.get("per_recording", []):
                        if rec.get("recording_id") in hold_recs and "error_m" in rec:
                            v0_errs.append(float(rec["error_m"]))
                sh["holdout_v0_multilat_mae_m"] = float(np.mean(v0_errs)) if v0_errs else None
                sh["n_holdout_v0"] = len(v0_errs)
                if v0_errs and "holdout_mae_v3gated_m" in sh and sh["holdout_mae_v3gated_m"] is not None:
                    sh["delta_v4gated_minus_v0_m"] = sh["holdout_mae_v3gated_m"] - float(np.mean(v0_errs))
                    log(f"  V0 multilat on holdout = {np.mean(v0_errs):.4f} m | "
                        f"Δ(V4gated − V0) = {sh['delta_v4gated_minus_v0_m']:+.4f} m")
            except Exception as e:
                log(f"  V0-on-holdout skipped: {type(e).__name__}: {e}")
            metrics["stages"]["v4_spatial_holdout"] = sh
        else:
            log("  spatial-holdout SKIPPED — insufficient train/holdout samples")
            metrics["stages"]["v4_spatial_holdout"] = {
                "skipped": f"train={len(train_pos)} holdout={len(holdout_pos)}"}
    except Exception as e:
        log(f"Stage 5b spatial-holdout skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v4_spatial_holdout"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    _stage_done("stage_5_v4_four_paradigms")

    # ================================================================= S6
    log("=== Stage 6 — V4 fusion deeper diagnostics ===")
    if "fusion" not in v4_results:
        log("Skipped — V4 fusion failed or skipped in Stage 5")
        metrics["stages"]["v4_fusion_depth"] = {"skipped": "v4_fusion unavailable"}
    else:
        v4 = v4_results["fusion"]
        v4_depth: dict = {}
        try:
            log("V4 A3 ablation (unconditional=True) ...")
            t0 = time.time()
            a3_cfg = replace(v4_cfg, unconditional=True)
            v4_a3 = train_v4_localization(v4_samples, cfg=a3_cfg, grid=grid)
            log(f"  V4 A3 {time.time()-t0:.0f}s — val MAE={v4_a3.val_mae_3d:.4f} m")
            v4_depth["a3_unconditional"] = {
                "val_mae_3d": float(v4_a3.val_mae_3d),
                "val_mae_ci95_low": float(v4_a3.val_mae_ci_low),
                "val_mae_ci95_high": float(v4_a3.val_mae_ci_high),
                "val_p95_3d": float(v4_a3.val_p95_3d),
            }
            if (
                v4.val_predictions.shape[0] == v4_a3.val_predictions.shape[0]
                and v4.val_predictions.shape[0] >= 4
            ):
                err_v4 = np.linalg.norm(v4.val_predictions - v4.val_targets, axis=-1).astype(np.float64)
                err_a3 = np.linalg.norm(v4_a3.val_predictions - v4_a3.val_targets, axis=-1).astype(np.float64)
                pt = paired_bootstrap_test(
                    err_v4, err_a3, lower_is_better=True, n_boot=1000, seed=v4_cfg.seed,
                )
                log(f"  V4 vs A3 paired test: Δ_MAE={pt.delta_point*1000:.1f} mm "
                    f"[{pt.delta_ci_low*1000:.1f}, {pt.delta_ci_high*1000:.1f}] mm "
                    f"p={pt.p_value_two_sided:.4f}")
                v4_depth["v4_vs_a3_paired_test"] = {
                    "delta_mae_m": pt.delta_point,
                    "delta_mae_ci95_low_m": pt.delta_ci_low,
                    "delta_mae_ci95_high_m": pt.delta_ci_high,
                    "p_value_two_sided": pt.p_value_two_sided,
                    "direction": pt.direction,
                    "n_paired_windows": int(err_v4.shape[0]),
                    "method": "paired_percentile_bootstrap_1000",
                }
        except Exception as e:
            log(f"V4 A3 skipped: {type(e).__name__}: {e}")
            v4_depth["a3_unconditional"] = {"skipped": f"{type(e).__name__}: {e}"}

        metrics["stages"]["v4_fusion_depth"] = v4_depth
    _stage_done("stage_6_v4_depth")

    # ================================================================= S7
    log("=== Stage 7 — V5.1 fan-noise robustness conditioning ===")
    try:
        speed_lookup = {**d3_speed_lookup(d3_segments), **d3_speed_lookup(D4.list_segments())}
        if speed_lookup and len(v4_samples) >= 4:
            log(f"  speed lookup: {len(speed_lookup)} recordings")
            scada_dim = next(iter(speed_lookup.values())).shape[0]
            v5_1_samples = []
            for s in v4_samples:
                scada = speed_lookup.get(s.recording_id)
                if scada is None:
                    scada = np.zeros(scada_dim, dtype=np.float32)
                v5_1_samples.append(V4Sample(
                    srp_volume=s.srp_volume, tdoa_tokens=s.tdoa_tokens,
                    context=s.context, x_for_v3=s.x_for_v3,
                    target_xyz=s.target_xyz, scada=scada,
                    mode_label=s.mode_label, recording_id=s.recording_id,
                    source_dir=s.source_dir, dataset_id=s.dataset_id,
                    multilat_xyz=s.multilat_xyz,
                ))
            v5_1_cfg = v4_config(quick, scada_dim=scada_dim)
            t0 = time.time()
            v5_1 = train_v4_localization(v5_1_samples, cfg=v5_1_cfg, grid=grid)
            log(f"  V5.1 {time.time()-t0:.0f}s — val MAE={v5_1.val_mae_3d:.3f} m "
                f"[{v5_1.val_mae_ci_low:.3f}, {v5_1.val_mae_ci_high:.3f}]")
            torch.save(v5_1.head.state_dict(), out_dir / "v5_1" / "head_speed.pt")
            metrics["stages"]["v5_1"] = {
                "scada_dim": scada_dim,
                "val_mae_3d": float(v5_1.val_mae_3d),
                "val_mae_ci95_low": float(v5_1.val_mae_ci_low),
                "val_mae_ci95_high": float(v5_1.val_mae_ci_high),
                "val_p95_3d": float(v5_1.val_p95_3d),
            }
        else:
            log("V5.1 SKIPPED — no speed segments or insufficient V4 samples")
            metrics["stages"]["v5_1"] = {"skipped": True}
    except Exception as e:
        log(f"V5.1 skipped: {type(e).__name__}: {e}")
        metrics["stages"]["v5_1"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    _stage_done("stage_7_v5_1")

    # =============================================== deep-vs-simple summary
    # One-stop comparison block for the thesis chapter: each deep stage's
    # headline metric vs the closed-form / classical baseline already
    # computed elsewhere in this pipeline.  A near-zero or negative Δ on
    # any row means the deep model has not earned its complexity for that
    # stage on the current cohort.
    log("\n=== Deep-vs-simple summary ===")
    deep_vs_simple: dict = {}

    # V3 fusion vs KDE-on-c_t (per-cluster gaussian_kde on V2 c_t buckets).
    try:
        from ..anomaly.kde_baseline import fit_and_score_kde_on_ct

        v3_fus = v3_results.get("fusion")
        if v3_fus is not None and v3_fus.train_x is not None and v3_fus.val_x is not None:
            kde_res = fit_and_score_kde_on_ct(
                x_train=v3_fus.train_x,
                c_train=v3_fus.train_contexts,
                x_val=v3_fus.val_x,
                c_val=v3_fus.val_contexts,
                n_clusters=v3_cfg.n_threshold_clusters,
                seed=v3_cfg.seed,
            )
            v3_nll_val = float(v3_fus.val_nll[-1]) if v3_fus.val_nll else float("nan")
            delta_nll = v3_nll_val - kde_res.val_nll_mean  # CNF wins if < 0
            deep_vs_simple["anomaly"] = {
                "deep_model": "V3 CNF (fusion)",
                "simple_baseline": "KDE-on-c_t per K-means cluster",
                "deep_val_nll_mean": v3_nll_val,
                "simple_val_nll_mean": kde_res.val_nll_mean,
                "delta_deep_minus_simple": delta_nll,
                "deep_wins": delta_nll < 0.0,
                "n_clusters_used": kde_res.n_clusters_used,
                "kde_n_per_cluster_train": kde_res.n_per_cluster_train.tolist(),
                "kde_n_per_cluster_val": kde_res.n_per_cluster_val.tolist(),
            }
            log(f"  V3 vs KDE: V3 NLL={v3_nll_val:.3f} | KDE NLL={kde_res.val_nll_mean:.3f} | "
                f"Δ={delta_nll:+.3f} ({'V3 wins' if delta_nll < 0 else 'KDE wins'})")
        else:
            deep_vs_simple["anomaly"] = {"skipped": "v3_fusion train_x/val_x unavailable"}
    except Exception as e:
        log(f"  V3 vs KDE skipped: {type(e).__name__}: {e}")
        deep_vs_simple["anomaly"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    # V4 fusion vs V0 accel-TDOA multilateration (closed-form, no trainable
    # parameters).  Both already computed above; this block surfaces the Δ.
    try:
        v4_fus_metrics = metrics["stages"].get("v4_four_paradigms", {}).get("fusion", {})
        v0_multi = metrics["stages"].get("v0_multilateration", {})
        v4_mae = v4_fus_metrics.get("val_mae_3d")
        # Pool V0 MAE across D2/D3/D4 (mean of per-dataset means, weighted
        # by n_successful).  Single value comparable to V4's pooled val MAE.
        v0_errs: list[float] = []
        for payload in v0_multi.values():
            n = int(payload.get("n_successful", 0))
            mean = payload.get("mean_error_m")
            if n > 0 and isinstance(mean, (int, float)) and not (isinstance(mean, float) and (mean != mean)):
                v0_errs.extend([float(mean)] * n)
        v0_mae = float(np.mean(v0_errs)) if v0_errs else float("nan")
        if isinstance(v4_mae, (int, float)) and v0_errs:
            delta_mae = float(v4_mae) - v0_mae  # V4 wins if < 0
            deep_vs_simple["localisation"] = {
                "deep_model": "V4 fusion head",
                "simple_baseline": "V0 accel-TDOA multilateration (closed-form)",
                "deep_val_mae_m": float(v4_mae),
                "simple_val_mae_m": v0_mae,
                "delta_deep_minus_simple_m": delta_mae,
                "deep_wins": delta_mae < 0.0,
                "n_recordings_v0": len(v0_errs),
            }
            log(f"  V4 vs V0 multilat: V4 MAE={v4_mae:.3f} m | V0 MAE={v0_mae:.3f} m | "
                f"Δ={delta_mae:+.3f} m ({'V4 wins' if delta_mae < 0 else 'V0 wins'})")
        else:
            deep_vs_simple["localisation"] = {"skipped": "v4_fusion or v0_multilateration unavailable"}
    except Exception as e:
        log(f"  V4 vs V0 multilat skipped: {type(e).__name__}: {e}")
        deep_vs_simple["localisation"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    # V2 fusion clustering vs LightGBM mode classifier (D1).  Metrics are not
    # directly comparable (NMI vs macro-F1) so we report both side-by-side
    # without a Δ; the thesis chapter can frame the comparison qualitatively.
    try:
        v2_metrics = metrics["stages"].get("v2", {})
        v0_lgbm = metrics["stages"].get("v0", {}).get("v0_lgbm_d1", {})
        if v2_metrics and v0_lgbm and "val_macro_f1" in v0_lgbm:
            deep_vs_simple["mode_clustering"] = {
                "deep_model": "V2 fusion (K=3 K-means on c_t)",
                "simple_baseline": "V0 LightGBM mode classifier (D1)",
                "deep_rq1_nmi": v2_metrics.get("rq1_nmi"),
                "deep_rq1_purity": v2_metrics.get("rq1_purity"),
                "simple_val_macro_f1": v0_lgbm.get("val_macro_f1"),
                "note": "metrics not directly comparable; reported side-by-side",
            }
            log(f"  V2 vs V0 LGBM: V2 NMI={v2_metrics.get('rq1_nmi', 0):.3f} | "
                f"V0 F1={v0_lgbm.get('val_macro_f1', 0):.3f} (different units)")
    except Exception as e:
        log(f"  V2 vs V0 LGBM skipped: {type(e).__name__}: {e}")
        deep_vs_simple["mode_clustering"] = {"skipped_reason": f"{type(e).__name__}: {e}"}

    metrics["deep_vs_simple"] = deep_vs_simple
    _stage_done("stage_7b_deep_vs_simple")

    # ============================================================ persist
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"\nWrote interim metrics to {metrics_path}")

    manifest = {
        "timestamp": timestamp,
        "label": label,
        "variant": "b5_cma",
        "quick": quick,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "host": socket.gethostname(),
        "configs": {
            "v1_cfg": asdict(v1_cfg),
            "v2_cfg": asdict(v2_cfg),
            "v3_cfg": asdict(v3_cfg),
            "v4_cfg": asdict(v4_cfg),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"Wrote manifest to {out_dir / 'manifest.json'}")

    # ================================================================= S8
    log("\n=== Stage 8 — late-fusion (AND / OR / score-weighted / MAX) eval ===")
    try:
        rel_run = str(out_dir.relative_to(REPO_ROOT))
        rc = subprocess.call(
            [sys.executable, "-m", "src.modeling.eval.rq2_three_paradigm_eval",
             "--v3-three-run", rel_run, "--source-run", rel_run],
            cwd=str(REPO_ROOT),
        )
        if rc == 0:
            comparison = out_dir / "rq2_paradigm_comparison.json"
            if comparison.exists():
                metrics["stages"]["rq2_paradigm_comparison"] = json.loads(comparison.read_text())
                log("  rq2 eval done — see rq2_paradigm_comparison.{json,md}")
        else:
            log(f"  rq2 eval exited with rc={rc}")
            metrics["stages"]["rq2_paradigm_comparison"] = {"skipped_reason": f"rc={rc}"}
    except Exception as e:
        log(f"rq2 eval skipped: {type(e).__name__}: {e}")
        metrics["stages"]["rq2_paradigm_comparison"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    _stage_done("stage_8_rq2_lf_eval")

    # ================================================================= S9
    log("=== Stage 9 — RQ3 LF confidence-gated localisation eval ===")
    try:
        rel_run = str(out_dir.relative_to(REPO_ROOT))
        rc = subprocess.call(
            [sys.executable, "-m", "src.modeling.eval.rq3_three_paradigm_eval",
             "--v4-three-run", rel_run],
            cwd=str(REPO_ROOT),
        )
        if rc == 0:
            comp = out_dir / "rq3_paradigm_comparison.json"
            if comp.exists():
                metrics["stages"]["rq3_paradigm_comparison"] = json.loads(comp.read_text())
                log("  rq3 eval done — see rq3_paradigm_comparison.{json,md}")
        else:
            log(f"  rq3 eval exited with rc={rc}")
            metrics["stages"]["rq3_paradigm_comparison"] = {"skipped_reason": f"rc={rc}"}
    except Exception as e:
        log(f"rq3 eval skipped: {type(e).__name__}: {e}")
        metrics["stages"]["rq3_paradigm_comparison"] = {"skipped_reason": f"{type(e).__name__}: {e}"}
    _stage_done("stage_9_rq3_lf_eval")

    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"Final metrics written to {metrics_path}")

    total = sum(metrics["timings_s"].values())
    log(f"\nTotal wall-clock: {total:.0f}s ({total/60:.1f} min)")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick", action="store_true",
        help="Halve epoch counts at every stage for a smoke run (~25 min CPU).",
    )
    p.add_argument(
        "--sync-audit", action="store_true",
        help="Also run the opt-in cross-modal sync audit (Stage 0).",
    )
    p.add_argument(
        "--v0-baselines", action="store_true",
        help="Also run the opt-in V0 reference baselines (Stage 1).",
    )
    args = p.parse_args()
    main(quick=args.quick, run_sync_audit=args.sync_audit, run_v0_baselines=args.v0_baselines)
