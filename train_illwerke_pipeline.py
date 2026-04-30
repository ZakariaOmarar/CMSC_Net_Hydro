"""CLI: Train the 5-layer physics-first mode detection pipeline.

Usage
-----
python train_illwerke_pipeline.py --config configs/illwerke/pipeline.yaml
python train_illwerke_pipeline.py --config configs/illwerke/pipeline.yaml --layers 1 2
python train_illwerke_pipeline.py --config configs/illwerke/pipeline.yaml --force-retrain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _need(*paths: Path) -> bool:
    """Return True if any artifact is missing (checkpoint not found)."""
    return any(not p.exists() for p in paths)


def run_layer1(cfg: dict, data_root: Path, out: Path, campaign) -> dict:
    """Layer 1: Physics Oracle."""
    from src.ingestion.illwerke_loader import IllwerkeCampaign
    from src.features.rms_temporal import compute_net_head
    from src.modeling.mode.p1_physics.thresholds import derive_thresholds, save_thresholds
    from src.modeling.mode.p1_physics.oracle import run_oracle
    from src.modeling.mode.p1_physics.validation import build_validation_report, save_report

    l1_cfg = cfg.get("layer1", {})
    d_cfg = cfg.get("data", {})

    thr_path   = out / "signal_thresholds.json"
    labels_path = out / "oracle_labels.npy"
    conf_path   = out / "oracle_confidence.npy"
    steady_path = out / "oracle_steady_mask.npy"
    report_path = out / "validation_report_L1.json"

    force = cfg.get("force_retrain", False)

    if not force and not _need(thr_path, labels_path, conf_path):
        print("[L1] Checkpoints found — skipping (use --force-retrain to override)")
        from src.modeling.mode.p1_physics.thresholds import load_thresholds
        from src.modeling.mode.p1_physics.oracle import OracleResult
        return {
            "thresholds_path": thr_path,
            "labels":   np.load(labels_path),
            "confidence": np.load(conf_path),
            "steady_mask": np.load(steady_path),
        }

    print("[L1] Deriving thresholds from Allg_M1 data ...")
    thresholds = derive_thresholds(
        campaign.allg,
        campaign.channel_names_allg,
        random_state=l1_cfg.get("gmm_random_state", 42),
        debounce_s=l1_cfg.get("debounce_s", 5),
        gate_hysteresis_pct=l1_cfg.get("gate_hysteresis_pct", 2.0),
        valve_hysteresis_pct=l1_cfg.get("valve_hysteresis_pct", 2.0),
        windage_ceiling_mw=l1_cfg.get("windage_ceiling_mw", 10.0),
        stable_gate_pct=l1_cfg.get("stable_gate_pct", 0.5),
        stable_rpm_rpm=l1_cfg.get("stable_rpm_rpm", 5.0),
        stable_power_mw=l1_cfg.get("stable_power_mw", 5.0),
    )
    save_thresholds(thresholds, thr_path)
    print(f"[L1] Thresholds saved -> {thr_path}")
    print(f"     v_grid_kv={thresholds.v_grid_connected_kv:.2f}  "
          f"rpm_spinning={thresholds.rpm_spinning:.1f}  "
          f"gate_open={thresholds.gate_open_pct:.1f}%  "
          f"power_gen={thresholds.power_generating_mw:.1f} MW")

    print("[L1] Running physics oracle ...")
    from src.features.rms_temporal import compute_net_head
    net_head = compute_net_head(
        campaign.allg,
        campaign.channel_names_allg,
        upper_channel=d_cfg.get("upper_head_channel", "Oberwasserpegel"),
        lower_channel=d_cfg.get("lower_level_channel", "UW_Pegel_Rodund"),
    )

    oracle = run_oracle(
        campaign.allg,
        campaign.channel_names_allg,
        thresholds,
        steady_window_s=l1_cfg.get("steady_window_s", 30),
    )

    np.save(labels_path, oracle.labels)
    np.save(conf_path, oracle.confidence)
    np.save(steady_path, oracle.steady_mask)

    report = build_validation_report(
        oracle,
        campaign.allg,
        campaign.channel_names_allg,
        net_head=net_head,
        alpha=l1_cfg.get("head_independence_alpha", 0.05),
        freeze_window_s=l1_cfg.get("freeze_window_s", 60),
    )
    save_report(report, report_path)

    print(f"[L1] Done. Steady coverage: {report['steady_coverage_pct']:.1f}%")
    dwell = report.get("dwell_ratios", {})
    for mode in ["ST", "TU", "PU", "PH", "TRANSITION", "UNKNOWN"]:
        pct = dwell.get(mode, 0.0) * 100
        print(f"     {mode}: {pct:.1f}%")

    return {
        "thresholds_path": thr_path,
        "labels":     oracle.labels,
        "confidence": oracle.confidence,
        "steady_mask": oracle.steady_mask,
        "thresholds": thresholds,
    }


def run_layer2(cfg: dict, out: Path, campaign, layer1_out: dict) -> dict:
    """Layer 2: Transition Typing."""
    from src.modeling.mode.p1_physics.oracle import OracleResult
    from src.modeling.mode.p2_transitions.typing import type_transitions, save_transition_segments
    from src.modeling.mode.p1_physics.thresholds import load_thresholds

    l2_cfg = cfg.get("layer2", {})
    segs_path = out / "transition_segments.json"
    force = cfg.get("force_retrain", False)

    if not force and not _need(segs_path):
        print("[L2] Checkpoint found — skipping")
        from src.modeling.mode.p2_transitions.typing import load_transition_segments
        return {"segments": load_transition_segments(segs_path)}

    thresholds = layer1_out.get("thresholds") or load_thresholds(layer1_out["thresholds_path"])
    oracle = OracleResult(
        labels=layer1_out["labels"],
        confidence=layer1_out["confidence"],
        steady_mask=layer1_out["steady_mask"],
    )

    print("[L2] Typing transition intervals ...")
    segments = type_transitions(
        oracle,
        campaign.allg,
        campaign.channel_names_allg,
        thresholds,
        micro_event_threshold_s=l2_cfg.get("micro_event_threshold_s", 60),
        search_window_s=l2_cfg.get("search_window_s", 3600),
    )
    save_transition_segments(segments, segs_path)
    typed = [s for s in segments if s.transition_type not in ("MICRO_EVENT", "INVALID_TOPOLOGY", "UNKNOWN")]
    print(f"[L2] {len(segments)} intervals typed  ({len(typed)} with known type)")
    by_type: dict[str, int] = {}
    for s in segments:
        by_type[s.transition_type] = by_type.get(s.transition_type, 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"     {k}: {v}")

    return {"segments": segments}


def run_layer3(cfg: dict, out: Path, campaign, layer1_out: dict, layer2_out: dict) -> dict:
    """Layer 3: RMS Distillation."""
    from src.features.rms_temporal import compute_net_head
    from src.modeling.mode.p3_rms.features import fit_rms_transform, apply_rms_features, RMSFeatureTransform
    from src.modeling.mode.p3_rms.lgbm_classifier import (
        build_training_set, train_lgbm, save_lgbm, predict_proba,
    )
    from src.modeling.mode.p3_rms.per_mode_ae import train_per_mode_ae, save_aes

    l3_cfg = cfg.get("layer3", {})
    d_cfg  = cfg.get("data", {})
    feat_path  = out / "feature_transform.npz"
    xfeat_path = out / "X_feat.npy"
    lgbm_path  = out / "lgbm_model.txt"
    proba_path = out / "lgbm_proba.npy"
    force = cfg.get("force_retrain", False)

    # Features
    if force or _need(feat_path, xfeat_path):
        print("[L3] Fitting RMS feature transform ...")
        net_head = compute_net_head(
            campaign.allg, campaign.channel_names_allg,
            upper_channel=d_cfg.get("upper_head_channel", "Oberwasserpegel"),
            lower_channel=d_cfg.get("lower_level_channel", "UW_Pegel_Rodund"),
        )
        transform = fit_rms_transform(
            campaign.rms,
            campaign.channel_names_rms,
            net_head,
            forced_drop_channels=d_cfg.get("forced_drop_channels", []),
            head_poly_degree=l3_cfg.get("head_poly_degree", 2),
            train_fraction=l3_cfg.get("train_day_fraction", 0.75),
            windows=tuple(l3_cfg.get("feature_windows_s", [5, 60, 300])),
        )
        transform.save(feat_path)
        X_feat, cav_mask = apply_rms_features(
            campaign.rms, net_head, transform,
            cavitation_kurtosis_threshold=l3_cfg.get("cavitation_kurtosis_threshold", 8.0),
        )
        np.save(xfeat_path, X_feat)
        np.save(out / "cav_mask.npy", cav_mask)
    else:
        print("[L3] Feature transform checkpoint found — loading ...")
        transform = RMSFeatureTransform.load(feat_path)
        X_feat = np.load(xfeat_path)

    # LightGBM
    if force or _need(lgbm_path):
        print("[L3] Building training set and training LightGBM ...")
        X_train, y_train, day_idx = build_training_set(
            X_feat,
            layer1_out["labels"],
            layer1_out["confidence"],
            layer2_out["segments"],
            campaign.timestamps_ns,
            max_steady_ratio=l3_cfg.get("max_steady_ratio", 5),
            random_seed=cfg.get("seed", 42),
        )
        print(f"     {len(X_train)} training samples, {len(np.unique(y_train))} classes")
        lgbm_cfg = l3_cfg.get("lgbm", {})
        clf = train_lgbm(X_train, y_train, **lgbm_cfg)
        save_lgbm(clf, lgbm_path)
        print(f"[L3] LightGBM saved -> {lgbm_path}")

        proba = predict_proba(clf, X_feat)
        np.save(proba_path, proba)
    else:
        print("[L3] LightGBM checkpoint found — loading ...")
        from src.modeling.mode.p3_rms.lgbm_classifier import load_lgbm
        clf = load_lgbm(lgbm_path)
        proba = np.load(proba_path) if proba_path.exists() else predict_proba(clf, X_feat)

    # Per-mode autoencoders
    ae_any_missing = any(not (out / f"ae_{m}.pt").exists() for m in ["ST", "TU", "PU", "PH"])
    if force or ae_any_missing:
        print("[L3] Training per-mode autoencoders ...")
        ae_cfg = l3_cfg.get("ae", {})
        aes = train_per_mode_ae(
            X_feat,
            layer1_out["labels"],
            layer1_out["confidence"],
            **ae_cfg,
        )
        save_aes(aes, out)
        print(f"[L3] AEs saved -> {out}/ae_*.pt")
    else:
        print("[L3] AE checkpoints found — skipping")

    return {"X_feat": X_feat, "lgbm_proba": proba}


def run_layer4(cfg: dict, out: Path, campaign, layer1_out: dict, layer2_out: dict, layer3_out: dict) -> dict:
    """Layer 4: HMM Smoother."""
    from src.modeling.mode.p4_smoother.hmm_fixed import (
        smooth, build_mode_timeline, save_mode_timeline,
    )
    from src.modeling.mode.p4_smoother.topology import validate_sequence, IDX_TO_STATE

    l4_cfg = cfg.get("layer4", {})
    seq_path      = out / "smoothed_state_sequence.npy"
    timeline_path = out / "mode_timeline.json"
    force = cfg.get("force_retrain", False)

    if not force and not _need(seq_path, timeline_path):
        print("[L4] Checkpoints found — skipping")
        return {"state_seq": np.load(seq_path)}

    print("[L4] Running Viterbi smoother ...")
    state_seq, segments = smooth(
        layer1_out["labels"],
        layer1_out["confidence"],
        layer3_out.get("lgbm_proba"),
        micro_event_threshold_s=l4_cfg.get("micro_event_threshold_s", 60),
    )
    np.save(seq_path, state_seq)

    violations = validate_sequence(state_seq)
    if violations:
        print(f"  [WARNING] {len(violations)} topology violations detected!")
    else:
        print("  Topology check: PASS (zero violations)")

    timeline = build_mode_timeline(
        segments,
        layer2_out["segments"],
        campaign.timestamps_ns,
    )
    save_mode_timeline(timeline, timeline_path)

    # Dwell summary
    from collections import Counter
    label_counts: Counter = Counter()
    for seg in segments:
        label_counts[seg.label] += seg.duration_s
    total_s = sum(label_counts.values()) or 1
    print("[L4] Smoothed dwell times:")
    for mode in ["ST", "TU", "PU", "PH"]:
        pct = 100.0 * label_counts.get(mode, 0) / total_s
        print(f"     {mode}: {pct:.1f}%")
    print(f"  mode_timeline.json: {len(timeline)} events -> {timeline_path}")

    return {"state_seq": state_seq, "segments": segments}


def run_layer5(cfg: dict, out: Path, campaign, layer3_out: dict, layer4_out: dict) -> None:
    """Layer 5: Anomaly Detection."""
    from src.modeling.mode.p3_rms.per_mode_ae import load_aes, compute_reconstruction_errors
    from src.modeling.mode.p5_anomaly.per_mode_baseline import compute_anomaly_scores_fast, save_anomaly_scores
    from src.modeling.mode.p5_anomaly.sub_modes import discover_tu_sub_modes
    from src.modeling.mode.p5_anomaly.events import detect_anomaly_events, save_anomaly_events, anomaly_summary

    l5_cfg = cfg.get("layer5", {})
    force = cfg.get("force_retrain", False)

    print("[L5] Loading per-mode autoencoders ...")
    ae_cfg = cfg.get("layer3", {}).get("ae", {})
    aes = load_aes(out, d_feat=ae_cfg.get("d_feat", 33), hidden=ae_cfg.get("hidden", 32),
                   bottleneck=ae_cfg.get("bottleneck", 16))

    print("[L5] Computing reconstruction errors ...")
    from src.modeling.mode.p1_physics.oracle import LABEL_CODE
    recon_errors = compute_reconstruction_errors(
        aes, layer3_out["X_feat"], layer4_out["state_seq"],
    )

    np.save(out / "anomaly_reconstruction_errors.npy", recon_errors)

    print("[L5] Computing anomaly scores ...")
    z_scores, alert_level = compute_anomaly_scores_fast(
        recon_errors,
        layer4_out["state_seq"],
        baseline_window_s=l5_cfg.get("baseline_window_s", 86400),
        min_baseline_samples=l5_cfg.get("min_baseline_samples", 300),
        watch_sigma=l5_cfg.get("watch_sigma", 4.0),
        alert_sigma=l5_cfg.get("alert_sigma", 6.0),
    )
    save_anomaly_scores(z_scores, alert_level, out)

    print("[L5] Discovering TU sub-modes ...")
    sub_labels = discover_tu_sub_modes(
        layer4_out["state_seq"], campaign.allg, campaign.channel_names_allg,
        band_fractions=l5_cfg.get("tu_load_bands", [0.0, 0.20, 0.60, 0.90, 1.01]),
    )
    np.save(out / "tu_sub_mode_labels.npy", sub_labels)

    events = detect_anomaly_events(
        z_scores, alert_level, layer4_out["state_seq"], campaign.timestamps_ns,
        sub_labels,
        min_event_duration_s=l5_cfg.get("min_event_duration_s", 5),
        merge_gap_s=l5_cfg.get("merge_gap_s", 30),
    )
    save_anomaly_events(events, out / "anomaly_events.json")

    from collections import Counter
    mode_seconds: dict[str, int] = {}
    for state_id in range(4):
        from src.modeling.mode.p4_smoother.topology import IDX_TO_STATE
        mode_seconds[IDX_TO_STATE[state_id]] = int(np.sum(layer4_out["state_seq"] == state_id))
    summary = anomaly_summary(events, mode_seconds)
    with open(out / "anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    watch = summary["watch_events"]
    alert = summary["alert_events"]
    print(f"[L5] {len(events)} events ({watch} watch, {alert} alert)")
    for mode, stats in summary.get("per_mode", {}).items():
        print(f"     {mode}: {stats['n_events']} events, alert_rate={stats['alert_rate_pct']:.3f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Illwerke physics-first pipeline")
    parser.add_argument("--config", default="configs/illwerke/pipeline.yaml")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which layers to run (default: 1 2 3 4 5)")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.force_retrain:
        cfg["force_retrain"] = True

    data_root = Path(cfg["data_root"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    d_cfg = cfg.get("data", {})
    layers_to_run = set(args.layers)

    print("=" * 60)
    print("Illwerke ROWII — Physics-First Mode Detection Pipeline")
    print(f"Data root : {data_root}")
    print(f"Output dir: {out}")
    print(f"Layers    : {sorted(layers_to_run)}")
    print("=" * 60)

    # Load campaign (always needed)
    print("\n[Data] Loading campaign ...")
    t0 = time.time()
    from src.ingestion.illwerke_loader import load_campaign
    campaign = load_campaign(
        data_root,
        days=d_cfg.get("days"),
        forced_drop_channels=d_cfg.get("forced_drop_channels"),
        allg_drop_channels=d_cfg.get("allg_drop_channels"),
    )
    T = campaign.rms.shape[0]
    print(f"[Data] {T} timesteps ({T/3600:.1f} h)  — elapsed: {time.time()-t0:.1f}s")

    layer1_out: dict = {}
    layer2_out: dict = {}
    layer3_out: dict = {}
    layer4_out: dict = {}

    if 1 in layers_to_run:
        t0 = time.time()
        layer1_out = run_layer1(cfg, data_root, out, campaign)
        print(f"[L1] Wall time: {time.time()-t0:.1f}s")

    if 2 in layers_to_run:
        if not layer1_out:
            layer1_out = {
                "labels":      np.load(out / "oracle_labels.npy"),
                "confidence":  np.load(out / "oracle_confidence.npy"),
                "steady_mask": np.load(out / "oracle_steady_mask.npy"),
                "thresholds_path": out / "signal_thresholds.json",
            }
        t0 = time.time()
        layer2_out = run_layer2(cfg, out, campaign, layer1_out)
        print(f"[L2] Wall time: {time.time()-t0:.1f}s")

    if 3 in layers_to_run:
        if not layer1_out:
            layer1_out = {
                "labels":      np.load(out / "oracle_labels.npy"),
                "confidence":  np.load(out / "oracle_confidence.npy"),
                "steady_mask": np.load(out / "oracle_steady_mask.npy"),
                "thresholds_path": out / "signal_thresholds.json",
            }
        if not layer2_out:
            from src.modeling.mode.p2_transitions.typing import load_transition_segments
            layer2_out = {"segments": load_transition_segments(out / "transition_segments.json")}
        t0 = time.time()
        layer3_out = run_layer3(cfg, out, campaign, layer1_out, layer2_out)
        print(f"[L3] Wall time: {time.time()-t0:.1f}s")

    if 4 in layers_to_run:
        if not layer1_out:
            layer1_out = {
                "labels":      np.load(out / "oracle_labels.npy"),
                "confidence":  np.load(out / "oracle_confidence.npy"),
                "steady_mask": np.load(out / "oracle_steady_mask.npy"),
                "thresholds_path": out / "signal_thresholds.json",
            }
        if not layer2_out:
            from src.modeling.mode.p2_transitions.typing import load_transition_segments
            layer2_out = {"segments": load_transition_segments(out / "transition_segments.json")}
        if not layer3_out:
            layer3_out = {"X_feat": np.load(out / "X_feat.npy"),
                          "lgbm_proba": np.load(out / "lgbm_proba.npy") if (out / "lgbm_proba.npy").exists() else None}
        t0 = time.time()
        layer4_out = run_layer4(cfg, out, campaign, layer1_out, layer2_out, layer3_out)
        print(f"[L4] Wall time: {time.time()-t0:.1f}s")

    if 5 in layers_to_run:
        if not layer3_out:
            layer3_out = {"X_feat": np.load(out / "X_feat.npy"),
                          "lgbm_proba": None}
        if not layer4_out:
            layer4_out = {"state_seq": np.load(out / "smoothed_state_sequence.npy")}
        t0 = time.time()
        run_layer5(cfg, out, campaign, layer3_out, layer4_out)
        print(f"[L5] Wall time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("Pipeline complete. Artifacts written to:", out)
    print("=" * 60)


if __name__ == "__main__":
    main()
