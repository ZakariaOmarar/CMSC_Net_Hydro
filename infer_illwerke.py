"""CLI: Apply a frozen trained pipeline to new campaign data.

Usage
-----
python infer_illwerke.py \\
    --config configs/illwerke/pipeline.yaml \\
    --artifact-dir results/illwerke/pipeline \\
    --output-dir results/illwerke/infer_20230428
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Illwerke pipeline on new data")
    parser.add_argument("--config",       required=True, help="pipeline.yaml path")
    parser.add_argument("--artifact-dir", required=True, help="Directory with trained artifacts")
    parser.add_argument("--output-dir",   required=True, help="Where to write inference outputs")
    parser.add_argument("--days", nargs="+", default=None,
                        help="YYYYMMDD campaign days (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    artifact_dir = Path(args.artifact_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data_root"])
    d_cfg = cfg.get("data", {})
    l1_cfg = cfg.get("layer1", {})
    l3_cfg = cfg.get("layer3", {})
    l4_cfg = cfg.get("layer4", {})
    l5_cfg = cfg.get("layer5", {})

    print("=" * 60)
    print("Illwerke ROWII — Pipeline Inference")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Output dir  : {out}")
    print("=" * 60)

    # --- Load campaign ---------------------------------------------------
    t0 = time.time()
    from src.ingestion.illwerke_loader import load_campaign
    from src.features.rms_temporal import compute_net_head

    days = args.days or d_cfg.get("days")
    campaign = load_campaign(
        data_root,
        days=days,
        forced_drop_channels=d_cfg.get("forced_drop_channels"),
        allg_drop_channels=d_cfg.get("allg_drop_channels"),
    )
    T = campaign.rms.shape[0]
    print(f"[Data] {T} timesteps ({T/3600:.1f} h)  — {time.time()-t0:.1f}s")

    net_head = compute_net_head(
        campaign.allg, campaign.channel_names_allg,
        upper_channel=d_cfg.get("upper_head_channel", "Oberwasserpegel"),
        lower_channel=d_cfg.get("lower_level_channel", "UW_Pegel_Rodund"),
    )

    # --- Layer 1: Oracle (frozen thresholds) ----------------------------
    from src.modeling.mode.p1_physics.thresholds import load_thresholds
    from src.modeling.mode.p1_physics.oracle import run_oracle

    thresholds = load_thresholds(artifact_dir / "signal_thresholds.json")
    oracle = run_oracle(
        campaign.allg, campaign.channel_names_allg, thresholds,
        steady_window_s=l1_cfg.get("steady_window_s", 30),
    )
    np.save(out / "oracle_labels.npy", oracle.labels)
    np.save(out / "oracle_confidence.npy", oracle.confidence)
    np.save(out / "oracle_steady_mask.npy", oracle.steady_mask)
    print("[L1] Oracle done")

    # --- Layer 2: Transition Typing ------------------------------------
    from src.modeling.mode.p2_transitions.typing import type_transitions, save_transition_segments
    from src.modeling.mode.p1_physics.oracle import OracleResult

    l2_cfg = cfg.get("layer2", {})
    segments_l2 = type_transitions(
        oracle, campaign.allg, campaign.channel_names_allg, thresholds,
        micro_event_threshold_s=l2_cfg.get("micro_event_threshold_s", 60),
        search_window_s=l2_cfg.get("search_window_s", 3600),
    )
    save_transition_segments(segments_l2, out / "transition_segments.json")
    print(f"[L2] {len(segments_l2)} transition intervals typed")

    # --- Layer 3: RMS Features + LightGBM (frozen) --------------------
    from src.modeling.mode.p3_rms.features import RMSFeatureTransform, apply_rms_features
    from src.modeling.mode.p3_rms.lgbm_classifier import load_lgbm, predict_proba

    transform = RMSFeatureTransform.load(artifact_dir / "feature_transform.npz")
    X_feat, cav_mask = apply_rms_features(
        campaign.rms, net_head, transform,
        cavitation_kurtosis_threshold=l3_cfg.get("cavitation_kurtosis_threshold", 8.0),
    )
    np.save(out / "X_feat.npy", X_feat)

    lgbm_path = artifact_dir / "lgbm_model.txt"
    if lgbm_path.exists():
        clf = load_lgbm(lgbm_path)
        proba = predict_proba(clf, X_feat)
        np.save(out / "lgbm_proba.npy", proba)
    else:
        proba = None
        print("[L3] No LightGBM model found — using oracle-only emissions in L4")
    print("[L3] Features and predictions done")

    # --- Layer 4: Smoother (frozen topology) --------------------------
    from src.modeling.mode.p4_smoother.hmm_fixed import (
        smooth, build_mode_timeline, save_mode_timeline,
    )
    from src.modeling.mode.p4_smoother.topology import validate_sequence

    state_seq, seg4 = smooth(
        oracle.labels, oracle.confidence, proba,
        micro_event_threshold_s=l4_cfg.get("micro_event_threshold_s", 60),
    )
    np.save(out / "smoothed_state_sequence.npy", state_seq)

    violations = validate_sequence(state_seq)
    status = "PASS" if not violations else f"FAIL ({len(violations)} violations)"
    print(f"[L4] Topology check: {status}")

    timeline = build_mode_timeline(seg4, segments_l2, campaign.timestamps_ns)
    save_mode_timeline(timeline, out / "mode_timeline.json")
    print(f"[L4] {len(timeline)} mode events → {out}/mode_timeline.json")

    # --- Layer 5: Anomaly Scoring -------------------------------------
    from src.modeling.mode.p3_rms.per_mode_ae import load_aes, compute_reconstruction_errors
    from src.modeling.mode.p5_anomaly.per_mode_baseline import compute_anomaly_scores_fast, save_anomaly_scores
    from src.modeling.mode.p5_anomaly.sub_modes import discover_tu_sub_modes
    from src.modeling.mode.p5_anomaly.events import detect_anomaly_events, save_anomaly_events, anomaly_summary

    ae_cfg = l3_cfg.get("ae", {})
    aes = load_aes(artifact_dir, d_feat=ae_cfg.get("d_feat", 33),
                   hidden=ae_cfg.get("hidden", 32), bottleneck=ae_cfg.get("bottleneck", 16))
    recon_errors = compute_reconstruction_errors(aes, X_feat, state_seq)

    z_scores, alert_level = compute_anomaly_scores_fast(
        recon_errors, state_seq,
        baseline_window_s=l5_cfg.get("baseline_window_s", 86400),
        min_baseline_samples=l5_cfg.get("min_baseline_samples", 300),
        watch_sigma=l5_cfg.get("watch_sigma", 4.0),
        alert_sigma=l5_cfg.get("alert_sigma", 6.0),
    )
    save_anomaly_scores(z_scores, alert_level, out)

    sub_labels = discover_tu_sub_modes(
        state_seq, campaign.allg, campaign.channel_names_allg,
        band_fractions=l5_cfg.get("tu_load_bands", [0.0, 0.20, 0.60, 0.90, 1.01]),
    )
    np.save(out / "tu_sub_mode_labels.npy", sub_labels)

    events = detect_anomaly_events(
        z_scores, alert_level, state_seq, campaign.timestamps_ns, sub_labels,
        min_event_duration_s=l5_cfg.get("min_event_duration_s", 5),
        merge_gap_s=l5_cfg.get("merge_gap_s", 30),
    )
    save_anomaly_events(events, out / "anomaly_events.json")

    from src.modeling.mode.p4_smoother.topology import IDX_TO_STATE
    mode_seconds = {IDX_TO_STATE[s]: int(np.sum(state_seq == s)) for s in range(4)}
    summary = anomaly_summary(events, mode_seconds)
    with open(out / "anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[L5] {len(events)} anomaly events detected")
    print("\n" + "=" * 60)
    print(f"Inference complete → {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
