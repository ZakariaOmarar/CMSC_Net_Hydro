"""Chart figures driven by thesis-table numbers and archived result files.

Renders (thesis figure plan numbering):
  19  acoustic-collapse evidence            <- results/fusion_forensics_v2_20260515.json
  20  threshold-transfer FPR                <- Results ch. Tables 6 (res_rq2_shift)
  22  latent-SNR conditioning lift          <- Table res_rq2_auc
  23  specificity audit                     <- Table res_rq2_spec
  26  LORO paradigm comparison              <- Table res_rq3_loro
  27  LOPO per-fold distribution            <- results/runs/.../lopo/folds.jsonl
  28  cross-session transfer to D5          <- results/runs/.../cross_dataset/summary.json
  31  SCADA mutual information with mode    <- Table res_rq4_mode
  33  robustness panels (reg grid + seeds)  <- results/reports/finalize_results_*.json

Run with:  python -m scripts.figures.fig_charts
"""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CHANNEL_MODE_COLORS,
    CHANNEL_MODE_LABELS,
    CLASSICAL,
    INTERMEDIATE,
    LATE_FUSION,
    REPO_ROOT,
    VIBRATION,
    save,
)

# Canonical seed-42 run of the five-seed set the Results tables are built on
# (the full_run / lopo_dir of finalize_results_20260617_042101.json).
RUN_DIR = REPO_ROOT / "results" / "runs" / "20260615_112939__full_pipeline_b5_cma"
# LOPO folds of that canonical seed-42 run (tdoa-only mean MAE 0.126 m over 16
# folds; the five-seed median is 0.129 m, Table tab:res_rq3_lopo).
LOPO_RUN_DIR = RUN_DIR
# reg_grid lives only in the earlier finalize report (the May sweep was not
# re-run); the seed strip in panel (b) is read from the canonical five-seed
# aggregate instead, so it matches the Results robustness table.
FINALIZE = REPO_ROOT / "results" / "reports" / "finalize_results_20260611_012822.json"
MULTISEED = REPO_ROOT / "results" / "reports" / "multiseed_complete_20260617_042101.json"
FORENSICS = REPO_ROOT / "results" / "fusion_forensics_v2_20260515.json"


# ─────────────────────────────────────────────────────────────────────────
# 19 — acoustic collapse: gradient sensitivity + cosine invariance
# ─────────────────────────────────────────────────────────────────────────
def fig19_acoustic_collapse() -> None:
    d = json.loads(FORENSICS.read_text())
    grad_ac = d["input_gradients"]["grad_norm_acoustic_mean"]
    grad_vib = d["input_gradients"]["grad_norm_vibration_mean"]
    ratio = d["input_gradients"]["grad_norm_ratio_acoustic_to_vibration"]
    cos_vib0 = d["ct_cross_modal_contribution"]["cosine_ct_full_vs_zero_vib_mean"]
    cos_ac0 = d["ct_cross_modal_contribution"]["cosine_ct_full_vs_zero_ac_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))

    ax1.barh(
        ["acoustic\nstream", "vibration\nstream"],
        [grad_ac, grad_vib],
        color=[ACOUSTIC, VIBRATION],
        height=0.55,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"input-gradient norm  $\Vert\partial\Vert c_t\Vert^2/\partial x\Vert$")
    ax1.set_title("(a) Sensitivity of the context vector")
    ax1.text(
        grad_vib * 1.6,
        1,
        f"{ratio:.0f}:1",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=ANOMALY,
    )

    bars = ax2.barh(
        ["vibration\nzeroed", "acoustic\nzeroed"],
        [cos_vib0, cos_ac0],
        color=[VIBRATION, ACOUSTIC],
        height=0.55,
    )
    ax2.set_xlim(0, 1.05)
    ax2.axvline(1.0, color="0.4", lw=0.8, ls="--")
    ax2.set_xlabel(r"$\cos(c_t,\; c_t\,|\,\mathrm{stream\ zeroed})$")
    ax2.set_title("(b) Context change when one stream is muted")
    for b, v in zip(bars, [cos_vib0, cos_ac0]):
        ax2.text(min(v, 0.97) - 0.02, b.get_y() + b.get_height() / 2,
                 f"{v:.3f}", va="center", ha="right", color="white", fontweight="bold")

    fig.suptitle(
        "Joint-pool fusion is acoustically driven: zeroing vibration barely moves "
        "$c_t$ (single reference encoder)",
        fontsize=9.5, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig19_acoustic_collapse")


# ─────────────────────────────────────────────────────────────────────────
# 20 — threshold-transfer FPR, baselines and heads on one axis
# ─────────────────────────────────────────────────────────────────────────
def fig20_threshold_transfer() -> None:
    # Matched transfer protocol (Results ch., tab:res_rq2_shift): each per-cluster
    # threshold is fitted on one set of healthy operating conditions and the
    # held-out healthy FPR is read on a disjoint set, over five split seeds.
    # Each method is drawn as its across-seed [min, max] range with the median
    # marked; the worst-case (max) is what decides whether a method collapses.
    rows = [
        # (label, median, min, max, seeds_collapsed, colour, is_head)
        ("OC-SVM (ac.)", 0.00, 0.00, 1.00, "1/5", CLASSICAL, False),
        ("$k$-means (ac.)", 0.00, 0.00, 0.34, "1/5", CLASSICAL, False),
        ("KDE (ac.)", 0.00, 0.00, 0.37, "1/5", CLASSICAL, False),
        ("OC-SVM (vib.)", 0.06, 0.02, 0.92, "2/5", CLASSICAL, False),
        ("$k$-means (vib.)", 0.06, 0.04, 0.90, "1/5", CLASSICAL, False),
        ("KDE (vib.)", 0.25, 0.05, 0.93, "3/5", CLASSICAL, False),
        ("V3-acoustic", 0.20, 0.11, 0.31, "3/5", ACOUSTIC, True),
        ("V3-vibration", 0.04, 0.03, 0.12, "0/5", VIBRATION, True),
        ("V3-fusion", 0.16, 0.03, 0.36, "2/5", INTERMEDIATE, True),
        ("Late-fusion AND", 0.004, 0.001, 0.012, "0/5", LATE_FUSION, True),
    ]
    floor = 7e-4  # log-axis floor so exact-zero medians/mins stay visible
    clamp = lambda v: max(v, floor)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(rows))[::-1]  # table order, top to bottom
    for yi, (lbl, med, lo, hi, nco, color, is_head) in zip(y, rows):
        ax.plot([clamp(lo), clamp(hi)], [yi, yi], color=color, lw=2.4, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.plot(clamp(lo), yi, marker="|", ms=8, color=color, mew=1.6, zorder=3)
        ax.plot(clamp(hi), yi, marker="|", ms=8, color=color, mew=1.6, zorder=3)
        ax.plot(clamp(med), yi, marker="o", ms=7, color=color, mec="0.2", mew=0.6,
                zorder=4)
        ax.annotate(f"{nco}", (1.55, yi), va="center", ha="left", fontsize=7.5,
                    color=ANOMALY if nco != "0/5" else "0.45",
                    fontweight="bold" if nco != "0/5" else "normal")

    ax.axvline(0.05, color="0.35", lw=0.9, ls="--")
    ax.text(0.05, len(rows) - 0.3, "0.05 target", va="bottom", ha="center",
            fontsize=7.5, color="0.35")
    ax.axvline(0.20, color=ANOMALY, lw=0.9, ls=":")
    ax.text(0.20, -0.85, "collapse\n($>0.2$)", va="top", ha="center", fontsize=7.5,
            color=ANOMALY)
    ax.text(1.55, len(rows) - 0.3, "seeds\ncollapsed", va="bottom", ha="left",
            fontsize=7.0, color="0.3")

    ax.set_xscale("log")
    ax.set_xlim(floor, 1.4)
    ax.set_ylim(-1.4, len(rows) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("healthy false-positive rate after threshold transfer (log; min--median--max over 5 seeds)")
    # divider between the unconditional baselines (top six rows) and the
    # proposed heads (bottom four rows); the y-axis labels name each group
    ax.axhline(3.5, color="0.8", lw=0.8)
    ax.set_title(
        "Threshold transfer: every unconditional baseline and the acoustic and\n"
        "fusion heads collapse on at least one seed; the AND rule holds at 0.004 (0/5)",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig20_threshold_transfer_fpr")


# ─────────────────────────────────────────────────────────────────────────
# 22 — latent-SNR conditioning lift (tab:res_rq2_auc)
# ─────────────────────────────────────────────────────────────────────────
def fig22_latent_snr_lift() -> None:
    # Five-seed median curves with across-seed [min, max] bands (tab:res_rq2_auc).
    snr = np.array([-10, -5, 0, 5, 10])
    cond = np.array([0.777, 0.692, 0.569, 0.563, 0.583])
    cond_lo = np.array([0.596, 0.546, 0.518, 0.554, 0.532])
    cond_hi = np.array([0.957, 0.973, 0.958, 0.910, 0.836])
    uncond = np.array([0.575, 0.558, 0.552, 0.518, 0.491])
    uncond_lo = np.array([0.373, 0.378, 0.365, 0.374, 0.420])
    uncond_hi = np.array([0.736, 0.703, 0.711, 0.731, 0.665])

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(0.5, color="0.35", lw=1.0, ls=":", zorder=1, label="chance (0.5)")
    ax.fill_between(snr, cond_lo, cond_hi, color=INTERMEDIATE, alpha=0.13, lw=0,
                    label="conditional, across-seed range")
    ax.fill_between(snr, uncond_lo, uncond_hi, color=CLASSICAL, alpha=0.13, lw=0,
                    label="unconditional, across-seed range")
    ax.plot(snr, cond, "o-", color=INTERMEDIATE, label="conditional flow (median)")
    ax.plot(snr, uncond, "s--", color=CLASSICAL, label="unconditional ablation (median)")
    for x, c, u in zip(snr, cond, uncond):
        ax.annotate(f"+{c - u:.2f}", (x, max(c, u) + 0.015), fontsize=7.5,
                    ha="center", color=INTERMEDIATE)
    ax.set_xlabel("latent signal-to-noise ratio (dB)")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(snr)
    ax.set_ylim(0.33, 1.0)
    ax.set_title("Conditioning lifts the median detection curve at every\n"
                 "SNR rung (largest at low SNR), but the seed bands overlap",
                 fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=7.0)
    fig.tight_layout()
    save(fig, "fig22_latent_snr_lift")


# ─────────────────────────────────────────────────────────────────────────
# 23 — specificity audit (tab:res_rq2_spec)
# ─────────────────────────────────────────────────────────────────────────
def fig23_specificity_audit() -> None:
    # Five-seed medians (tab:res_rq2_spec); neither = 1 - (a + v + both).
    cohorts = ["healthy\nhold-out", "D2 anomaly", "D3 anomaly", "D4 anomaly"]
    a_only = np.array([0.122, 0.050, 0.846, 0.144])
    v_only = np.array([0.025, 0.089, 0.000, 0.311])
    both = np.array([0.003, 0.007, 0.045, 0.200])
    neither = np.clip(1.0 - (a_only + v_only + both), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(cohorts))
    b0 = np.zeros(len(cohorts))
    for vals, color, label in [
        (both, LATE_FUSION, "both fire (AND alert)"),
        (a_only, ACOUSTIC, "acoustic only"),
        (v_only, VIBRATION, "vibration only"),
        (neither, "0.85", "neither"),
    ]:
        ax.bar(x, vals, 0.6, bottom=b0, color=color, label=label)
        b0 += vals
    for xi, b in zip(x, both):
        if b > 0.08:  # label inside the orange segment
            ax.annotate(f"{b:.3f}", (xi, b / 2), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        else:  # zero share: label inside the (grey) bar with a leader line
            ax.annotate(f"both: {b:.3f}", (xi, 0.0), xytext=(xi, 0.14),
                        fontsize=7.5, fontweight="bold", color=LATE_FUSION,
                        ha="center", va="bottom",
                        arrowprops={"arrowstyle": "-", "color": LATE_FUSION, "lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("fraction of cohort windows")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Per-cohort co-firing (five-seed medians): the modalities almost never\n"
        "co-fire on healthy windows (0.003), yet co-fire on every anomaly cohort"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    save(fig, "fig23_specificity_audit")


# ─────────────────────────────────────────────────────────────────────────
# 26 — LORO paradigm comparison (tab:res_rq3_loro)
# ─────────────────────────────────────────────────────────────────────────
def fig26_loro_paradigms() -> None:
    # Five-seed median macro MAE with across-seed [min, max] whiskers
    # (tab:res_rq3_loro), ordered best-first.  The earlier "intermediate fusion
    # is 4x tighter" stability claim does not survive the multi-seed reruns:
    # the seed ranges overlap, so no paradigm is clearly tighter.
    rows = [
        # (label, median, min, max, colour, is_best)
        ("Late fusion: confidence-gated", 0.181, 0.160, 0.218, LATE_FUSION, True),
        ("Intermediate: V4-fusion", 0.189, 0.163, 0.278, INTERMEDIATE, False),
        ("Late fusion: uniform avg", 0.194, 0.163, 0.215, LATE_FUSION, False),
        ("Unimodal: V4-acoustic", 0.198, 0.158, 0.286, ACOUSTIC, False),
        ("Unimodal: V4-vibration", 0.231, 0.211, 0.334, VIBRATION, False),
        ("Late fusion: weighted avg", 0.252, 0.246, 0.254, LATE_FUSION, False),
    ]
    labels = [r[0] for r in rows]
    mae = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    colors = [r[4] for r in rows]
    xerr = np.vstack([mae - lo, hi - mae])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.barh(y, mae, 0.6, color=colors, alpha=0.9, xerr=xerr, capsize=3,
            error_kw={"elinewidth": 1.1, "ecolor": "0.35"})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.36)
    ax.set_xlabel("LORO macro MAE (m), median with across-seed [min, max]")
    for yi, (m, h, best) in enumerate(zip(mae, hi, [r[5] for r in rows])):
        ax.annotate(f"{m:.3f}", (h + 0.006, yi), va="center", fontsize=7.5,
                    color="0.2", fontweight="bold" if best else "normal")
    ax.annotate("fitted in sample:\nworst out of sample", (0.255, 5), va="center",
                ha="left", fontsize=7.0, color=ANOMALY, xytext=(0.30, 5),
                textcoords="data")
    fig.suptitle(
        "Leave-one-recording-out: late fusion (confidence-gated) wins on median\n"
        "accuracy; the seed ranges overlap, so no paradigm is clearly tighter",
        fontsize=9.5, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig26_loro_paradigm_comparison")


# ─────────────────────────────────────────────────────────────────────────
# 27 — LOPO per-fold distribution (lopo/folds.jsonl)
# ─────────────────────────────────────────────────────────────────────────
def fig27_lopo_per_fold() -> None:
    by_mode: dict[str, list[float]] = {}
    with open(LOPO_RUN_DIR / "lopo" / "folds.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by_mode.setdefault(r["channel_mode"], []).append(r["val_mae_3d_m"])

    order = ["tdoa_only", "both", "srp_only", "vibration_only_learned"]
    data = [by_mode[m] for m in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bp = ax.boxplot(
        data, vert=True, patch_artist=True, widths=0.5, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "0.2", "markersize": 5},
        medianprops={"color": "0.2"},
    )
    rng = np.random.default_rng(0)
    top = max(max(v) for v in data)
    ax.set_ylim(0, top * 1.18)
    for i, (mode, vals) in enumerate(zip(order, data)):
        bp["boxes"][i].set_facecolor(CHANNEL_MODE_COLORS[mode])
        bp["boxes"][i].set_alpha(0.55)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.plot(np.full(len(vals), i + 1) + jitter, vals, "o",
                color=CHANNEL_MODE_COLORS[mode], ms=3.5, mec="0.25", mew=0.4, zorder=3)
        ax.annotate(f"mean {np.mean(vals):.3f}", (i + 1, top * 1.10),
                    ha="center", va="bottom", fontsize=7.5, color="0.25")
    ax.set_xticklabels([CHANNEL_MODE_LABELS[m] for m in order],
                       fontsize=7.5, rotation=12, ha="center")
    ax.set_ylabel("held-out-position MAE (m)")
    ax.set_title(
        "Leave-one-position-out, 16 folds per channel mode:\n"
        "the accelerometer-TDOA pathway alone generalizes best to unseen positions"
    )
    fig.tight_layout()
    save(fig, "fig27_lopo_per_fold")


# ─────────────────────────────────────────────────────────────────────────
# 28 — cross-session transfer to the unseen D5 session
# ─────────────────────────────────────────────────────────────────────────
def fig28_cross_session_d5() -> None:
    # Five-seed medians of the cross-session table (tab:res_rq3_cross):
    # (mode, val MAE median, across-seed min, across-seed max, val p95 median)
    rows = [
        ("tdoa_only", 0.113, 0.107, 0.147, 0.259),
        ("both", 0.145, 0.117, 0.173, 0.282),
        ("srp_only", 0.151, 0.140, 0.176, 0.320),
        ("vibration_only_learned", 0.291, 0.245, 0.304, 0.441),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    for i, (mode, mae, lo, hi, p95) in enumerate(rows):
        c = CHANNEL_MODE_COLORS[mode]
        ax.errorbar(mae, i, xerr=[[mae - lo], [hi - mae]],
                    fmt="o", color=c, ms=7, capsize=4, elinewidth=1.4, zorder=3)
        ax.plot(p95, i, marker="|", ms=11, color=c, mew=2, zorder=3)
        ax.annotate(f"{mae:.3f} m", (mae, i - 0.22), ha="center", va="bottom",
                    fontsize=8, color=c, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([CHANNEL_MODE_LABELS[m] for m, *_ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.75)
    ax.set_xlabel("MAE on the 63 unseen D5 knock windows (m)")
    ax.set_title(
        "Cross-session transfer (train D2/D3/D4 $\\rightarrow$ test D5), five-seed:\n"
        "dot = median MAE with across-seed range, tick = median 95th percentile"
    )
    fig.tight_layout()
    save(fig, "fig28_cross_session_d5")


# ─────────────────────────────────────────────────────────────────────────
# 31 — SCADA mutual information with operating mode (tab:res_rq4_mode)
# ─────────────────────────────────────────────────────────────────────────
def fig31_scada_mode_mi() -> None:
    rows = [
        ("active power (1_P_Ist)", "electrical", 0.96),
        ("generator voltage (1_21kV Gen. Spg.)", "electrical", 0.93),
        ("speed (1_Drehzahl_Ist)", "rotational", 0.93),
        ("valve position (1_KS Stellung)", "other", 0.91),
        ("excitation current (1_Erregerstrom)", "electrical", 0.90),
        ("guide vane (1_Leitapparat Stell.)", "hydraulic", 0.90),
        ("runner pressure (1_Laufraddruck)", "pressure", 0.86),
        ("spiral-case pressure (1_Spiraldruck)", "pressure", 0.81),
        ("flow (1_Q_Ist)", "hydraulic", 0.70),
    ]
    fam_colors = {
        "electrical": "#c44e52",
        "rotational": "#dd8452",
        "hydraulic": "#4c72b0",
        "pressure": "#55a868",
        "other": "#8c8c8c",
    }
    labels = [r[0] for r in rows][::-1]
    fams = [r[1] for r in rows][::-1]
    mi = np.array([r[2] for r in rows])[::-1]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.barh(np.arange(len(rows)), mi, color=[fam_colors[f] for f in fams], height=0.62)
    for i, v in enumerate(mi):
        ax.annotate(f"{v:.2f}", (v + 0.012, i), va="center", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mutual information with operating mode (nats)")
    ax.set_xlim(0, 1.06)
    ax.set_title(
        "ROW II SCADA channels identify the operating mode strongly\n"
        "(~two orders of magnitude above their per-anomaly information)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
    ax.legend(handles, fam_colors.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig31_scada_mode_mi")


# ─────────────────────────────────────────────────────────────────────────
# 33 — robustness panels: regularization-grid F1 heatmap + five-seed strip
# ─────────────────────────────────────────────────────────────────────────
def fig33_robustness() -> None:
    cells = json.loads(FINALIZE.read_text())["reg_grid"]
    pat = re.compile(r"^(2026052[67]|202605\d{2})_\d{6}__v3deep_v3_(d(\d)_w(\d)|cap_(\w+))_s(\d+)$")

    grid: dict[tuple[int, int], float] = {}
    for c in cells:
        m = pat.match(c["cell"])
        if not m or not c["cell"].startswith(("20260526", "20260527")):
            continue  # final-protocol block only (earlier block used another protocol)
        seed = int(m.group(6))
        if m.group(3) is not None and seed == 42:
            grid[(int(m.group(3)), int(m.group(4)))] = c["f1"]

    # Panel (b): the five canonical seeds the Results tables report, not the May
    # reg-grid seeds.  This is the event-level F1 that couples to the K-means
    # assignment (median 0.989, with the seed-99 low tail at 0.756).
    seeds = {int(s): v["rq2_real_f1"]
             for s, v in json.loads(MULTISEED.read_text())["per_seed"].items()}

    dropout_levels = [0.0, 0.1, 0.2, 0.3]
    wd_levels = {4: "1e-4", 5: "5e-4", 3: "1e-3"}
    wd_order = [4, 5, 3]
    M = np.array([[grid[(di, wi)] for wi in wd_order] for di in range(4)])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    im = ax1.imshow(M, cmap="viridis", vmin=0.90, vmax=0.97, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax1.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.945 else "black", fontsize=8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([wd_levels[w] for w in wd_order])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"{d:g}" for d in dropout_levels])
    ax1.set_xlabel("weight decay")
    ax1.set_ylabel("dropout")
    ax1.set_title("(a) Regularization grid, real-anomaly F1\n(seed 42; precision $\\geq 0.994$ everywhere)")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, shrink=0.85)

    seed_order = sorted(seeds)
    vals = np.array([seeds[s] for s in seed_order])
    med = float(np.median(vals))
    rng = np.random.default_rng(1)
    xj = rng.uniform(-0.05, 0.05, len(vals))
    ax2.plot(np.zeros(len(vals)) + xj, vals, "o", ms=7, color=INTERMEDIATE,
             mec="0.2", mew=0.5, zorder=3)
    ax2.axhline(med, color="0.35", lw=1.0, ls="--")
    ax2.annotate(f"median {med:.3f} (robust)", (0.50, med + 0.004), fontsize=7.5,
                 ha="right", va="bottom", color="0.25")
    lo = seed_order[int(np.argmin(vals))]
    ax2.annotate(f"seed {lo}: {vals.min():.3f}\n(degenerate\nK-means init)",
                 (xj[int(np.argmin(vals))], vals.min()),
                 xytext=(0.18, vals.min() + 0.020), fontsize=7.5, color=ANOMALY,
                 va="center",
                 arrowprops={"arrowstyle": "-", "color": ANOMALY, "lw": 0.8})
    ax2.set_xlim(-0.3, 0.55)
    ax2.set_ylim(vals.min() - 0.03, vals.max() + 0.035)
    ax2.set_xticks([])
    ax2.set_ylabel("real-anomaly F1")
    ax2.set_title("(b) Five seeds, selected cell")
    fig.tight_layout()
    save(fig, "fig33_robustness_panels")


def main() -> None:
    style.apply_style()
    fig19_acoustic_collapse()
    fig20_threshold_transfer()
    fig22_latent_snr_lift()
    fig23_specificity_audit()
    fig26_loro_paradigms()
    fig27_lopo_per_fold()
    fig28_cross_session_d5()
    fig31_scada_mode_mi()
    fig33_robustness()


if __name__ == "__main__":
    main()
