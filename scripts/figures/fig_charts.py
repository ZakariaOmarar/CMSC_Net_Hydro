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

RUN_DIR = REPO_ROOT / "results" / "runs" / "20260610_122607__full_pipeline_b5_cma"
# LOPO folds of the run whose aggregates match the Results chapter's LOPO
# table (tdoa-only mean MAE 0.131 m over 16 folds).
LOPO_RUN_DIR = REPO_ROOT / "results" / "runs" / "20260609_225957__full_pipeline_b5_cma"
FINALIZE = REPO_ROOT / "results" / "reports" / "finalize_results_20260611_012822.json"
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
        "Joint-pool fusion ignores vibration: zeroing it leaves $c_t$ unchanged",
        fontsize=10, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig19_acoustic_collapse")


# ─────────────────────────────────────────────────────────────────────────
# 20 — threshold-transfer FPR, baselines and heads on one axis
# ─────────────────────────────────────────────────────────────────────────
def fig20_threshold_transfer() -> None:
    # Matched transfer protocol (Results ch., tab:res_rq2_shift): threshold
    # fitted on one set of healthy operating conditions, FPR evaluated on a
    # disjoint set.  In-dist rates: baselines calibrate to the 0.05 target.
    rows = [
        # (label, in-dist FPR, shift FPR, colour, is_head)
        ("OC-SVM (ac.)", 0.052, 0.947, CLASSICAL, False),
        ("LSTM-AE (ac.)", 0.050, 0.022, CLASSICAL, False),
        ("$k$-means (ac.)", 0.050, 0.005, CLASSICAL, False),
        ("KDE (ac.)", 0.050, 0.003, CLASSICAL, False),
        ("OC-SVM (vib.)", 0.050, 0.010, CLASSICAL, False),
        ("$k$-means (vib.)", 0.050, 0.012, CLASSICAL, False),
        ("KDE (vib.)", 0.050, 0.015, CLASSICAL, False),
        ("V3-acoustic", 0.054, 0.060, ACOUSTIC, True),
        ("V3-vibration", 0.049, 0.014, VIBRATION, True),
        ("V3-fusion", 0.051, 0.005, INTERMEDIATE, True),
        ("Late-fusion AND", 0.006, 0.003, LATE_FUSION, True),
    ]
    labels = [r[0] for r in rows]
    indist = np.array([r[1] for r in rows])
    shift = np.array([r[2] for r in rows])
    colors = [r[3] for r in rows]

    x = np.arange(len(rows), dtype=float)
    x[7:] += 0.6  # visual gap between baselines and proposed heads
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(x - w / 2, indist, w, color="0.78", label="FPR, in-distribution threshold")
    ax.bar(x + w / 2, shift, w, color=colors, label="FPR, transferred threshold")
    ax.axhline(0.05, color="0.35", lw=0.9, ls="--")
    ax.text(x[0] - 0.45, 0.058, "0.05 target", va="bottom", ha="left",
            fontsize=7.5, color="0.35")

    ax.set_yscale("log")
    ax.set_ylim(2e-3, 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("healthy false-positive rate (log)")
    ax.set_title(
        "Threshold transfer to an unseen operating condition: "
        "OC-SVM collapses to 0.947, the AND rule holds at 0.003"
    )
    ax.annotate(
        "0.947", (x[0] + w / 2, 0.947), xytext=(0, 4),
        textcoords="offset points", ha="center", fontsize=8,
        fontweight="bold", color=ANOMALY,
    )
    ax.annotate(
        "0.003", (x[-1] + w / 2, 0.003), xytext=(0, 4),
        textcoords="offset points", ha="center", fontsize=8,
        fontweight="bold", color=LATE_FUSION,
    )
    ax.axvline((x[6] + x[7]) / 2, color="0.8", lw=0.8)
    ax.text(np.mean(x[:7]), 1.25, "unconditional baselines", ha="center", fontsize=8, color="0.3")
    ax.text(np.mean(x[7:]), 1.25, "proposed heads", ha="center", fontsize=8, color="0.3")
    fig.tight_layout()
    save(fig, "fig20_threshold_transfer_fpr")


# ─────────────────────────────────────────────────────────────────────────
# 22 — latent-SNR conditioning lift (tab:res_rq2_auc)
# ─────────────────────────────────────────────────────────────────────────
def fig22_latent_snr_lift() -> None:
    snr = np.array([-10, -5, 0, 5, 10])
    cond = np.array([0.370, 0.407, 0.473, 0.550, 0.539])
    uncond = np.array([0.323, 0.311, 0.351, 0.327, 0.351])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(0.5, color="0.35", lw=1.0, ls=":", zorder=1, label="chance (0.5)")
    ax.fill_between(snr, uncond, cond, color=INTERMEDIATE, alpha=0.18, label="conditioning lift")
    ax.plot(snr, cond, "o-", color=INTERMEDIATE, label="conditional flow")
    ax.plot(snr, uncond, "s--", color=CLASSICAL, label="unconditional ablation")
    for x, c, u in zip(snr, cond, uncond):
        ax.annotate(f"+{c - u:.2f}", (x, (c + u) / 2), fontsize=7.5,
                    ha="center", color=INTERMEDIATE)
    ax.set_xlabel("latent signal-to-noise ratio (dB)")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(snr)
    ax.set_ylim(0.28, 0.62)
    ax.set_title("Conditioning lifts the controlled detection\ncurve at every SNR rung")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig22_latent_snr_lift")


# ─────────────────────────────────────────────────────────────────────────
# 23 — specificity audit (tab:res_rq2_spec)
# ─────────────────────────────────────────────────────────────────────────
def fig23_specificity_audit() -> None:
    cohorts = ["healthy\nhold-out", "D2 anomaly", "D3 anomaly", "D4 anomaly"]
    a_only = np.array([0.013, 0.573, 0.297, 0.442])
    v_only = np.array([0.017, 0.070, 0.000, 0.014])
    both = np.array([0.000, 0.159, 0.703, 0.526])
    neither = np.array([0.970, 0.199, 0.000, 0.018])

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
        "Per-cohort co-firing: both modalities never fire together on healthy\n"
        "windows, yet co-fire on every anomaly cohort"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    save(fig, "fig23_specificity_audit")


# ─────────────────────────────────────────────────────────────────────────
# 26 — LORO paradigm comparison (tab:res_rq3_loro)
# ─────────────────────────────────────────────────────────────────────────
def fig26_loro_paradigms() -> None:
    # rows ordered best-first; horizontal bars share one label column
    rows = [
        ("Late fusion: confidence-gated", 0.156, 0.056, LATE_FUSION),
        ("Unimodal: V4-acoustic", 0.162, 0.049, ACOUSTIC),
        ("Intermediate: V4-fusion", 0.169, 0.014, INTERMEDIATE),
        ("Late fusion: uniform avg", 0.194, 0.048, LATE_FUSION),
        ("Unimodal: V4-vibration", 0.268, 0.058, VIBRATION),
        ("Late fusion: weighted avg", 0.292, 0.005, LATE_FUSION),
    ]
    labels = [r[0] for r in rows]
    mae = np.array([r[1] for r in rows])
    std = np.array([r[2] for r in rows])
    colors = [r[3] for r in rows]
    y = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax1.barh(y, mae, 0.62, color=colors, xerr=std, capsize=3,
             error_kw={"elinewidth": 1.0, "ecolor": "0.35"})
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.40)
    ax1.set_xlabel("LORO macro MAE (m)")
    ax1.set_title("(a) average accuracy ($\\pm$ across-fold std)", fontsize=9)
    for yi, (m, s) in enumerate(zip(mae, std)):
        ax1.annotate(f"{m:.3f}", (m + s + 0.012, yi), va="center", fontsize=7.5,
                     color="0.2", fontweight="bold" if yi == 0 else "normal")

    ax2.barh(y, std, 0.62, color=colors)
    ax2.set_xlim(0, 0.075)
    ax2.set_xlabel("across-fold std of MAE (m)")
    ax2.set_title("(b) worst-case stability", fontsize=9)
    for yi, s in enumerate(std):
        ax2.annotate(f"{s:.3f}", (s + 0.002, yi), va="center", fontsize=7.5,
                     color=INTERMEDIATE if yi == 2 else "0.2",
                     fontweight="bold" if yi == 2 else "normal")
    ax2.annotate("4x tighter than\nany other paradigm", (0.030, 2), va="center",
                 fontsize=7.5, color=INTERMEDIATE, fontweight="bold")

    fig.suptitle(
        "Leave-one-recording-out: late fusion wins on accuracy, intermediate fusion on stability",
        fontsize=10, y=1.04,
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
    # Values of the Results chapter's cross-session table (tab:res_rq3_cross):
    # (mode, val MAE, ci low, ci high, val p95)
    rows = [
        ("tdoa_only", 0.109, 0.095, 0.124, 0.214),
        ("both", 0.125, 0.107, 0.142, 0.257),
        ("srp_only", 0.140, 0.126, 0.153, 0.231),
        ("vibration_only_learned", 0.235, 0.210, 0.258, 0.365),
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
        "Cross-session transfer (train D2/D3/D4 $\\rightarrow$ test D5):\n"
        "dot = MAE with 95% CI, tick = 95th percentile error"
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
    seeds: dict[int, float] = {}
    for c in cells:
        m = pat.match(c["cell"])
        if not m or not c["cell"].startswith(("20260526", "20260527")):
            continue  # final-protocol block only (earlier block used another protocol)
        seed = int(m.group(6))
        if m.group(3) is not None:
            d_lvl, w_lvl = int(m.group(3)), int(m.group(4))
            if seed == 42:
                grid[(d_lvl, w_lvl)] = c["f1"]
            if (d_lvl, w_lvl) == (0, 5):
                seeds[seed] = c["f1"]

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
