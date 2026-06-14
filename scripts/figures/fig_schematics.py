"""Schematic / conceptual figures (no model, no signal data).

Renders (thesis figure plan numbering):
   1  domain-shift motivation timeline (conceptual)
   2  annotated-photo placeholder for the prototype
   4  campaign overview card strip (D1..D5, from tab:datasets/durations)
   5  preprocessing pipeline flowchart with tensor shapes
  17  split-protocol diagram (recording-level splits + 3 localization protocols)
  34  paradigm-map summary (demand -> winning paradigm)

Run with:  python -m scripts.figures.fig_schematics
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.figures import style
from scripts.figures.style import (
    ACOUSTIC,
    ANOMALY,
    CAMPAIGN_COLORS,
    INTERMEDIATE,
    LATE_FUSION,
    MODE_COLORS,
    VIBRATION,
    save,
)


# ── drawing helpers ─────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, *, fc="#f0f0f0", ec="0.35", fs=8, lw=1.0,
        text_color="0.1", weight="normal", rounding=0.06):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight=weight, zorder=3)
    return p


def arrow(ax, p0, p1, *, color="0.3", lw=1.2, style="-|>", shrink=2.0,
          connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
        shrinkA=shrink, shrinkB=shrink, connectionstyle=connectionstyle,
        linestyle=ls, zorder=1.5,
    )
    ax.add_patch(a)
    return a


def blank_axes(figsize, xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────
# 1 — domain-shift motivation
# ─────────────────────────────────────────────────────────────────────────
def fig01_domain_shift() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 100, 1200)
    tu_end, pu_start = 44, 52

    base = np.where(t < tu_end, 1.0, np.where(t > pu_start, 2.9, 0.0))
    ramp = (t >= tu_end) & (t <= pu_start)
    base[ramp] = 1.0 + (2.9 - 1.0) * (t[ramp] - tu_end) / (pu_start - tu_end)
    spike = 2.6 * np.exp(-0.5 * ((t - (tu_end + pu_start) / 2) / 1.9) ** 2)
    score = base + spike + 0.14 * rng.standard_normal(t.size)
    score = np.convolve(score, np.ones(5) / 5, mode="same")
    t, score = t[5:-5], score[5:-5]  # drop smoothing edge artefacts

    thr = 2.05

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axvspan(tu_end, pu_start, color="0.92", zorder=0)
    above = score > thr
    ax.plot(t, score, color=ACOUSTIC, lw=1.2, label="anomaly score of a static detector")
    ax.fill_between(t, thr, score, where=above, color=ANOMALY, alpha=0.45,
                    label="false alarms (machine healthy throughout)")
    ax.axhline(thr, color="0.2", lw=1.1, ls="--")
    ax.text(1.5, thr + 0.1, "static threshold (fit on TU)", fontsize=8, color="0.2")

    ax.text(tu_end / 2, 4.55, "Turbine (TU)", ha="center", fontsize=9,
            color=MODE_COLORS["Turbine"], fontweight="bold")
    ax.text((tu_end + pu_start) / 2, 4.55, "transition", ha="center", fontsize=8, color="0.35")
    ax.text((pu_start + 100) / 2, 4.55, "Pump (PU)", ha="center", fontsize=9,
            color=MODE_COLORS["Pump"], fontweight="bold")

    ax.annotate("synchronized\nfalse-alarm spike", xy=(46.5, 4.1), xytext=(28, 3.6),
                fontsize=8, color=ANOMALY, ha="center",
                arrowprops={"arrowstyle": "-|>", "color": ANOMALY, "lw": 1.0})
    ax.annotate("healthy PU baseline sits above\nthe TU-fit threshold: permanent alarm",
                xy=(83, 3.0), xytext=(60, 1.0), fontsize=8, color="0.15",
                arrowprops={"arrowstyle": "-|>", "color": "0.4", "lw": 1.0})

    ax.set_xlabel("time")
    ax.set_ylabel("anomaly score")
    ax.set_ylim(0, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    save(fig, "fig01_domain_shift_motivation")


# ─────────────────────────────────────────────────────────────────────────
# 2 — annotated photographs of the two rigs
# ─────────────────────────────────────────────────────────────────────────
def fig02_photo() -> None:
    from scripts.figures.style import REPO_ROOT

    img_rect = plt.imread(REPO_ROOT / "data" / "second_test_dataset" / "1000052773.jpg")
    img_circ = plt.imread(REPO_ROOT / "data" / "third_test_dataset" / "1000052877.jpg")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 4.3), gridspec_kw={"width_ratios": [1.25, 0.75]}
    )

    def marker(ax, x_frac, y_frac, num, img):
        h, w = img.shape[:2]
        x, y = x_frac * w, y_frac * h
        ax.add_patch(plt.Circle((x, y), 0.045 * w, fc="white", ec="0.1",
                                lw=1.4, alpha=0.95, zorder=5))
        ax.text(x, y, num, ha="center", va="center", fontsize=9,
                fontweight="bold", color="0.1", zorder=6)

    ax_a.imshow(img_rect)
    for xf, yf, n in [(0.13, 0.50, "1"), (0.67, 0.70, "2"), (0.14, 0.72, "3"),
                      (0.42, 0.43, "3"), (0.55, 0.08, "4")]:
        marker(ax_a, xf, yf, n, img_rect)
    ax_a.set_title("(a) rectangular rig (D1, D2)", fontsize=9)

    ax_b.imshow(img_circ)
    for xf, yf, n in [(0.45, 0.42, "5"), (0.10, 0.75, "3"), (0.68, 0.46, "3"),
                      (0.85, 0.18, "4")]:
        marker(ax_b, xf, yf, n, img_circ)
    ax_b.set_title("(b) circular 3D-printed rig (D3-D5)", fontsize=9)

    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    key = ("1  inlet valve      2  pump / drive      3  sensor breakouts on the casing "
           "(mics + accelerometers, taped labels)      4  acquisition boards (shared trigger)"
           "      5  circular casing (~10 cm)")
    fig.text(0.5, 0.015, key, ha="center", fontsize=7.4, color="0.2")
    fig.suptitle(
        "The two bench-top prototypes that produced the corpus",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save(fig, "fig02_prototype_photo", png_only=True)


# ─────────────────────────────────────────────────────────────────────────
# 4 — campaign overview cards
# ─────────────────────────────────────────────────────────────────────────
def fig04_campaign_overview() -> None:
    cards = [
        ("D1", "4 mics + 4 accel", "peak vib (~4 Hz)",
         "modes labeled:\nPU / ST / TU", "39 min healthy\n11 min anomaly", "0 knock pos."),
        ("D2", "5 mics + 5 accel", "peak vib (4 Hz)",
         "modes labeled:\nPU / ST / TU", "20 min healthy\n8 min anomaly", "3 knock pos.\n(single-mode)"),
        ("D3", "9 mics + 4 accel", "peak vib (16 Hz)",
         "speed{1,2,3}\n(fan-noise level)", "7 min healthy\n0.6 min anomaly", "1 knock pos.\n(midpoint)"),
        ("D4", "9 mics + 4 accel", "raw vib (~376 Hz)",
         "speed{1,2,3}\n(fan-noise level)", "33 min healthy\n77 min anomaly\n(sparse)", "6 knock pos."),
        ("D5", "9 mics + 4 accel", "raw vib (~446 Hz)",
         "flat healthy pool\n(no speed token)", "4 min healthy\n27 min anomaly", "6 knock pos."),
    ]
    fig, ax = blank_axes((7.4, 3.4), xlim=(0, 25.4), ylim=(0, 11))

    # timeline spine
    arrow(ax, (0.4, 0.7), (25.0, 0.7), color="0.4", lw=1.4)
    ax.text(24.9, 0.18, "campaign order", ha="right", fontsize=7.5, color="0.4")

    w, gap = 4.6, 0.4
    for i, (name, sensors, vib, labels, dur, spatial) in enumerate(cards):
        x = 0.4 + i * (w + gap)
        c = CAMPAIGN_COLORS[name]
        box(ax, x, 1.4, w, 9.0, "", fc="white", ec=c, lw=1.6, rounding=0.12)
        ax.add_patch(FancyBboxPatch((x, 9.35), w, 1.05,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=c, ec=c, zorder=2))
        ax.text(x + w / 2, 9.88, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=3)
        rows = [
            (sensors, "0.1", "bold"),
            (vib, VIBRATION if "raw" in vib else "0.35", "normal"),
            (labels, MODE_COLORS["Turbine"] if "modes" in labels else "0.35", "normal"),
            (dur, "0.1", "normal"),
            (spatial, ANOMALY if "knock" in spatial and not spatial.startswith("0") else "0.45", "normal"),
        ]
        y = 8.55
        for txt, color, weight in rows:
            ax.text(x + w / 2, y, txt, ha="center", va="center", fontsize=7.3,
                    color=color, fontweight=weight, zorder=3)
            y -= 1.55
        ax.plot([x + w / 2], [0.7], "o", color=c, ms=7, zorder=3)

    # group braces
    ax.text(0.4 + (2 * w + gap) / 2, 0.18 + 10.9 - 10.9, "", fontsize=1)  # noop spacing
    for (i0, i1, label) in [(0, 1, "mode detection"), (2, 3, "fault detection + localization"),
                            (4, 4, "localization")]:
        x0 = 0.4 + i0 * (w + gap)
        x1 = 0.4 + i1 * (w + gap) + w
        ax.annotate("", xy=(x0, 10.78), xytext=(x1, 10.78),
                    arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 1.0})
        ax.text((x0 + x1) / 2, 10.92, label, ha="center", fontsize=7.3, color="0.35")
    save(fig, "fig04_campaign_overview")


# ─────────────────────────────────────────────────────────────────────────
# 5 — preprocessing pipeline flowchart
# ─────────────────────────────────────────────────────────────────────────
def fig05_preprocessing() -> None:
    fig, ax = blank_axes((7.8, 3.9), xlim=(0, 28), ylim=(0, 13))

    box(ax, 0.2, 5.0, 3.7, 3.0, "recording\ndirectory\n(WAVs + CSVs)", fc="#f7f7f7", fs=7.0)

    box(ax, 5.0, 8.1, 4.8, 3.0, "audio load\n16 kHz, stack,\nstrict channel check", fc="#e8f0fa",
        ec=ACOUSTIC, fs=7.0)
    box(ax, 5.0, 1.9, 4.8, 3.0, "vibration parse\npeak (D1-D3) /\nraw DMA (D4-D5)", fc="#e9f6e9",
        ec=VIBRATION, fs=7.0)

    box(ax, 11.0, 5.0, 4.5, 3.0, "cross-modal\nsync check\n(4 gates, $\\pm$0.5 s)", fc="#fdf3e3",
        ec="0.4", fs=7.0)

    box(ax, 17.0, 8.1, 5.4, 3.0, "acoustic features\nlog-mel 96 + CWT 64\n($n_{fft}$ 4096, hop 2048)",
        fc="#e8f0fa", ec=ACOUSTIC, fs=7.0)
    box(ax, 17.0, 1.9, 5.4, 3.0, "vibration features\namplitude + envelope\n+ impulsiveness",
        fc="#e9f6e9", ec=VIBRATION, fs=7.0)

    box(ax, 24.0, 5.0, 3.9, 3.0, "windowing\n50% overlap,\noctave scales,\nshape-keyed batches",
        fc="#f0eaf7", ec=INTERMEDIATE, fs=6.8)

    # arrows with shape annotations
    arrow(ax, (3.9, 7.0), (5.0, 9.4), connectionstyle="arc3,rad=0.25")
    arrow(ax, (3.9, 6.0), (5.0, 3.6), connectionstyle="arc3,rad=-0.25")
    arrow(ax, (9.8, 9.4), (11.6, 8.0), connectionstyle="arc3,rad=0.2")
    arrow(ax, (9.8, 3.6), (11.6, 5.0), connectionstyle="arc3,rad=-0.2")
    ax.text(10.4, 11.7, r"$(n_{mic}, T)$ @ 16 kHz", fontsize=7.0, color=ACOUSTIC, ha="center")
    ax.text(10.4, 0.9, r"$(n_{vib}, T_{vib})$ @ 4-446 Hz", fontsize=7.0, color=VIBRATION, ha="center")

    arrow(ax, (15.5, 8.0), (17.0, 9.4), connectionstyle="arc3,rad=0.2")
    arrow(ax, (15.5, 5.0), (17.0, 3.6), connectionstyle="arc3,rad=-0.2")
    arrow(ax, (22.4, 9.4), (24.4, 8.0), connectionstyle="arc3,rad=0.2")
    arrow(ax, (22.4, 3.6), (24.4, 5.0), connectionstyle="arc3,rad=-0.2")
    ax.text(23.4, 11.7, r"$(n_{mic}, 2, 96, T_{ac})$ @ 7.81 Hz", fontsize=7.0,
            color=ACOUSTIC, ha="center")
    ax.text(23.4, 0.9, r"$(n_{vib}, 3, T_{vib})$", fontsize=7.0, color=VIBRATION, ha="center")

    arrow(ax, (25.9, 5.0), (25.9, 3.6), color="0.4")
    ax.text(25.9, 2.4, "per-window tuple:\nfeatures + positions (m)\n+ campaign id + labels",
            ha="center", fontsize=6.8, color="0.3")
    save(fig, "fig05_preprocessing_pipeline")


# ─────────────────────────────────────────────────────────────────────────
# 17 — split protocol
# ─────────────────────────────────────────────────────────────────────────
def fig17_split_protocol() -> None:
    fig, ax = blank_axes((7.6, 4.8), xlim=(0, 34), ylim=(0, 20))
    LBL_X, ROW_X = 0.3, 8.6  # label column | diagram column

    ax.text(0.3, 19.2, "Recording-level splits (windows of one recording never cross a fold)",
            fontsize=9.5, fontweight="bold", color="0.15")

    # — anomaly-detection split: healthy pool —
    ax.text(0.3, 17.7, "RQ2 anomaly detection", fontsize=8.5, color="0.25", fontweight="bold")
    segs = [
        (ROW_X, 7.4, "#dbe9f6", ACOUSTIC, "train (healthy only)\nencoder + flow fit"),
        (ROW_X + 7.8, 6.2, "#fdf3e3", "#c08a2d", "threshold fit\n(held-out healthy)"),
        (ROW_X + 14.4, 7.6, "#e9f6e9", VIBRATION, "evaluation (disjoint\nhealthy + anomaly cohorts)"),
    ]
    for x, w, fc, ec, txt in segs:
        box(ax, x, 14.6, w, 2.4, txt, fc=fc, ec=ec, fs=7.0)
    ax.text(ROW_X + 14.2, 13.55, "fit and evaluation sets are disjoint by construction",
            fontsize=6.8, color=ANOMALY, ha="center", va="top")
    ax.text(LBL_X, 15.8, "threshold transfer: fit on\none operating condition,\nevaluate on a disjoint one",
            fontsize=7.0, color="0.35", va="center")

    # — localization protocols —
    ax.text(0.3, 12.4, "RQ3 localization: three protocols, increasing strictness",
            fontsize=8.5, color="0.25", fontweight="bold")

    def cells(y, n, held, label, cw=1.05, x0=ROW_X):
        for i in range(n):
            fc = "#f2f2f2" if i not in held else "#fbdcdc"
            ec = "0.55" if i not in held else ANOMALY
            box(ax, x0 + i * (cw + 0.12), y, cw, 1.3, "", fc=fc, ec=ec, fs=6, rounding=0.04)
        ax.text(LBL_X, y + 0.65, label, fontsize=7.2, color="0.2", va="center")
        return x0 + n * (cw + 0.12)

    xe = cells(10.4, 10, {3}, "LORO: hold out one\nrecording per fold")
    ax.text(xe + 0.5, 11.05, "same position may still\nappear on both sides",
            fontsize=6.8, color="0.35", va="center")

    xe = cells(8.0, 16, {6}, "LOPO: hold out one of\nthe 16 labelled positions", cw=0.78)
    ax.text(xe + 0.5, 8.65, "head retrained\nfrom scratch per fold",
            fontsize=6.8, color="0.35", va="center")

    # cross-session
    y = 5.4
    for i, (name, c) in enumerate([("D2", CAMPAIGN_COLORS["D2"]), ("D3", CAMPAIGN_COLORS["D3"]),
                                   ("D4", CAMPAIGN_COLORS["D4"])]):
        box(ax, ROW_X + i * 2.3, y, 2.1, 1.3, name, fc="white", ec=c, fs=8)
    box(ax, ROW_X + 8.6, y, 2.1, 1.3, "D5", fc="#fbdcdc", ec=ANOMALY, fs=8)
    arrow(ax, (ROW_X + 7.0, y + 0.65), (ROW_X + 8.5, y + 0.65), color="0.3")
    ax.text(LBL_X, y + 0.65, "cross-session: train on\nD2/D3/D4, test on unseen D5",
            fontsize=7.2, color="0.2", va="center")
    ax.text(ROW_X + 11.2, y + 0.65, "test session never seen at training time (strongest claim)",
            fontsize=6.8, color="0.35", va="center")

    # label discipline footer
    box(ax, 0.3, 0.6, 33.2, 3.2,
        "label discipline:  self-supervised stages and thresholds see healthy data only;  "
        "mode labels enter at evaluation only;\nspatial labels supervise and score the localization "
        "head only;  anomaly windows reach V4 only through the V3 alert gate",
        fc="#f7f7f7", ec="0.5", fs=6.6)
    save(fig, "fig17_split_protocol")


# ─────────────────────────────────────────────────────────────────────────
# 34 — paradigm map
# ─────────────────────────────────────────────────────────────────────────
def fig34_paradigm_map() -> None:
    fig, ax = blank_axes((6.8, 3.8), xlim=(0, 28), ylim=(0, 16))

    demands = [
        ("no false alarms on healthy data", "healthy FPR 0.000, shift FPR 0.003"),
        ("average localization accuracy", "LORO macro MAE 0.156 m"),
        ("worst-case stability", "across-fold std 0.014 m"),
        ("transfer to unseen positions / sessions", "LOPO 0.131 m; D5 0.109 m"),
        ("label-free mode discovery", "acoustic-trunk result (RQ1)"),
    ]
    winners = [
        ("Late fusion: AND rule", LATE_FUSION),
        ("Late fusion: confidence gate", LATE_FUSION),
        ("Intermediate fusion (FiLM on $c_t$)", INTERMEDIATE),
        ("Unimodal: accelerometer TDOA", VIBRATION),
        ("Unimodal: acoustic encoder", ACOUSTIC),
    ]
    y = 12.9
    for (d, v), (wname, wc) in zip(demands, winners):
        box(ax, 0.4, y, 11.8, 2.2, f"{d}\n{v}", fc="#f7f7f7", ec="0.45", fs=7.2)
        box(ax, 16.4, y, 11.0, 2.2, wname, fc=wc, ec=wc, fs=7.5, text_color="white",
            weight="bold")
        arrow(ax, (12.2, y + 1.1), (16.4, y + 1.1), color=wc, lw=1.6)
        y -= 2.85

    ax.text(6.3, 15.6, "operational demand (measured)", fontsize=9, ha="center",
            fontweight="bold", color="0.2")
    ax.text(21.9, 15.6, "winning fusion paradigm", fontsize=9, ha="center",
            fontweight="bold", color="0.2")
    ax.text(14.0, 0.1,
            "No single paradigm wins everywhere: the verdict is demand-specific.",
            fontsize=8, ha="center", color="0.25", style="italic")
    save(fig, "fig34_paradigm_map")


def main() -> None:
    style.apply_style()
    fig01_domain_shift()
    fig02_photo()
    fig04_campaign_overview()
    fig05_preprocessing()
    fig17_split_protocol()
    fig34_paradigm_map()


if __name__ == "__main__":
    main()
