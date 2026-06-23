"""Side-by-side comparison: full_run at hop=43 vs the hop=512 baseline.

Reads ``results/full_run_hop512_baseline/metrics.json`` and
``results/full_run_hop43/metrics.json`` and prints a Markdown-friendly table of
every comparable headline metric, grouped by stage. Highlights regressions and
improvements with a +/- delta.

The comparison itself is data: ``SECTIONS`` below declares, per stage, which
metric keys to show and how to format them. ``main()`` just loads the two runs
and renders that table — to add or drop a metric, edit ``SECTIONS``.

Usage::

    python scripts/hop_length_study/compare_hop43_vs_baseline.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results" / "full_run_hop512_baseline" / "metrics.json"
NEW  = REPO / "results" / "full_run_hop43"           / "metrics.json"


# ---------------------------------------------------------------------------
# Comparison specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One metric row: where to read it, how to label and format it."""

    label: str
    path: tuple[str, ...]                 # key path under metrics["stages"]
    prec: int = 3
    skip_if_both_none: bool = True        # drop the row when neither run has it
    note_path: tuple[str, ...] | None = None  # optional "n_b=…, n_n=…" sample sizes


@dataclass(frozen=True)
class Section:
    """A titled group of rows printed under one Markdown sub-header."""

    title: str
    rows: tuple[Row, ...]


def _metric_rows(
    stage: str,
    keys: tuple[str, ...],
    *,
    prec: int = 3,
    skip_if_both_none: bool = True,
) -> tuple[Row, ...]:
    """Rows for ``keys`` read from a single ``stage``, labelled by key name."""
    return tuple(
        Row(k, (stage, k), prec=prec, skip_if_both_none=skip_if_both_none)
        for k in keys
    )


_CLUSTER_KEYS = ("sanity_nmi", "sanity_ari", "sanity_purity", "train_loss_final", "val_loss_final")
_RQ1_KEYS = ("rq1_nmi", "rq1_ari", "rq1_purity")
_MAE_CI_KEYS = ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high")

SECTIONS: tuple[Section, ...] = (
    Section(
        "V0 baselines (hop-independent - sanity check)",
        (
            *(Row(f"V0 LGBM mode {k} macro-F1", ("v0", f"v0_lgbm_{k}", "val_macro_f1"))
              for k in ("d1", "d2")),
            *(Row(f"V0 LSTM-AE {k} val MSE", ("v0", f"v0_lstm_ae_{k}", "val_recon_mse"), prec=4)
              for k in ("d1", "d2", "d3", "d4")),
            *(Row(f"V0 SRP-PHAT {k} mean MAE (m)", ("v0", f"v0_srp_phat_{k}", "mean_mae_m"))
              for k in ("d2", "d3", "d4")),
        ),
    ),
    Section(
        "V1 acoustic SSL (sanity gate = mode clustering)",
        _metric_rows("v1_acoustic", _CLUSTER_KEYS, skip_if_both_none=False),
    ),
    Section(
        "V1 vibration SSL (note: should be unchanged; any delta is RNG side-effect)",
        _metric_rows("v1_vibration", _CLUSTER_KEYS, skip_if_both_none=False),
    ),
    Section(
        "V2 fusion (RQ1 mode-clustering on fused tokens)",
        _metric_rows(
            "v2",
            _RQ1_KEYS + ("train_loss_final", "val_loss_final", "train_simclr_final", "train_lmm_final"),
            skip_if_both_none=False,
        ),
    ),
    Section(
        "V2-A1 ablation (drop vibration during training)",
        _metric_rows("v2_a1_drop_vibration", _RQ1_KEYS, skip_if_both_none=False),
    ),
    Section(
        "V2 modality probe (eval-time modality zero-out)",
        tuple(
            Row(f"{path_name}.{k}", ("v2_modality_probe", path_name, k), skip_if_both_none=False)
            for path_name in ("both", "acoustic_only", "vibration_only")
            for k in ("nmi", "ari", "purity")
        ),
    ),
    Section(
        "V3 conditional CNF training",
        _metric_rows("v3", ("train_nll_final", "val_nll_final"), prec=2, skip_if_both_none=False),
    ),
    Section(
        "V3 transition false-alert rate (synthetic transition stress-test)",
        _metric_rows(
            "v3_rq2_transition_fpr",
            ("d1_pump_to_turbine", "d1_turbine_to_pump", "d1_to_d2_pump", "d1_to_d2_turbine"),
            prec=4,
            skip_if_both_none=False,
        ),
    ),
    Section(
        "V3 vs A2 (conditional vs unconditional) paired bootstrap on val NLL",
        _metric_rows(
            "v3_vs_a2_paired_test",
            ("delta_point", "delta_ci95_low", "delta_ci95_high", "p_value_two_sided", "direction"),
            prec=4,
            skip_if_both_none=False,
        ),
    ),
    Section(
        "V3 per-cohort alert rate (THE anomaly-detection headline)",
        tuple(
            Row(
                f"{cohort} alert_rate",
                ("v3_cohort_validation", cohort, "alert_rate"),
                skip_if_both_none=False,
                note_path=("v3_cohort_validation", cohort, "n"),
            )
            for cohort in ("healthy_holdout", "d2_random_fault", "d3_hit", "d4_random_fault")
        ),
    ),
    Section(
        "V4 localization head (3-D MAE; lower is better)",
        _metric_rows(
            "v4",
            _MAE_CI_KEYS + ("val_p95_3d", "val_init_mae_3d", "val_residual_norm_mean",
                            "train_loss_final", "val_loss_final"),
            prec=4,
        ),
    ),
    Section(
        "V4-A3 unconditional ablation",
        _metric_rows("v4_a3_unconditional", _MAE_CI_KEYS + ("val_p95_3d",), prec=4),
    ),
    Section(
        "V4 vs A3 paired bootstrap on per-window 3-D error",
        _metric_rows(
            "v4_vs_a3_paired_test",
            ("delta_mae_m", "delta_mae_ci95_low_m", "delta_mae_ci95_high_m",
             "p_value_two_sided", "direction"),
            prec=4,
            skip_if_both_none=False,
        ),
    ),
    Section("V4-A5 SRP-only ablation", _metric_rows("v4_a5_srp_only", _MAE_CI_KEYS, prec=4)),
    Section("V4-A5 TDOA-only ablation", _metric_rows("v4_a5_tdoa_only", _MAE_CI_KEYS, prec=4)),
    Section("V5.1 fan-noise conditioning", _metric_rows("v5_1", _MAE_CI_KEYS, prec=4)),
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _fmt(v, prec=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _delta(b, n, prec=3):
    if b is None or n is None:
        return "—"
    try:
        d = float(n) - float(b)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{prec}f}"
    except Exception:
        return "—"


def row(label, base_val, new_val, prec=3, note=""):
    print(f"| {label:<38} | {_fmt(base_val, prec):>10} | {_fmt(new_val, prec):>10} | {_delta(base_val, new_val, prec):>9} | {note}")


def section(title):
    print(f"\n### {title}")
    print("| metric                                 | hop=512    | hop=43     |     delta     | note")
    print("|----------------------------------------|------------|------------|-----------|------")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_section(sec: Section, base: dict, new: dict) -> None:
    section(sec.title)
    for r in sec.rows:
        base_val = _get(base, "stages", *r.path)
        new_val = _get(new, "stages", *r.path)
        if r.skip_if_both_none and base_val is None and new_val is None:
            continue
        note = ""
        if r.note_path is not None:
            n_b = _get(base, "stages", *r.note_path)
            n_n = _get(new, "stages", *r.note_path)
            note = f"n_b={n_b}, n_n={n_n}"
        row(r.label, base_val, new_val, prec=r.prec, note=note)


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"Baseline missing: {BASE}")
    if not NEW.exists():
        raise SystemExit(f"hop=43 metrics missing: {NEW}")
    base = json.load(BASE.open())
    new = json.load(NEW.open())

    print("# hop=43 vs hop=512 baseline — full pipeline comparison\n")
    print(f"Baseline: {BASE}")
    print(f"hop=43:   {NEW}\n")

    for sec in SECTIONS:
        _render_section(sec, base, new)

    print("\n(rows where both columns are dashes have no metric key in either run; skipped)\n")


if __name__ == "__main__":
    main()
