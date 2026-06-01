"""Side-by-side comparison: full_run at hop=43 vs the hop=512 baseline.

Reads ``results/full_run_hop512_baseline/metrics.json`` and
``results/full_run_hop43/metrics.json`` and prints a Markdown-friendly table of
every comparable headline metric, grouped by stage. Highlights regressions and
improvements with a +/- delta.

Usage::

    python scripts/compare_hop43_vs_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results" / "full_run_hop512_baseline" / "metrics.json"
NEW  = REPO / "results" / "full_run_hop43"           / "metrics.json"


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


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"Baseline missing: {BASE}")
    if not NEW.exists():
        raise SystemExit(f"hop=43 metrics missing: {NEW}")
    B = json.load(BASE.open())
    N = json.load(NEW.open())

    print("# hop=43 vs hop=512 baseline — full pipeline comparison\n")
    print(f"Baseline: {BASE}")
    print(f"hop=43:   {NEW}\n")

    # ------- V0 baselines -------
    section("V0 baselines (hop-independent - sanity check)")
    for k in ("d1", "d2"):
        bv = _get(B, "stages", "v0", f"v0_lgbm_{k}", "val_macro_f1")
        nv = _get(N, "stages", "v0", f"v0_lgbm_{k}", "val_macro_f1")
        if bv is not None or nv is not None:
            row(f"V0 LGBM mode {k} macro-F1", bv, nv)
    for k in ("d1", "d2", "d3", "d4"):
        bv = _get(B, "stages", "v0", f"v0_lstm_ae_{k}", "val_recon_mse")
        nv = _get(N, "stages", "v0", f"v0_lstm_ae_{k}", "val_recon_mse")
        if bv is not None or nv is not None:
            row(f"V0 LSTM-AE {k} val MSE", bv, nv, prec=4)
    for k in ("d2", "d3", "d4"):
        bv = _get(B, "stages", "v0", f"v0_srp_phat_{k}", "mean_mae_m")
        nv = _get(N, "stages", "v0", f"v0_srp_phat_{k}", "mean_mae_m")
        if bv is not None or nv is not None:
            row(f"V0 SRP-PHAT {k} mean MAE (m)", bv, nv)

    # ------- V1 acoustic -------
    section("V1 acoustic SSL (sanity gate = mode clustering)")
    for k in ("sanity_nmi", "sanity_ari", "sanity_purity", "train_loss_final", "val_loss_final"):
        row(k, _get(B, "stages", "v1_acoustic", k), _get(N, "stages", "v1_acoustic", k))

    # ------- V1 vibration -------
    section("V1 vibration SSL (note: should be unchanged; any delta is RNG side-effect)")
    for k in ("sanity_nmi", "sanity_ari", "sanity_purity", "train_loss_final", "val_loss_final"):
        row(k, _get(B, "stages", "v1_vibration", k), _get(N, "stages", "v1_vibration", k))

    # ------- V2 -------
    section("V2 fusion (RQ1 mode-clustering on fused tokens)")
    for k in ("rq1_nmi", "rq1_ari", "rq1_purity", "train_loss_final", "val_loss_final", "train_simclr_final", "train_lmm_final"):
        row(k, _get(B, "stages", "v2", k), _get(N, "stages", "v2", k))

    section("V2-A1 ablation (drop vibration during training)")
    for k in ("rq1_nmi", "rq1_ari", "rq1_purity"):
        row(k, _get(B, "stages", "v2_a1_drop_vibration", k), _get(N, "stages", "v2_a1_drop_vibration", k))

    section("V2 modality probe (eval-time modality zero-out)")
    for path_name in ("both", "acoustic_only", "vibration_only"):
        for k in ("nmi", "ari", "purity"):
            row(f"{path_name}.{k}",
                _get(B, "stages", "v2_modality_probe", path_name, k),
                _get(N, "stages", "v2_modality_probe", path_name, k))

    # ------- V3 -------
    section("V3 conditional CNF training")
    for k in ("train_nll_final", "val_nll_final"):
        row(k, _get(B, "stages", "v3", k), _get(N, "stages", "v3", k), prec=2)

    section("V3 transition false-alert rate (synthetic transition stress-test)")
    for k in ("d1_pump_to_turbine", "d1_turbine_to_pump", "d1_to_d2_pump", "d1_to_d2_turbine"):
        row(k, _get(B, "stages", "v3_rq2_transition_fpr", k), _get(N, "stages", "v3_rq2_transition_fpr", k), prec=4)

    section("V3 vs A2 (conditional vs unconditional) paired bootstrap on val NLL")
    for k in ("delta_point", "delta_ci95_low", "delta_ci95_high", "p_value_two_sided", "direction"):
        row(k, _get(B, "stages", "v3_vs_a2_paired_test", k), _get(N, "stages", "v3_vs_a2_paired_test", k), prec=4)

    section("V3 per-cohort alert rate (THE anomaly-detection headline)")
    for cohort in ("healthy_holdout", "d2_random_fault", "d3_hit", "d4_random_fault"):
        bv = _get(B, "stages", "v3_cohort_validation", cohort, "alert_rate")
        nv = _get(N, "stages", "v3_cohort_validation", cohort, "alert_rate")
        n_b = _get(B, "stages", "v3_cohort_validation", cohort, "n")
        n_n = _get(N, "stages", "v3_cohort_validation", cohort, "n")
        note = f"n_b={n_b}, n_n={n_n}"
        row(f"{cohort} alert_rate", bv, nv, note=note)

    # ------- V4 -------
    section("V4 localization head (3-D MAE; lower is better)")
    for k in ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high", "val_p95_3d",
              "val_init_mae_3d", "val_residual_norm_mean", "train_loss_final", "val_loss_final"):
        bv = _get(B, "stages", "v4", k)
        nv = _get(N, "stages", "v4", k)
        if bv is not None or nv is not None:
            row(k, bv, nv, prec=4)

    section("V4-A3 unconditional ablation")
    for k in ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high", "val_p95_3d"):
        bv = _get(B, "stages", "v4_a3_unconditional", k)
        nv = _get(N, "stages", "v4_a3_unconditional", k)
        if bv is not None or nv is not None:
            row(k, bv, nv, prec=4)

    section("V4 vs A3 paired bootstrap on per-window 3-D error")
    for k in ("delta_mae_m", "delta_mae_ci95_low_m", "delta_mae_ci95_high_m", "p_value_two_sided", "direction"):
        row(k, _get(B, "stages", "v4_vs_a3_paired_test", k), _get(N, "stages", "v4_vs_a3_paired_test", k), prec=4)

    section("V4-A5 SRP-only ablation")
    for k in ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high"):
        bv = _get(B, "stages", "v4_a5_srp_only", k)
        nv = _get(N, "stages", "v4_a5_srp_only", k)
        if bv is not None or nv is not None:
            row(k, bv, nv, prec=4)

    section("V4-A5 TDOA-only ablation")
    for k in ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high"):
        bv = _get(B, "stages", "v4_a5_tdoa_only", k)
        nv = _get(N, "stages", "v4_a5_tdoa_only", k)
        if bv is not None or nv is not None:
            row(k, bv, nv, prec=4)

    section("V5.1 fan-noise conditioning")
    for k in ("val_mae_3d", "val_mae_ci95_low", "val_mae_ci95_high"):
        bv = _get(B, "stages", "v5_1", k)
        nv = _get(N, "stages", "v5_1", k)
        if bv is not None or nv is not None:
            row(k, bv, nv, prec=4)

    print("\n(rows where both columns are dashes have no metric key in either run; skipped)\n")


if __name__ == "__main__":
    main()
