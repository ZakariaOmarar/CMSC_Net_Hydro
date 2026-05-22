"""Aggregate `scripts.ablation_full_pipeline` cells into a markdown report.

Reads every ``results/runs/*__ablation_*/metrics.json`` + ``cell_config.json``
and emits four tables to ``results/runs/ablation_report_<ts>.md``:

  1. **V1/V2 per-cell metrics** — sanity NMI, RQ1 NMI, modality-probe Δ,
     train/val gaps.  This is the primary Phase 2 / 3 selection table.
  2. **Deep-vs-simple comparison** — V3 CNF vs KDE NLL, V4 fusion vs V0
     multilateration MAE, per cell.  Empty rows for cells that ran with
     --skip-v3 / --skip-v4.
  3. **Phase 2 / 3 axis sweeps** — cells grouped by axis with the
     baseline_v2 row highlighted.  Lets you eyeball the per-axis ordering.
  4. **Phase 5 multi-seed verdict** — when a cell ID appears at multiple
     seeds, report mean ± std across seeds.

Selection guidance (printed at the bottom of the report):
  - V1 / V2 tuple: maximise (sanity_NMI, sanity_purity, modality_probe Δ),
    minimise absolute train/val gap.
  - Deep-vs-simple: cells where `deep_wins=True` for V3 or V4 dominate
    the framing-flip discussion.

Run::

    python -m scripts.analyze_ablation
    python -m scripts.analyze_ablation --filter "p2_a1_*"   # glob
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "results" / "runs"


def _load_runs(
    filter_glob: str | None,
    campaign_dir: Path | None = None,
) -> list[dict]:
    """Read ablation run dirs' metrics + config; filter by cell id and/or campaign.

    When ``campaign_dir`` is set, the run-dir set is restricted to those listed
    in the campaign's ``state.json``.  This keeps the report scoped to a single
    campaign even when many historical ablation runs live in ``results/runs``.
    Cells from PHASE 6 (`v4_aug_sweep`) live in `*__v4aug_*` dirs and are
    included in the campaign scope too.
    """
    if campaign_dir is not None:
        state_path = campaign_dir / "state.json"
        if not state_path.exists():
            return []
        state = json.loads(state_path.read_text())
        run_dirs: list[Path] = []
        baseline = state.get("baseline_v2") or {}
        if baseline.get("run_dir"):
            run_dirs.append(Path(baseline["run_dir"]))
        for _key, entry in (state.get("cells") or {}).items():
            if entry.get("run_dir"):
                run_dirs.append(Path(entry["run_dir"]))
    else:
        run_dirs = sorted(RUNS_DIR.glob("*__ablation_*")) + sorted(
            RUNS_DIR.glob("*__v4aug_*")
        )

    rows: list[dict] = []
    seen_dirs: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        config_path = run_dir / "cell_config.json"
        try:
            metrics = json.loads(metrics_path.read_text())
            cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
        except Exception:
            continue
        # Baseline_v2 dirs (``*__full_pipeline_b5_cma``) have no
        # cell_config.json; synthesise a minimal one.
        cell = cfg.get("cell")
        if cell is None:
            if "__full_pipeline_b5_cma" in run_dir.name:
                cell = "baseline_v2"
            else:
                cell = metrics.get("cell", "?")
        if filter_glob and not fnmatch.fnmatch(cell, filter_glob):
            continue
        rows.append({"dir": run_dir, "cell": cell, "cfg": cfg, "metrics": metrics})
    return rows


def _fmt(v, prec: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _gap_abs(train: float | None, val: float | None) -> float | None:
    if train is None or val is None:
        return None
    if not isinstance(train, (int, float)) or not isinstance(val, (int, float)):
        return None
    return abs(val - train)


def _table_v1_v2(rows: list[dict]) -> str:
    cols = [
        "cell", "seed", "v1ac_NMI", "v1ac_purity", "v1ac_gap",
        "v1vib_NMI", "v1vib_purity", "v1vib_gap",
        "v2_NMI", "v2_purity", "modΔ", "v2_gap", "v1ac_stop", "v2_stop",
    ]
    out = ["## Table 1 — V1/V2 per-cell metrics", "",
           "| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        st = r["metrics"].get("stages", {})
        v1a = st.get("v1_acoustic", {})
        v1v = st.get("v1_vibration", {})
        v2 = st.get("v2", {})
        probe = st.get("v2_modality_probe", {})
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _fmt(v1a.get("sanity_nmi")),
            _fmt(v1a.get("sanity_purity")),
            _fmt(_gap_abs(v1a.get("train_loss_final"), v1a.get("val_loss_final"))),
            _fmt(v1v.get("sanity_nmi")),
            _fmt(v1v.get("sanity_purity")),
            _fmt(_gap_abs(v1v.get("train_loss_final"), v1v.get("val_loss_final"))),
            _fmt(v2.get("rq1_nmi")),
            _fmt(v2.get("rq1_purity")),
            _fmt(probe.get("delta_both_minus_acoustic")),
            _fmt(_gap_abs(v2.get("train_loss_final"), v2.get("val_loss_final"))),
            _fmt(v1a.get("early_stopped_epoch"), prec=0),
            _fmt(v2.get("early_stopped_epoch"), prec=0),
        ]) + " |")
    return "\n".join(out) + "\n"


def _table_deep_vs_simple(rows: list[dict]) -> str:
    cols = [
        "cell", "seed",
        "V3 NLL", "KDE NLL", "Δ (V3-KDE)", "V3 wins",
        "V4 MAE (m)", "V0 multilat MAE (m)", "Δ (V4-V0)", "V4 wins",
    ]
    out = ["## Table 2 — Deep-vs-simple comparison", "",
           "Δ < 0 means the deep model beats the simple baseline.", "",
           "| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        dvs = r["metrics"].get("full_run_deep_vs_simple", {}) or r["metrics"].get("deep_vs_simple", {})
        anom = dvs.get("anomaly", {}) if isinstance(dvs, dict) else {}
        loc = dvs.get("localisation", {}) if isinstance(dvs, dict) else {}
        out.append("| " + " | ".join([
            r["cell"], str(r["cfg"].get("seed", "?")),
            _fmt(anom.get("deep_val_nll_mean")),
            _fmt(anom.get("simple_val_nll_mean")),
            _fmt(anom.get("delta_deep_minus_simple")),
            _fmt(anom.get("deep_wins")),
            _fmt(loc.get("deep_val_mae_m")),
            _fmt(loc.get("simple_val_mae_m")),
            _fmt(loc.get("delta_deep_minus_simple_m")),
            _fmt(loc.get("deep_wins")),
        ]) + " |")
    return "\n".join(out) + "\n"


def _axis_grouping(rows: list[dict]) -> str:
    """Group p2_* and p3_* cells by axis level for eyeball-comparison."""
    out = ["## Table 3 — Phase 2 / 3 axis sweeps", ""]

    # Phase 2 — aug × vibration_dropout grid
    p2_rows = [r for r in rows if r["cell"].startswith("p2_")]
    if p2_rows:
        out.append("### Phase 2: aug strength × vibration_dropout (V2 RQ1 NMI / modality Δ)")
        out.append("")
        out.append("| aug \\ vib_drop | 0.3 (v3) | 0.5 (v5, baseline) | 0.7 (v7) |")
        out.append("|---|---|---|---|")
        for aug_lvl, aug_label in (
            ("a0", "mid (baseline)"), ("a1", "strong"), ("a2", "very-strong"),
        ):
            cells = ["| " + aug_label]
            for vib_lvl in ("v3", "v5", "v7"):
                cell_id = f"p2_{aug_lvl}_{vib_lvl}"
                matched = [r for r in p2_rows if r["cell"] == cell_id]
                if not matched:
                    cells.append("—")
                    continue
                r = matched[0]
                v2 = r["metrics"].get("stages", {}).get("v2", {})
                probe = r["metrics"].get("stages", {}).get("v2_modality_probe", {})
                cells.append(f"NMI={_fmt(v2.get('rq1_nmi'))} / Δ={_fmt(probe.get('delta_both_minus_acoustic'))}")
            out.append(" | ".join(cells) + " |")
        out.append("")

    # Phase 3 — mixup × embed_dim grid
    p3_rows = [r for r in rows if r["cell"].startswith("p3_")]
    if p3_rows:
        out.append("### Phase 3: mixup_alpha × embed_dim (V2 RQ1 NMI / modality Δ)")
        out.append("")
        out.append("| mixup \\ embed_dim | 32 | 64 (baseline) | 128 |")
        out.append("|---|---|---|---|")
        for mix_lvl, mix_label in (("m0", "0.0"), ("m2", "0.2"), ("m4", "0.4")):
            cells = ["| " + mix_label]
            for edim_lvl in ("e32", "e64", "e128"):
                cell_id = f"p3_{mix_lvl}_{edim_lvl}"
                matched = [r for r in p3_rows if r["cell"] == cell_id]
                if not matched:
                    cells.append("—")
                    continue
                r = matched[0]
                v2 = r["metrics"].get("stages", {}).get("v2", {})
                probe = r["metrics"].get("stages", {}).get("v2_modality_probe", {})
                cells.append(f"NMI={_fmt(v2.get('rq1_nmi'))} / Δ={_fmt(probe.get('delta_both_minus_acoustic'))}")
            out.append(" | ".join(cells) + " |")
        out.append("")

    return "\n".join(out) + "\n"


def _multi_seed(rows: list[dict]) -> str:
    """Group by cell id and report mean±std when ≥ 2 seeds present."""
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    out = ["## Table 4 — Multi-seed verdict", ""]
    out.append("Cells with ≥ 2 seeds.  Mean ± std reported across seeds.")
    out.append("")
    out.append("| cell | n_seeds | V2 NMI | V4 MAE (m) | V3 NLL | V3-KDE Δ |")
    out.append("|---|---|---|---|---|---|")
    any_row = False
    for cell, runs in sorted(by_cell.items()):
        if len(runs) < 2:
            continue
        any_row = True

        def _collect(path: list[str]) -> list[float]:
            vals: list[float] = []
            for r in runs:
                node = r["metrics"]
                for k in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(k)
                if isinstance(node, (int, float)) and not (
                    isinstance(node, float) and math.isnan(node)
                ):
                    vals.append(float(node))
            return vals

        def _mean_std(vals: list[float]) -> str:
            if not vals:
                return "—"
            if len(vals) == 1:
                return f"{vals[0]:.3f}"
            return f"{mean(vals):.3f} ± {stdev(vals):.3f}"

        nmi = _collect(["stages", "v2", "rq1_nmi"])
        mae = _collect(["full_run_stages", "v4_four_paradigms", "fusion", "val_mae_3d"])
        v3_nll = _collect(["full_run_deep_vs_simple", "anomaly", "deep_val_nll_mean"])
        d_anom = _collect(["full_run_deep_vs_simple", "anomaly", "delta_deep_minus_simple"])
        out.append(f"| {cell} | {len(runs)} | {_mean_std(nmi)} | {_mean_std(mae)} | "
                   f"{_mean_std(v3_nll)} | {_mean_std(d_anom)} |")
    if not any_row:
        out.append("| (no multi-seed cells yet) | | | | | |")
    return "\n".join(out) + "\n"


def _guidance(rows: list[dict]) -> str:
    return (
        "## Selection guidance\n\n"
        "- **Phase 2 winner:** pick the cell in Table 3 (Phase 2 grid) with the highest V2 RQ1 NMI AND positive modality-probe Δ.  Cells whose Δ is more negative than baseline_v2 are rejected even if their NMI is higher (acoustic-only collapse).\n"
        "- **Phase 3 winner:** repeat over Table 3 (Phase 3 grid) with the Phase 2 winner already baked into the base cell.\n"
        "- **Phase 4 promotion:** top 3 cells from Phases 2+3 by V2 tuple → re-run without `--skip-v3 --skip-v4` so Tables 2 and 4 fill in.\n"
        "- **Phase 5 verdict:** the Phase 4 winner re-run at seeds {1337, 2024}; Table 4 should show the cell beating baseline_v2 by > 2× the seed std on V3 NLL Δ or V4 MAE.\n"
        "- **Deep-vs-simple framing (Table 2):** if `V3 wins` is False, the thesis primary anomaly result is KDE.  If `V4 wins` is False, the thesis primary localisation result is V0 multilateration.  Run the campaign anyway — the rest of the table is still a contribution.\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default=None,
                   help="Cell-id glob (e.g. 'p2_a1_*'); default: include all ablation runs.")
    p.add_argument(
        "--campaign-dir", default=None,
        help="Path to results/runs/campaign_<ts>; scope report to that campaign's cells only.",
    )
    args = p.parse_args()

    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else None
    rows = _load_runs(args.filter, campaign_dir=campaign_dir)
    if not rows:
        scope = f"campaign {campaign_dir}" if campaign_dir else f"{RUNS_DIR}"
        print(f"No ablation run dirs found in {scope}.")
        return

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    source = f"{campaign_dir}" if campaign_dir else f"{RUNS_DIR}"
    report = [
        f"# Ablation Report ({timestamp})",
        "",
        f"Source: {source}",
        f"Cells loaded: {len(rows)}",
        f"Filter: `{args.filter or '(none)'}`",
        "",
        _table_v1_v2(rows),
        _table_deep_vs_simple(rows),
        _axis_grouping(rows),
        _multi_seed(rows),
        _guidance(rows),
    ]
    # When scoped to a campaign, write the report INSIDE the campaign dir so
    # multiple campaigns can coexist on disk without report-name collisions.
    if campaign_dir:
        out_path = campaign_dir / "ablation_report.md"
    else:
        out_path = RUNS_DIR / f"ablation_report_{timestamp}.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
