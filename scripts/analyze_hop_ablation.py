"""Analyse the hop_length ablation CSV: pre-registered hypothesis test +
per-condition summary + per-pair effect size.

Pre-registered decision rule (mirrors the runner docstring):
  Reject H0 (i.e. conclude hop_length matters) ONLY IF both
    (a) |mean condition-difference in AUC@0dB| > 2 * pooled within-condition std
        across seeds, AND
    (b) Wilcoxon signed-rank p < 0.05 across seed-paired differences.
  Else accept H0; report Cohen's d and 95% bootstrap CI on the difference.

The Wilcoxon signed-rank test treats the per-seed AUC differences (cond_X - cond_Y)
as paired observations. With n_seeds=3 it is severely underpowered — interpret
non-significance as "no evidence of effect at this power", not "proven null".
With n_seeds>=5 the test starts to have meaningful sensitivity.

Usage::

    python scripts/analyze_hop_ablation.py --csv results/ablation_hop/<ts>/metrics.csv
    python scripts/analyze_hop_ablation.py --csv ... --metric auc_neg5db  # alt SNR
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


HEADLINE_METRIC = "auc_0db"  # primary; CLI can override


def _wilcoxon_signed_rank_p(diffs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (small-n exact)."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return float("nan")
    abs_ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    signs = np.sign(nz)
    w_plus = float(((signs > 0) * abs_ranks).sum())
    # Exact two-sided p for tiny n via enumerating all 2^n sign assignments
    if n <= 12:
        total = 1 << n
        count_ge = 0
        for mask in range(total):
            w = 0.0
            for i in range(n):
                if mask & (1 << i):
                    w += abs_ranks[i]
            if w >= w_plus:
                count_ge += 1
        # two-sided
        one_sided = count_ge / total
        return float(min(1.0, 2.0 * min(one_sided, 1.0 - one_sided + 1.0 / total)))
    # Normal approximation
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma if sigma > 0 else 0.0
    # erfc approximation for two-sided p
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _cohens_d_paired(diffs: list[float]) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    if len(diffs) < 2:
        return float("nan")
    sd = pstdev(diffs)
    if sd == 0:
        return float("inf") if mean(diffs) != 0 else 0.0
    return mean(diffs) / sd


def _bootstrap_ci_mean_diff(diffs: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs_arr = np.asarray(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(diffs_arr), size=len(diffs_arr))
        boots[i] = diffs_arr[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--metric", default=HEADLINE_METRIC,
                    help=f"Metric column from CSV (default: {HEADLINE_METRIC})")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("CSV is empty")

    # Bucket metric by condition; key by seed for pairing
    metric_by_cond: dict[str, dict[int, float]] = defaultdict(dict)
    walls_by_cond: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        cond = r["condition"]
        seed = int(r["seed"])
        val_raw = r.get(args.metric, "")
        if val_raw == "":
            continue
        try:
            val = float(val_raw)
        except ValueError:
            continue
        if math.isnan(val):
            continue
        metric_by_cond[cond][seed] = val
        try:
            walls_by_cond[cond].append(float(r["wall_total_s"]))
        except (ValueError, KeyError):
            pass

    conditions = sorted(metric_by_cond)
    if not conditions:
        raise SystemExit(f"No finite values found for metric '{args.metric}'")

    # ---------------- Per-condition summary ----------------
    print(f"\n== Hop-length ablation report — metric: {args.metric} ==")
    print(f"   CSV: {args.csv}")
    print(f"   conditions: {conditions}")
    print()

    print(f"{'condition':<16} | {'n_seeds':>7} | {'mean':>8} | {'std':>8} | {'min':>8} | {'max':>8} | {'wall_min/run':>13}")
    print("-" * 86)
    per_cond_stats = {}
    for cond in conditions:
        vals = sorted(metric_by_cond[cond].values())
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        wm = mean(walls_by_cond[cond]) / 60.0 if walls_by_cond[cond] else float("nan")
        per_cond_stats[cond] = (m, s, len(vals))
        print(f"{cond:<16} | {len(vals):>7d} | {m:>8.4f} | {s:>8.4f} | {vals[0]:>8.4f} | {vals[-1]:>8.4f} | {wm:>13.1f}")

    # Pooled within-condition std (for the "2 * pooled std" decision rule)
    all_devs = []
    for cond in conditions:
        vals = list(metric_by_cond[cond].values())
        if len(vals) > 1:
            m = mean(vals)
            all_devs.extend((v - m) ** 2 for v in vals)
    pooled_std = math.sqrt(mean(all_devs)) if all_devs else 0.0
    print(f"\nPooled within-condition std: {pooled_std:.4f}")
    print(f"2x pooled std (effect-size floor): {2 * pooled_std:.4f}")

    # ---------------- Pairwise test ----------------
    print("\n== Pairwise hypothesis tests (pre-registered rule) ==")
    print(f"{'pair':<32} | {'n_pairs':>7} | {'mean_diff':>10} | {'cohens_d':>9} | {'wilcoxon_p':>11} | {'CI_low':>8} | {'CI_high':>8} | {'verdict':<28}")
    print("-" * 130)

    n_cond = len(conditions)
    pair_results = []
    for i in range(n_cond):
        for j in range(i + 1, n_cond):
            c1, c2 = conditions[i], conditions[j]
            seeds_both = sorted(set(metric_by_cond[c1]) & set(metric_by_cond[c2]))
            diffs = [metric_by_cond[c1][s] - metric_by_cond[c2][s] for s in seeds_both]
            if not diffs:
                continue
            md = mean(diffs)
            d_eff = _cohens_d_paired(diffs)
            p = _wilcoxon_signed_rank_p(diffs)
            ci_low, ci_high = _bootstrap_ci_mean_diff(diffs)
            rule_a = abs(md) > 2 * pooled_std
            rule_b = (not math.isnan(p)) and (p < 0.05)
            if rule_a and rule_b:
                verdict = "REJECT H0 (effect)"
            elif rule_a and not rule_b:
                verdict = "effect-size yes, p>=0.05"
            elif (not rule_a) and rule_b:
                verdict = "p<0.05 but effect<2*sigma"
            else:
                verdict = "fail to reject H0"
            pair_label = f"{c1} - {c2}"
            print(
                f"{pair_label:<32} | {len(diffs):>7d} | {md:>+10.4f} | {d_eff:>+9.3f} | "
                f"{p:>11.4f} | {ci_low:>+8.4f} | {ci_high:>+8.4f} | {verdict:<28}"
            )
            pair_results.append({
                "pair": pair_label,
                "n_pairs": len(diffs),
                "mean_diff": md,
                "cohens_d": d_eff,
                "wilcoxon_p": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "verdict": verdict,
            })

    print()
    print("Decision rule reminder:")
    print("  Reject H0 only if BOTH |mean_diff| > 2*pooled_std AND wilcoxon_p < 0.05.")
    print(f"  With n_seeds={max(per_cond_stats[c][2] for c in conditions)} the Wilcoxon test is power-limited;")
    print("  treat 'fail to reject H0' as 'no evidence of effect at this power', not as proof of null.")


if __name__ == "__main__":
    main()
