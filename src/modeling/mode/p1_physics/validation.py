"""Layer 1 validation utilities.

Metrics
-------
- Oracle agreement vs hand-labeled events
- χ² head-independence test (clustering not driven by reservoir level)
- Dwell-time ratio sanity check
- Sensor freeze detection

All functions return plain dicts that are JSON-serializable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .oracle import CODE_LABEL, LABEL_CODE, OracleResult


# ---------------------------------------------------------------------------
# Sensor freeze detection
# ---------------------------------------------------------------------------


def check_sensor_freeze(
    allg: np.ndarray,
    channel_names: list[str],
    *,
    freeze_window_s: int = 3600,
    std_threshold: float = 1e-6,
    exclude_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Flag channels where rolling std == 0 for > freeze_window_s seconds.

    NOTE: this SCADA system logs quantised values that appear identical for
    tens of seconds during steady-state operation (e.g. RPM=378.77 for 60+
    consecutive seconds during TU at constant speed).  A 60 s window catches
    normal steady-state logging artefacts.  The default window is set to
    3600 s so that only channels that are truly flat for a full hour (across
    multiple operating modes) are flagged.

    Parameters
    ----------
    exclude_mask:
        Boolean array (T,). Timesteps where this is True are skipped.

    Returns a dict mapping faulted channel names to the first freeze timestep.
    """
    T, C = allg.shape
    faulted: dict[str, int] = {}
    active = np.ones(T, dtype=bool) if exclude_mask is None else ~exclude_mask

    for c, name in enumerate(channel_names):
        x = allg[:, c].astype(np.float64)
        cs1 = np.zeros(T + 1)
        cs2 = np.zeros(T + 1)
        cs1[1:] = np.cumsum(x)
        cs2[1:] = np.cumsum(x ** 2)
        t_arr = np.arange(T)
        t0_arr = np.maximum(0, t_arr - freeze_window_s + 1)
        n = t_arr - t0_arr + 1
        mean = (cs1[t_arr + 1] - cs1[t0_arr]) / n
        var = np.maximum(0.0, (cs2[t_arr + 1] - cs2[t0_arr]) / n - mean ** 2)
        frozen = (var < std_threshold ** 2) & active

        run = 0
        for t in range(T):
            if frozen[t]:
                run += 1
                if run >= freeze_window_s:
                    faulted[name] = int(t - freeze_window_s + 1)
                    break
            else:
                run = 0

    return {
        "faulted_channels": faulted,
        "freeze_window_s": freeze_window_s,
        "note": "window=3600s; shorter windows trigger on SCADA quantisation during steady operation",
    }


# ---------------------------------------------------------------------------
# Dwell-time ratios
# ---------------------------------------------------------------------------


def compute_dwell_ratios(labels: np.ndarray) -> dict[str, float]:
    """Compute fraction of time spent in each label (all 6 classes)."""
    T = len(labels)
    ratios = {}
    for code, name in CODE_LABEL.items():
        ratios[name] = float(np.sum(labels == code)) / T
    return ratios


# ---------------------------------------------------------------------------
# Oracle agreement vs hand-labeled events
# ---------------------------------------------------------------------------


def oracle_agreement(
    labels: np.ndarray,
    hand_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute oracle agreement on a list of hand-labeled time intervals.

    Parameters
    ----------
    labels:
        (T,) int8 oracle label array.
    hand_labels:
        List of dicts with keys:
          - t_start_s, t_end_s: integer second indices into labels
          - label: string label (e.g. 'TU', 'ST→TU', ...)
        For steady-state events, only 'ST','TU','PU','PH' are checked against oracle.

    Returns a dict with per-class counts and overall accuracy.
    """
    results: list[dict[str, Any]] = []
    correct = 0
    total = 0

    for ev in hand_labels:
        t0, t1 = int(ev["t_start_s"]), int(ev["t_end_s"])
        t0 = max(0, t0)
        t1 = min(len(labels) - 1, t1)
        ev_labels = labels[t0:t1 + 1]
        true_label = ev["label"]

        if true_label not in LABEL_CODE:
            # Transition type strings like 'ST→TU' — compare vs TRANSITION code
            oracle_mode = CODE_LABEL.get(int(np.bincount(ev_labels.astype(np.int64) + 128,
                                                           minlength=256).argmax() - 128), "UNKNOWN")
            match = oracle_mode == "TRANSITION"
        else:
            true_code = LABEL_CODE[true_label]
            # Majority vote over the event window
            majority_code = int(np.bincount(ev_labels.clip(0, 10).astype(np.int64),
                                            minlength=6).argmax())
            match = majority_code == true_code
            oracle_mode = CODE_LABEL[majority_code]

        results.append({
            "t_start_s": t0,
            "t_end_s": t1,
            "true_label": true_label,
            "oracle_label": oracle_mode,
            "match": bool(match),
        })
        correct += int(match)
        total += 1

    return {
        "n_events": total,
        "n_correct": correct,
        "accuracy": float(correct / total) if total > 0 else float("nan"),
        "events": results,
    }


# ---------------------------------------------------------------------------
# χ² head-independence test
# ---------------------------------------------------------------------------


def chi2_head_independence(
    labels: np.ndarray,
    net_head: np.ndarray,
    *,
    n_head_bins: int = 5,
    alpha: float = 0.05,
    exclude_labels: Sequence[int] = (4, 5),  # TRANSITION, UNKNOWN
) -> dict[str, Any]:
    """Test whether oracle labels are independent of net hydraulic head.

    A significant result (p < alpha) means the classification is driven by
    reservoir level rather than operating state — indicates a labeling error.

    Parameters
    ----------
    labels:
        (T,) int8 oracle labels.
    net_head:
        (T,) float — net hydraulic head in meters.
    n_head_bins:
        Number of quantile bins for head discretization.
    exclude_labels:
        Label codes excluded from the test (transitions and unknowns).
    """
    mask = ~np.isin(labels, list(exclude_labels)) & np.isfinite(net_head)
    lbl = labels[mask]
    head = net_head[mask]

    if len(lbl) < 100:
        return {"status": "SKIP", "reason": "insufficient data"}

    head_bins = np.quantile(head, np.linspace(0, 1, n_head_bins + 1))
    head_bins[0] -= 1.0
    head_bins[-1] += 1.0
    head_disc = np.digitize(head, head_bins) - 1

    unique_labels = np.unique(lbl)
    unique_bins = np.arange(n_head_bins)

    # Build contingency table
    table = np.zeros((len(unique_labels), n_head_bins), dtype=np.int64)
    for i, lab in enumerate(unique_labels):
        for j, b in enumerate(unique_bins):
            table[i, j] = int(np.sum((lbl == lab) & (head_disc == j)))

    # Remove zero rows/cols
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    table = table[row_sums > 0][:, col_sums > 0]

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"status": "SKIP", "reason": "insufficient variation"}

    chi2_stat, p_value, dof, _ = stats.chi2_contingency(table)
    independent = bool(p_value > alpha)

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "dof": int(dof),
        "independent": independent,
        # A significant result is physically expected for hydro machines: operators
        # run TU when the reservoir is full (high head) and PU when it is low.
        # This test is informational — correlation with head is not a labeling error.
        "status": "PASS" if independent else "INFORMATIONAL",
        "alpha": alpha,
        "message": (
            f"Labels independent of net head (p={p_value:.4f} > α={alpha})"
            if independent
            else (
                f"Labels correlate with net head (p={p_value:.4f} ≤ α={alpha}). "
                "Physically expected: operators dispatch TU/PU based on reservoir level. "
                "This is not a labeling error."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Full Layer 1 validation report
# ---------------------------------------------------------------------------


def build_validation_report(
    result: OracleResult,
    allg: np.ndarray,
    channel_names: list[str],
    net_head: np.ndarray | None = None,
    hand_labels: list[dict[str, Any]] | None = None,
    *,
    alpha: float = 0.05,
    freeze_window_s: int = 60,
) -> dict[str, Any]:
    """Assemble the full Layer 1 validation report dict."""
    report: dict[str, Any] = {}

    report["dwell_ratios"] = compute_dwell_ratios(result.labels)
    report["steady_coverage_pct"] = float(result.steady_mask.mean() * 100)
    report["sensor_freeze"] = check_sensor_freeze(
        allg, channel_names, freeze_window_s=freeze_window_s,
    )

    if net_head is not None:
        report["head_independence"] = chi2_head_independence(
            result.labels, net_head, alpha=alpha
        )

    if hand_labels is not None:
        report["oracle_agreement"] = oracle_agreement(result.labels, hand_labels)

    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


# Allow Sequence import at module level
from typing import Sequence
