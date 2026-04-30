"""Layer 2 — Transition Typing.

For each TRANSITION interval identified by the Layer 1 oracle, this module:
1. Looks up the valid transition type from (label_before, label_after).
2. Verifies the physical signature from Allg_M1 derivatives, using both the
   oracle window and a pre-context window (PRE_CTX_S seconds before the interval).
3. Flags INVALID_TOPOLOGY if the pair is forbidden.
4. Flags micro_event=True if the interval is shorter than micro_event_threshold_s,
   but still types and checks the signature when flanking labels are known.

Output: list of TransitionSegment objects, serialisable to transition_segments.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.modeling.mode.p1_physics.oracle import CODE_LABEL, LABEL_CODE, OracleResult
from src.modeling.mode.p1_physics.thresholds import SignalThresholds
from .signatures import FORBIDDEN_DIRECT, SIGNATURE_CHECKERS, VALID_TRANSITIONS

# Seconds of Allg_M1 data to pass as pre-context to signature checkers.
# Many physical signal changes (gate closing, excitation ramp-down) begin before
# the oracle assigns LOW confidence.
PRE_CTX_S = 120


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransitionSegment:
    t_start_s: int
    t_end_s: int
    duration_s: int
    label_before: str         # steady-state label before the interval
    label_after: str          # steady-state label after the interval
    transition_type: str      # e.g. 'ST→TU', 'INVALID_TOPOLOGY', 'UNKNOWN'
    signature_match: bool
    signature_details: dict[str, Any] = field(default_factory=dict)
    micro_event: bool = False
    invalid_topology: bool = False


# ---------------------------------------------------------------------------
# Segment extraction from oracle label array
# ---------------------------------------------------------------------------


def _find_transition_intervals(
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    transition_code: int,
    unknown_code: int,
    low_conf_code: int,
) -> list[tuple[int, int]]:
    """Return (t_start, t_end) pairs for contiguous LOW-confidence intervals."""
    is_gap = (confidence == low_conf_code)
    intervals = []
    in_gap = False
    start = 0
    for t in range(len(labels)):
        if is_gap[t] and not in_gap:
            in_gap = True
            start = t
        elif not is_gap[t] and in_gap:
            in_gap = False
            intervals.append((start, t - 1))
    if in_gap:
        intervals.append((start, len(labels) - 1))
    return intervals


def _surrounding_steady(
    labels: np.ndarray,
    confidence: np.ndarray,
    t_start: int,
    t_end: int,
    *,
    steady_label_codes: frozenset,
    high_conf_code: int,
    search_window: int = 3600,
) -> tuple[str | None, str | None]:
    """Find the nearest HIGH-confidence steady labels before t_start and after t_end."""
    label_before = None
    search_lo = max(0, t_start - search_window)
    for t in range(t_start - 1, search_lo - 1, -1):
        if confidence[t] == high_conf_code and labels[t] in steady_label_codes:
            label_before = CODE_LABEL[int(labels[t])]
            break

    label_after = None
    search_hi = min(len(labels), t_end + search_window)
    for t in range(t_end + 1, search_hi):
        if confidence[t] == high_conf_code and labels[t] in steady_label_codes:
            label_after = CODE_LABEL[int(labels[t])]
            break

    return label_before, label_after


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def type_transitions(
    oracle: OracleResult,
    allg: np.ndarray,
    channel_names: list[str],
    thresholds: SignalThresholds,
    *,
    micro_event_threshold_s: int = 60,
    search_window_s: int = 3600,
) -> list[TransitionSegment]:
    """Type all TRANSITION/UNKNOWN intervals in the oracle output.

    Parameters
    ----------
    oracle:
        OracleResult from Layer 1.
    allg:
        (T, N_ch) float32 Allg_M1 data at 1 Hz.
    channel_names:
        Channel names matching allg columns.
    thresholds:
        Derived thresholds from Layer 1.
    micro_event_threshold_s:
        Intervals shorter than this are flagged micro_event=True, but are still
        typed and signature-checked when flanking labels are known.
    search_window_s:
        How far to look (in seconds) before/after an interval for the flanking
        steady labels.
    """
    from src.modeling.mode.p1_physics.oracle import (
        CONFIDENCE_CODE,
        LABEL_CODE,
        STEADY_LABEL_CODES,
    )

    low_code  = CONFIDENCE_CODE["LOW"]
    high_code = CONFIDENCE_CODE["HIGH"]

    intervals = _find_transition_intervals(
        oracle.labels,
        oracle.confidence,
        transition_code=LABEL_CODE["TRANSITION"],
        unknown_code=LABEL_CODE["UNKNOWN"],
        low_conf_code=low_code,
    )

    segments: list[TransitionSegment] = []

    for t_start, t_end in intervals:
        duration = t_end - t_start + 1
        is_micro = duration < micro_event_threshold_s

        label_before, label_after = _surrounding_steady(
            oracle.labels,
            oracle.confidence,
            t_start,
            t_end,
            steady_label_codes=STEADY_LABEL_CODES,
            high_conf_code=high_code,
            search_window=search_window_s,
        )

        # Cannot type without flanking labels
        if label_before is None or label_after is None:
            seg = TransitionSegment(
                t_start_s=t_start,
                t_end_s=t_end,
                duration_s=duration,
                label_before=label_before or "UNKNOWN",
                label_after=label_after or "UNKNOWN",
                transition_type="UNKNOWN",
                signature_match=False,
                micro_event=is_micro,
            )
            segments.append(seg)
            continue

        pair = (label_before, label_after)

        # Forbidden direct transition
        if pair in FORBIDDEN_DIRECT:
            seg = TransitionSegment(
                t_start_s=t_start,
                t_end_s=t_end,
                duration_s=duration,
                label_before=label_before,
                label_after=label_after,
                transition_type="INVALID_TOPOLOGY",
                signature_match=False,
                micro_event=is_micro,
                invalid_topology=True,
            )
            segments.append(seg)
            continue

        # Look up valid transition type
        trans_type = VALID_TRANSITIONS.get(pair)
        if trans_type is None:
            if label_before == label_after:
                trans_type = f"{label_before}→{label_after}"
            else:
                trans_type = f"{label_before}→{label_after}(undocumented)"

        # Pre-context window for signature checkers
        pre_start  = max(0, t_start - PRE_CTX_S)
        allg_pre   = allg[pre_start:t_start]
        allg_window = allg[t_start:t_end + 1]

        # Signature check (runs even for micro-events with valid flanking labels)
        checker = SIGNATURE_CHECKERS.get(trans_type)
        if checker is not None:
            sig_result = checker(allg_window, allg_pre, channel_names, thresholds)
        else:
            sig_result = {"match": None, "reason": "no checker defined"}

        sig_match = bool(sig_result.get("match", False))

        seg = TransitionSegment(
            t_start_s=t_start,
            t_end_s=t_end,
            duration_s=duration,
            label_before=label_before,
            label_after=label_after,
            transition_type=trans_type,
            signature_match=sig_match,
            signature_details=sig_result,
            micro_event=is_micro,
        )
        segments.append(seg)

    return segments


def save_transition_segments(
    segments: list[TransitionSegment],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(s) for s in segments], f, indent=2, default=str)


def load_transition_segments(path: str | Path) -> list[TransitionSegment]:
    with open(path) as f:
        data = json.load(f)
    return [TransitionSegment(**d) for d in data]
