"""Transition signature checkers for Layer 2.

Each checker receives the Allg_M1 signals over a candidate transition interval
(allg_window) and a pre-context window (allg_pre, up to 120 s before the
transition) and returns a dict with `match: bool` and per-criterion details.

Physical basis
--------------
ST→PU  SFC ramp + RPM rise 0→n_sync + valve opens
ST→TU  Gate ramp 0→open + rising RPM + (optional) spiral pressure rise
ST→PH  SFC ramp + RPM rise + gate stays closed (no water admission)
PH→TU  Gate opens from sealed position + (optional) spiral pressure transient
PH→PU  RPM direction reversal OR gate opens on pump side
PU→ST  Gate closed (was open in pre-context) + RPM decays + exc ramps down
TU→ST  Gate closes (from open) + RPM begins decaying (may enter PH en route)
PU→PH  Gate/valve closes + RPM holds at n_sync + exc holds
TU→PH  Gate closes + RPM holds + exc holds

Key timing note
---------------
The oracle LOW-confidence window starts when the STABILITY criterion fails,
which is after the physical signal changes have already begun. Pre-context
is required for checkers that need to confirm the starting state of a signal
(e.g., confirming gate was open before it started closing).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.modeling.mode.p1_physics.thresholds import (
    SignalThresholds,
    extract_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_rising(x: np.ndarray, *, min_rise_pct: float = 20.0) -> bool:
    """True if x rises by at least min_rise_pct of its range (p5→p95) over the interval."""
    clean = x[np.isfinite(x)]
    if len(clean) < 3:
        return False
    lo, hi = np.percentile(clean, [5, 95])
    return float(hi - lo) >= min_rise_pct / 100.0 * max(abs(lo), abs(hi), 1e-9)


def _is_falling(x: np.ndarray, *, min_fall_pct: float = 20.0) -> bool:
    return _is_rising(-x, min_rise_pct=min_fall_pct)


def _max_above(x: np.ndarray, threshold: float) -> bool:
    clean = x[np.isfinite(x)]
    return bool(len(clean) > 0 and np.max(clean) > threshold)


def _held_near(x: np.ndarray, target: float, tol: float, *, min_frac: float = 0.5) -> bool:
    """True if at least min_frac of the interval has |x − target| < tol."""
    clean = x[np.isfinite(x)]
    if len(clean) == 0:
        return False
    return float(np.mean(np.abs(clean - target) < tol)) >= min_frac


def _concat(pre: np.ndarray | None, arr: np.ndarray) -> np.ndarray:
    """Prepend pre-context to arr if available."""
    if pre is not None and len(pre) > 0:
        return np.concatenate([pre, arr])
    return arr


# ---------------------------------------------------------------------------
# Signal extractor for transition windows
# ---------------------------------------------------------------------------


def _signals(allg_window: np.ndarray, channel_names: list[str]) -> dict[str, np.ndarray | None]:
    """Extract all primary signals over a transition window."""
    names = ["rpm", "gate", "valve", "power", "exc", "sfc", "spiral_p", "flow_tu", "flow_pu"]
    return {n: extract_signal(allg_window, channel_names, n) for n in names}


# ---------------------------------------------------------------------------
# Individual signature checkers
# ---------------------------------------------------------------------------


def _check_sfc_active(
    sfc: np.ndarray | None, thr: SignalThresholds, min_duration_s: int = 60
) -> dict[str, Any]:
    if sfc is None:
        return {"match": False, "reason": "SFC channel not available"}
    active = np.abs(sfc) > thr.sfc_active_a
    n_active = int(np.sum(active))
    # For short transition windows scale down the requirement proportionally
    required = min(min_duration_s, max(30, len(sfc) // 2))
    return {"match": n_active >= required, "n_active_s": n_active, "required_s": required}


def _check_rpm_ramp_up(rpm: np.ndarray | None, thr: SignalThresholds) -> dict[str, Any]:
    if rpm is None:
        return {"match": False, "reason": "RPM channel not available"}
    match = _is_rising(np.abs(rpm), min_rise_pct=40.0)
    return {
        "match": bool(match),
        "rpm_start": float(np.abs(rpm[0])) if len(rpm) else float("nan"),
        "rpm_end": float(np.abs(rpm[-1])) if len(rpm) else float("nan"),
    }


def _check_rpm_ramp_down(
    rpm: np.ndarray | None, thr: SignalThresholds, *, pre_rpm: np.ndarray | None = None
) -> dict[str, Any]:
    if rpm is None:
        return {"match": False, "reason": "RPM channel not available"}
    # Use extended window (pre-context + transition) to capture decay that began before oracle window
    abs_pre = np.abs(pre_rpm) if pre_rpm is not None else None
    ext = _concat(abs_pre, np.abs(rpm))
    match = _is_falling(ext, min_fall_pct=25.0)
    return {"match": bool(match)}


def _check_gate_ramp_open(
    gate: np.ndarray | None, thr: SignalThresholds, *, pre_gate: np.ndarray | None = None
) -> dict[str, Any]:
    """Gate opens: was below open threshold, ramps up, ends above open threshold.

    The oracle TRANSITION window often starts after the gate has already crossed
    gate_open_pct (the oracle fires on stability loss, not on the gate crossing).
    Pre-context confirms the gate was closed before the transition began.
    """
    if gate is None:
        return {"match": False, "reason": "gate channel not available"}
    # Confirmed closed in pre-context (gate was below open threshold)
    pre_was_closed = (
        pre_gate is None or len(pre_gate) == 0
        or float(np.nanmax(pre_gate)) < thr.gate_open_pct
    )
    # Also accept: gate starts within 2 pct of open threshold (oracle fired early)
    starts_near_closed = len(gate) > 0 and float(gate[0]) < thr.gate_open_pct + 2.0
    was_closed = pre_was_closed or starts_near_closed

    ends_open = len(gate) > 0 and float(gate[-1]) > thr.gate_open_pct
    ramp = _is_rising(gate, min_rise_pct=30.0)
    return {"match": bool(was_closed and ends_open and ramp)}


def _check_gate_closes(
    gate: np.ndarray | None, thr: SignalThresholds, *, pre_gate: np.ndarray | None = None
) -> dict[str, Any]:
    """Gate closes: was above open threshold, now below it.

    Two physical cases:
    A) Gate closes within the oracle window (e.g. TU→ST: gate goes from 20% to 2%).
    B) Gate already closed before the oracle window started (e.g. PU→ST: gate
       closed during PH entry before oracle assigned LOW confidence).
    """
    if gate is None:
        return {"match": False, "reason": "gate channel not available"}
    gate_open_pct = thr.gate_open_pct

    pre_was_open = (
        pre_gate is not None and len(pre_gate) > 0
        and float(np.nanmax(pre_gate)) > gate_open_pct
    )
    starts_open = len(gate) > 0 and float(gate[0]) > gate_open_pct
    was_open = pre_was_open or starts_open

    # Case A: gate falls significantly within the oracle window and ends near the threshold.
    # Accept either the percentile-based test OR a large start→end drop: the latter handles
    # cases where gate holds steady then closes rapidly at the end of the oracle window.
    ramp = _is_falling(gate, min_fall_pct=30.0)
    start_end_drop = float(gate[0] - gate[-1]) if len(gate) > 1 else 0.0
    gate_dropped = ramp or (start_end_drop > gate_open_pct)
    # Allow up to 1 pct above gate_open_pct — the gate may not fully close within the window
    ends_near_closed = len(gate) > 0 and float(np.nanmin(gate)) < gate_open_pct + 1.0
    closed_within = starts_open and gate_dropped and ends_near_closed

    # Case B: gate was open in pre-context, already closed at window start
    already_closed = (
        pre_was_open and not starts_open
        and len(gate) > 0 and float(np.nanmax(gate)) < gate_open_pct
    )

    return {
        "match": bool(was_open and (closed_within or already_closed)),
        "pre_was_open": bool(pre_was_open),
        "starts_open": bool(starts_open),
        "closed_within": bool(closed_within),
        "already_closed": bool(already_closed),
    }


def _check_gate_stays_closed(gate: np.ndarray | None, thr: SignalThresholds) -> dict[str, Any]:
    if gate is None:
        return {"match": False, "reason": "gate channel not available"}
    all_closed = float(np.nanmax(gate)) < thr.gate_open_pct
    return {"match": bool(all_closed), "gate_max_pct": float(np.nanmax(gate))}


def _check_exc_ramp_down(
    exc: np.ndarray | None, thr: SignalThresholds, *, pre_exc: np.ndarray | None = None
) -> dict[str, Any]:
    if exc is None:
        return {"match": False, "reason": "exc channel not available"}
    # Use extended window — excitation may start ramping down before the oracle fires
    ext = _concat(pre_exc, exc)
    match = _is_falling(ext, min_fall_pct=30.0)
    return {"match": bool(match)}


def _check_exc_holds(exc: np.ndarray | None, thr: SignalThresholds) -> dict[str, Any]:
    if exc is None:
        return {"match": False, "reason": "exc channel not available"}
    high_exc = np.abs(exc) > thr.exc_energized_a
    return {"match": bool(np.mean(high_exc) > 0.8)}


def _check_rpm_holds(rpm: np.ndarray | None, thr: SignalThresholds) -> dict[str, Any]:
    if rpm is None:
        return {"match": False, "reason": "RPM channel not available"}
    match = _held_near(
        np.abs(rpm), thr.rpm_synchronous, tol=0.1 * thr.rpm_synchronous, min_frac=0.5
    )
    return {"match": bool(match), "rpm_mean": float(np.nanmean(np.abs(rpm)))}


def _check_rpm_direction_reversal(rpm: np.ndarray | None) -> dict[str, Any]:
    if rpm is None:
        return {"match": False, "reason": "RPM channel not available"}
    signs = np.sign(rpm[np.isfinite(rpm)])
    changed = bool(len(np.unique(signs)) > 1)
    return {"match": changed}


def _check_spiral_pressure_rise(spiral_p: np.ndarray | None) -> dict[str, Any]:
    if spiral_p is None:
        return {"match": None, "reason": "spiral pressure channel not available (optional)"}
    match = _is_rising(spiral_p, min_rise_pct=15.0)
    return {"match": bool(match)}


# ---------------------------------------------------------------------------
# Per-transition signature functions
# All accept: (allg_window, allg_pre, channel_names, thr)
# ---------------------------------------------------------------------------


def check_st_to_pu(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s = _signals(allg_window, channel_names)
    sfc_ok = _check_sfc_active(s["sfc"], thr, min_duration_s=60)
    rpm_ok = _check_rpm_ramp_up(s["rpm"], thr)
    exc_ok = {"match": s["exc"] is not None and _is_rising(s["exc"], min_rise_pct=30)}
    criteria = {"sfc_active": sfc_ok, "rpm_ramp_up": rpm_ok, "exc_ramp_up": exc_ok}
    match = sfc_ok["match"] and rpm_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_st_to_tu(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok   = _check_gate_ramp_open(s["gate"], thr, pre_gate=sp.get("gate"))
    spiral_ok = _check_spiral_pressure_rise(s["spiral_p"])
    rpm_ok    = _check_rpm_ramp_up(s["rpm"], thr)
    criteria  = {"gate_ramp_open": gate_ok, "spiral_rise": spiral_ok, "rpm_ramp_up": rpm_ok}
    match = gate_ok["match"] and rpm_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_st_to_ph(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s = _signals(allg_window, channel_names)
    sfc_ok  = _check_sfc_active(s["sfc"], thr, min_duration_s=60)
    rpm_ok  = _check_rpm_ramp_up(s["rpm"], thr)
    gate_ok = _check_gate_stays_closed(s["gate"], thr)
    criteria = {"sfc_active": sfc_ok, "rpm_ramp_up": rpm_ok, "gate_stays_closed": gate_ok}
    match = sfc_ok["match"] and rpm_ok["match"] and gate_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_ph_to_tu(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok   = _check_gate_ramp_open(s["gate"], thr, pre_gate=sp.get("gate"))
    spiral_ok = _check_spiral_pressure_rise(s["spiral_p"])
    criteria  = {"gate_ramp_open": gate_ok, "spiral_pressure_transient": spiral_ok}
    match = gate_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_ph_to_pu(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s = _signals(allg_window, channel_names)
    # PH entered from PU side: RPM stays negative (no direction reversal).
    # Primary indicator: power becomes strongly pumping (< power_pumping_mw threshold).
    reversal = _check_rpm_direction_reversal(s["rpm"])
    power_pumping = {"match": False, "reason": "power channel not available"}
    if s["power"] is not None:
        p_min = float(np.nanmin(s["power"]))
        power_pumping = {"match": p_min < thr.power_pumping_mw, "power_min_mw": p_min}
    criteria = {"rpm_direction_reversal": reversal, "power_pumping": power_pumping}
    match = reversal["match"] or power_pumping["match"]
    return {"match": bool(match), "criteria": criteria}


def check_pu_to_st(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok = _check_gate_closes(s["gate"], thr, pre_gate=sp.get("gate"))
    exc_ok  = _check_exc_ramp_down(s["exc"], thr, pre_exc=sp.get("exc"))
    rpm_ok  = _check_rpm_ramp_down(s["rpm"], thr, pre_rpm=sp.get("rpm"))
    criteria = {"gate_closes": gate_ok, "exc_ramp_down": exc_ok, "rpm_decay": rpm_ok}
    match = gate_ok["match"] and rpm_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_tu_to_st(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok = _check_gate_closes(s["gate"], thr, pre_gate=sp.get("gate"))
    exc_ok  = _check_exc_ramp_down(s["exc"], thr, pre_exc=sp.get("exc"))
    rpm_ok  = _check_rpm_ramp_down(s["rpm"], thr, pre_rpm=sp.get("rpm"))
    criteria = {"gate_closes": gate_ok, "exc_ramp_down": exc_ok, "rpm_decay": rpm_ok}
    # Gate closure is the primary criterion for TU→ST. The machine typically enters
    # PH briefly before stopping, so RPM/EXC decay may not complete within the oracle window.
    match = gate_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_pu_to_ph(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok = _check_gate_closes(s["gate"], thr, pre_gate=sp.get("gate"))
    rpm_ok  = _check_rpm_holds(s["rpm"], thr)
    exc_ok  = _check_exc_holds(s["exc"], thr)
    criteria = {"gate_closes": gate_ok, "rpm_holds": rpm_ok, "exc_holds": exc_ok}
    match = gate_ok["match"] and rpm_ok["match"]
    return {"match": bool(match), "criteria": criteria}


def check_tu_to_ph(allg_window, allg_pre, channel_names, thr: SignalThresholds) -> dict[str, Any]:
    s  = _signals(allg_window, channel_names)
    sp = _signals(allg_pre, channel_names) if allg_pre is not None and len(allg_pre) > 0 else {}
    gate_ok = _check_gate_closes(s["gate"], thr, pre_gate=sp.get("gate"))
    rpm_ok  = _check_rpm_holds(s["rpm"], thr)
    exc_ok  = _check_exc_holds(s["exc"], thr)
    criteria = {"gate_closes": gate_ok, "rpm_holds": rpm_ok, "exc_holds": exc_ok}
    match = gate_ok["match"] and rpm_ok["match"]
    return {"match": bool(match), "criteria": criteria}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

SIGNATURE_CHECKERS: dict[str, Any] = {
    "ST→PU": check_st_to_pu,
    "ST→TU": check_st_to_tu,
    "ST→PH": check_st_to_ph,
    "PH→TU": check_ph_to_tu,
    "PH→PU": check_ph_to_pu,
    "PU→ST": check_pu_to_st,
    "TU→ST": check_tu_to_st,
    "PU→PH": check_pu_to_ph,
    "TU→PH": check_tu_to_ph,
}

# Valid directed arcs in the operating graph
VALID_TRANSITIONS: dict[tuple[str, str], str] = {
    ("ST", "TU"): "ST→TU",
    ("TU", "ST"): "TU→ST",
    ("ST", "PU"): "ST→PU",
    ("PU", "ST"): "PU→ST",
    ("ST", "PH"): "ST→PH",
    ("PH", "TU"): "PH→TU",
    ("PH", "PU"): "PH→PU",
    ("PU", "PH"): "PU→PH",
    ("TU", "PH"): "TU→PH",
}

FORBIDDEN_DIRECT = {("PU", "TU"), ("TU", "PU")}
