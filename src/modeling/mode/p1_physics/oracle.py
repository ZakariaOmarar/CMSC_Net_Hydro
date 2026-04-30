"""Layer 1 Physics Oracle — deterministic per-second mode classification from Allg_M1.

Label vocabulary
----------------
ST          Standstill
TU          Turbine (generating)
PU          Pump (consuming)
PH          Phasenschieber / Synchronous condenser
TRANSITION  Between two identified steady states (confidence LOW)
UNKNOWN     No rule matched; data may be faulted or a rare edge case

Confidence levels
-----------------
HIGH    Rule matched AND all three primary signals stable for ≥ steady_window_s
MEDIUM  Rule matched BUT at least one signal not yet stable
LOW     No rule matched (TRANSITION or UNKNOWN)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .thresholds import (
    CHANNEL_ALIASES,
    SignalThresholds,
    extract_signal,
)

# Integer codes for labels (written to oracle_labels.npy as int8)
LABEL_CODE: dict[str, int] = {
    "ST": 0,
    "TU": 1,
    "PU": 2,
    "PH": 3,
    "TRANSITION": 4,
    "UNKNOWN": 5,
}
CODE_LABEL: dict[int, str] = {v: k for k, v in LABEL_CODE.items()}

CONFIDENCE_CODE: dict[str, int] = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
CODE_CONFIDENCE: dict[int, str] = {v: k for k, v in CONFIDENCE_CODE.items()}

STEADY_LABEL_CODES = frozenset({LABEL_CODE["ST"], LABEL_CODE["TU"],
                                 LABEL_CODE["PU"], LABEL_CODE["PH"]})


# ---------------------------------------------------------------------------
# Vectorized rolling helpers
# ---------------------------------------------------------------------------


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Compute causal rolling standard deviation via prefix sums.

    Returns (T,) float64. For t < window the std is computed over [0..t].
    """
    T = len(x)
    x64 = x.astype(np.float64)
    cs1 = np.zeros(T + 1, dtype=np.float64)
    cs2 = np.zeros(T + 1, dtype=np.float64)
    cs1[1:] = np.cumsum(x64)
    cs2[1:] = np.cumsum(x64 ** 2)

    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = t_arr - t0_arr + 1  # actual window size per timestep

    sum1 = cs1[t_arr + 1] - cs1[t0_arr]
    sum2 = cs2[t_arr + 1] - cs2[t0_arr]
    var = np.maximum(0.0, sum2 / n - (sum1 / n) ** 2)
    return np.sqrt(var)


def _debounce(labels: np.ndarray, window: int) -> np.ndarray:
    """A label flip must persist for `window` seconds to commit.

    Scans forward: if a new label appears but reverts within `window` steps,
    keep the previous label. Simple O(T) pass.
    """
    out = labels.copy()
    T = len(out)
    t = 0
    while t < T:
        if t == 0 or out[t] != out[t - 1]:
            # New label starting at t — check it persists for `window` steps
            end = min(t + window, T)
            run_label = out[t]
            run_end = t
            for j in range(t, end):
                if out[j] == run_label:
                    run_end = j
                else:
                    break
            run_len = run_end - t + 1
            if run_len < window and t > 0:
                # Short run — revert to previous label
                prev = out[t - 1]
                out[t:run_end + 1] = prev
        t += 1
    return out


# ---------------------------------------------------------------------------
# Core oracle
# ---------------------------------------------------------------------------


@dataclass
class OracleResult:
    labels: np.ndarray       # (T,) int8 — LABEL_CODE values
    confidence: np.ndarray   # (T,) int8 — CONFIDENCE_CODE values
    steady_mask: np.ndarray  # (T,) bool — HIGH-confidence steady timesteps


def run_oracle(
    allg: np.ndarray,
    channel_names: list[str],
    thresholds: SignalThresholds,
    *,
    steady_window_s: int = 30,
) -> OracleResult:
    """Classify every 1-Hz timestep using deterministic physics rules.

    Parameters
    ----------
    allg:
        (T, N_ch) float32 — Allg_M1 data decimated to 1 Hz.
    channel_names:
        Column names matching allg.
    thresholds:
        Derived thresholds from `derive_thresholds`.
    steady_window_s:
        Rolling window length (seconds) used for the STEADY stability test.
    """
    T = allg.shape[0]
    thr = thresholds

    def get(name: str) -> np.ndarray:
        sig = extract_signal(allg, channel_names, name)
        if sig is None:
            return np.full(T, np.nan, dtype=np.float64)
        return sig.astype(np.float64)

    # --- Extract primary signals -----------------------------------------
    v_grid  = get("v_grid")
    rpm     = get("rpm")
    gate    = get("gate")
    valve   = get("valve")
    power   = get("power")
    exc     = get("exc")

    # --- Boolean signal conditions ----------------------------------------
    grid_on  = v_grid > thr.v_grid_connected_kv
    spinning = np.abs(rpm) > thr.rpm_spinning
    gate_open   = gate > thr.gate_open_pct
    # "Gate not open" = gate hasn't crossed the opening threshold (valley + full hysteresis).
    # Using gate_open_pct here (not gate_closed_pct) so that noise around the valley
    # doesn't create a dead zone where neither ST/PH nor TU/PU fires.
    gate_closed = gate < thr.gate_open_pct
    valve_open   = valve > thr.valve_open_pct
    valve_closed = valve < thr.valve_closed_pct
    generating = power > thr.power_generating_mw
    pumping    = power < thr.power_pumping_mw
    windage    = np.abs(power) < thr.power_windage_mw
    energized  = exc > thr.exc_energized_a

    # Handle NaN channels gracefully (treat as unknown)
    def _safe(arr: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(arr), False, arr).astype(bool)

    grid_on   = _safe(grid_on)
    spinning  = _safe(spinning)
    gate_open = _safe(gate_open)
    gate_closed = _safe(gate_closed)
    valve_open  = _safe(valve_open)
    valve_closed = _safe(valve_closed)
    generating = _safe(generating)
    pumping    = _safe(pumping)
    windage    = _safe(windage)
    energized  = _safe(energized)

    # --- Apply classification rules (priority order) ---------------------
    raw = np.full(T, LABEL_CODE["UNKNOWN"], dtype=np.int8)

    # PH: spinning + grid + gate sealed + windage band
    ph_mask = spinning & grid_on & gate_closed & windage
    raw[ph_mask] = LABEL_CODE["PH"]

    # TU: spinning + grid + gate open + generating
    tu_mask = spinning & grid_on & gate_open & generating
    raw[tu_mask] = LABEL_CODE["TU"]

    # PU: spinning + grid + (gate open or valve open) + pumping
    pu_mask = spinning & grid_on & (gate_open | valve_open) & pumping
    raw[pu_mask] = LABEL_CODE["PU"]

    # ST: not spinning + gate closed (energized state not required — machine may be de-energized)
    st_mask = (~spinning) & gate_closed & (~grid_on)
    raw[st_mask] = LABEL_CODE["ST"]

    # --- Apply debounce --------------------------------------------------
    labels = _debounce(raw, thr.debounce_s).astype(np.int8)

    # --- TRANSITION: gaps between identified steady labels ---------------
    # Any UNKNOWN timestep flanked by two different steady labels gets TRANSITION
    is_steady = np.isin(labels, list(STEADY_LABEL_CODES))
    unknown_mask = labels == LABEL_CODE["UNKNOWN"]

    # Forward-fill and backward-fill to find the surrounding steady labels
    prev_steady = _ffill(labels, is_steady)
    next_steady = _bfill(labels, is_steady)

    transition_mask = (
        unknown_mask
        & (prev_steady >= 0)
        & (next_steady >= 0)
        & (prev_steady != next_steady)
    )
    labels[transition_mask] = LABEL_CODE["TRANSITION"]

    # --- STEADY stability mask (HIGH confidence) -------------------------
    gate_std  = _rolling_std(gate.clip(-1e6, 1e6), steady_window_s)
    rpm_std   = _rolling_std(rpm.clip(-1e6, 1e6),  steady_window_s)
    power_std = _rolling_std(power.clip(-1e6, 1e6), steady_window_s)

    # Replace NaN in std arrays with large value (not stable)
    gate_std  = np.where(np.isnan(gate_std),  1e9, gate_std)
    rpm_std   = np.where(np.isnan(rpm_std),   1e9, rpm_std)
    power_std = np.where(np.isnan(power_std), 1e9, power_std)

    is_stable = (
        (gate_std  < thr.gate_stable_pct)
        & (rpm_std   < thr.rpm_stable_rpm)
        & (power_std < thr.power_stable_mw)
    )

    # --- Confidence array ------------------------------------------------
    confidence = np.full(T, CONFIDENCE_CODE["LOW"], dtype=np.int8)
    rule_matched = np.isin(labels, [LABEL_CODE["ST"], LABEL_CODE["TU"],
                                     LABEL_CODE["PU"], LABEL_CODE["PH"]])
    confidence[rule_matched & ~is_stable] = CONFIDENCE_CODE["MEDIUM"]
    confidence[rule_matched & is_stable]  = CONFIDENCE_CODE["HIGH"]

    steady_mask = (confidence == CONFIDENCE_CODE["HIGH"]) & rule_matched

    return OracleResult(labels=labels, confidence=confidence, steady_mask=steady_mask)


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------


def _ffill(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Forward-fill the last valid label into positions where valid_mask is False.

    Returns an array of the same shape as labels. Positions before the first
    valid index get -1.
    """
    out = np.full_like(labels, -1, dtype=np.int8)
    last = -1
    for t in range(len(labels)):
        if valid_mask[t]:
            last = labels[t]
        out[t] = last
    return out


def _bfill(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Backward-fill the next valid label into positions where valid_mask is False."""
    out = np.full_like(labels, -1, dtype=np.int8)
    nxt = -1
    for t in range(len(labels) - 1, -1, -1):
        if valid_mask[t]:
            nxt = labels[t]
        out[t] = nxt
    return out
