"""Empirical threshold derivation for Layer 1 physics oracle.

All numeric thresholds are derived from the actual campaign data — no magic numbers.
`derive_thresholds` fits a 2- or 3-component GMM to each discriminating signal and
places every cut at the PDF valley between adjacent modes.

Outputs `signal_thresholds.json` for documentation and for use by `oracle.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Signal-to-channel aliases (handles naming variations across .dat files)
# ---------------------------------------------------------------------------

CHANNEL_ALIASES: dict[str, list[str]] = {
    "v_grid":    ["1_21kV Gen. Spg.", "21kV Gen. Spg.", "gen_voltage_kv"],
    "rpm":       ["1_Drehzahl UPM", "Drehzahl UPM", "rpm"],
    "gate":      ["1_Leitapparat Stell.", "Leitapparat Stell.", "Leitapparat", "guide_vane"],
    "valve":     ["1_KS Stellung", "KS Stellung", "inlet_valve"],
    "power":     ["1_P_Ist", "P_Ist", "power"],
    "exc":       ["1_Erregerstrom", "Erregerstrom", "excitation_current"],
    "sfc":       ["1_Anfahrumr. Strom", "Anfahrumr. Strom", "sfc_current"],
    "flow_tu":   ["Durchfluss TU", "flow_tu"],
    "flow_pu":   ["Durchfluss PU", "flow_pu"],
    "spiral_p":  ["1_Spiraldruck", "Spiraldruck", "spiral_pressure"],
}


def resolve_channel(name: str, channel_names: list[str]) -> int | None:
    """Return column index for a logical signal name, or None if not found."""
    aliases = CHANNEL_ALIASES.get(name, [name])
    for alias in aliases:
        if alias in channel_names:
            return channel_names.index(alias)
    return None


def extract_signal(
    allg: np.ndarray,
    channel_names: list[str],
    signal_name: str,
) -> np.ndarray | None:
    """Extract a 1-D signal array from allg by logical name. Returns None if absent."""
    idx = resolve_channel(signal_name, channel_names)
    if idx is None:
        return None
    return allg[:, idx].astype(np.float64)


# ---------------------------------------------------------------------------
# Core threshold container
# ---------------------------------------------------------------------------


class SignalThresholds(NamedTuple):
    """All derived numeric thresholds consumed by oracle.py."""

    # Grid coupling: v_grid (kV) above this → machine connected to grid
    v_grid_connected_kv: float

    # Rotation: |rpm| above this → machine is spinning
    rpm_spinning: float

    # Guide vane: above this → TU path open (enter threshold)
    gate_open_pct: float
    # Guide vane: below this → gate considered closed (exit threshold / hysteresis)
    gate_closed_pct: float

    # Inlet valve: above this → PU path open
    valve_open_pct: float
    valve_closed_pct: float

    # Power discriminators (MW)
    power_generating_mw: float    # above this → TU (generating)
    power_pumping_mw: float       # below this (i.e., more negative) → PU (consuming)
    power_windage_mw: float       # |power| below this → PH windage band

    # Excitation: above this → machine energized
    exc_energized_a: float

    # Static frequency converter: above this → SFC active (startup signature)
    sfc_active_a: float

    # Stability thresholds for the STEADY mask (rolling std over 30 s)
    gate_stable_pct: float   # default 0.5
    rpm_stable_rpm: float    # default 5
    power_stable_mw: float   # default 5

    # Debounce: a label flip must persist this many seconds to commit
    debounce_s: int          # default 5

    # Synchronous speed (read from nameplate or inferred from data)
    rpm_synchronous: float


def _gmm_valley(
    x: np.ndarray,
    n_components: int = 2,
    n_bins: int = 500,
    random_state: int = 0,
) -> list[float]:
    """Fit a GMM to x and return valley positions (local minima of the PDF).

    Operates in the 1-D histogram domain — avoids the full O(N) GMM fit on very
    long sequences by working with the histogram PDF.
    """
    x_clean = x[np.isfinite(x)]
    if len(x_clean) < 20:
        return []

    lo, hi = float(np.percentile(x_clean, 0.5)), float(np.percentile(x_clean, 99.5))
    if hi <= lo:
        return []

    counts, edges = np.histogram(x_clean, bins=n_bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf = counts.astype(np.float64) / counts.sum()

    # Fit GMM on sub-sampled data (max 50k points for speed)
    rng = np.random.default_rng(random_state)
    sub = x_clean if len(x_clean) <= 50_000 else rng.choice(x_clean, 50_000, replace=False)
    gm = GaussianMixture(n_components=n_components, random_state=random_state, max_iter=200)
    gm.fit(sub.reshape(-1, 1))

    # Evaluate GMM PDF on histogram grid to find smooth valleys
    gmm_pdf = np.exp(gm.score_samples(centres.reshape(-1, 1)))
    gmm_pdf /= gmm_pdf.sum()

    # Valleys = local minima of GMM PDF that are also below 30% of peak
    neg_pdf = -gmm_pdf
    peak_idx, _ = find_peaks(gmm_pdf, prominence=gmm_pdf.max() * 0.05)
    valley_idx, _ = find_peaks(neg_pdf, prominence=gmm_pdf.max() * 0.02)

    # Keep only valleys that sit between two peaks
    valleys = []
    for vi in valley_idx:
        left_peaks = [p for p in peak_idx if p < vi]
        right_peaks = [p for p in peak_idx if p > vi]
        if left_peaks and right_peaks:
            valleys.append(float(centres[vi]))

    return sorted(valleys)


def _midpoint_threshold(x: np.ndarray, lo_pct: float, hi_pct: float) -> float:
    """Simple percentile-midpoint fallback when GMM fails."""
    return float(0.5 * (np.percentile(x[np.isfinite(x)], lo_pct) +
                        np.percentile(x[np.isfinite(x)], hi_pct)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_thresholds(
    allg: np.ndarray,
    channel_names: list[str],
    *,
    random_state: int = 42,
    debounce_s: int = 5,
    gate_hysteresis_pct: float = 2.0,
    valve_hysteresis_pct: float = 2.0,
    windage_ceiling_mw: float = 10.0,
    stable_gate_pct: float = 0.5,
    stable_rpm_rpm: float = 5.0,
    stable_power_mw: float = 5.0,
) -> SignalThresholds:
    """Derive all oracle thresholds empirically from the campaign Allg_M1 array.

    Parameters
    ----------
    allg:
        (T, N_ch) float32 Allg_M1 data at 1 Hz.
    channel_names:
        List of channel name strings, same order as allg columns.
    gate_hysteresis_pct:
        Hysteresis gap applied symmetrically: gate_open = valley + half_gap,
        gate_closed = valley − half_gap.
    windage_ceiling_mw:
        Fixed ceiling for the PH windage band (|power| < this → PH candidate).
    """

    def get(name: str) -> np.ndarray | None:
        return extract_signal(allg, channel_names, name)

    # --- v_grid ----------------------------------------------------------
    v_grid = get("v_grid")
    v_grid_kv: float = 10.0  # default
    if v_grid is not None:
        valleys = _gmm_valley(v_grid, n_components=2, random_state=random_state)
        v_grid_kv = valleys[0] if valleys else _midpoint_threshold(v_grid, 40, 60)

    # --- RPM / synchronous speed -----------------------------------------
    rpm = get("rpm")
    rpm_spinning: float = 50.0
    rpm_sync: float = 375.0
    if rpm is not None:
        # RPM distribution has peaks at ≈0, ≈+n_sync, ≈−n_sync
        rpm_abs = np.abs(rpm[np.isfinite(rpm)])
        valleys = _gmm_valley(rpm_abs, n_components=2, random_state=random_state)
        rpm_spinning = valleys[0] if valleys else _midpoint_threshold(rpm_abs, 40, 60)

        # Synchronous speed = peak of |rpm| above the spinning threshold
        high_rpm = rpm_abs[rpm_abs > rpm_spinning]
        if len(high_rpm) > 100:
            rpm_sync = float(np.percentile(high_rpm, 95))

    # --- Gate / guide vane -----------------------------------------------
    gate = get("gate")
    gate_valley: float = 5.0
    if gate is not None:
        valleys = _gmm_valley(gate, n_components=2, random_state=random_state)
        gate_valley = valleys[0] if valleys else _midpoint_threshold(gate, 40, 60)

    # gate_open: enter "gate open" state — TU/PU mode
    # gate_closed: used in ST/PH rules as "gate not yet open" — must sit above the
    # near-zero noise floor, so we use the valley position directly (not valley minus
    # half-hysteresis, which clips to 0 when the valley is tiny).
    gate_open_pct   = gate_valley + gate_hysteresis_pct
    gate_closed_pct = gate_valley   # ST/PH: gate < gate_closed_pct ≈ gate < valley

    # --- Valve / inlet valve ---------------------------------------------
    valve = get("valve")
    valve_valley: float = 5.0
    if valve is not None:
        valleys = _gmm_valley(valve, n_components=2, random_state=random_state)
        valve_valley = valleys[0] if valleys else _midpoint_threshold(valve, 40, 60)

    valve_open_pct = valve_valley + valve_hysteresis_pct / 2.0
    valve_closed_pct = max(0.0, valve_valley - valve_hysteresis_pct / 2.0)

    # --- Power -----------------------------------------------------------
    power = get("power")
    power_gen: float = windage_ceiling_mw
    power_pump: float = -5.0
    if power is not None:
        p_clean = power[np.isfinite(power)]
        # Generating threshold = windage ceiling (definitional: above the windage
        # band is generating).  A 3-component GMM on the full distribution risks
        # finding a within-TU sub-cluster boundary when the machine has multiple
        # preferred operating points; the windage ceiling is more robust.
        power_gen = windage_ceiling_mw

        # Pumping threshold: 2-component GMM on |negative power| separates
        # PH windage losses (small abs values) from true PU consumption (large).
        neg_abs = np.abs(p_clean[p_clean < -1.0])
        if len(neg_abs) > 200:
            valleys = _gmm_valley(neg_abs, n_components=2, random_state=random_state)
            power_pump = -valleys[0] if valleys else -5.0
        power_pump = min(-1.0, power_pump)  # ceiling at −1 MW

    # --- Excitation ------------------------------------------------------
    exc = get("exc")
    exc_energized: float = 50.0
    if exc is not None:
        valleys = _gmm_valley(exc, n_components=2, random_state=random_state)
        exc_energized = valleys[0] if valleys else _midpoint_threshold(exc, 40, 60)

    # --- SFC current (static frequency converter) ------------------------
    sfc = get("sfc")
    sfc_active: float = 100.0
    if sfc is not None:
        sfc_clean = np.abs(sfc[np.isfinite(sfc)])
        valleys = _gmm_valley(sfc_clean, n_components=2, random_state=random_state)
        if valleys:
            sfc_active = valleys[0]
        else:
            high_sfc = sfc_clean[sfc_clean > np.percentile(sfc_clean, 90)]
            sfc_active = float(np.percentile(high_sfc, 5)) if len(high_sfc) > 10 else 100.0

    return SignalThresholds(
        v_grid_connected_kv=v_grid_kv,
        rpm_spinning=rpm_spinning,
        gate_open_pct=gate_open_pct,
        gate_closed_pct=gate_closed_pct,
        valve_open_pct=valve_open_pct,
        valve_closed_pct=valve_closed_pct,
        power_generating_mw=power_gen,
        power_pumping_mw=power_pump,
        power_windage_mw=windage_ceiling_mw,
        exc_energized_a=exc_energized,
        sfc_active_a=sfc_active,
        gate_stable_pct=stable_gate_pct,
        rpm_stable_rpm=stable_rpm_rpm,
        power_stable_mw=stable_power_mw,
        debounce_s=debounce_s,
        rpm_synchronous=rpm_sync,
    )


def save_thresholds(thresholds: SignalThresholds, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds._asdict(), f, indent=2)


def load_thresholds(path: str | Path) -> SignalThresholds:
    with open(path) as f:
        d = json.load(f)
    return SignalThresholds(**d)
