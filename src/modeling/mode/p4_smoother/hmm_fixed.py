"""Layer 4 — 4-state fixed-emission Viterbi smoother.

Combines Layer 1 oracle confidence with Layer 3 LightGBM posteriors into a
single smoothed label sequence that enforces the physics topology from topology.py.

Key design: emissions are NOT learned via EM — they are computed analytically
from the oracle and LightGBM outputs. This makes the smoother deterministic,
interpretable, and fast (Viterbi runs in seconds for T=604,800).

NegBin duration correction is intentionally omitted in the first iteration.
Add HSMM machinery only if Viterbi shows visible artifacts (premature switching).

Emission model
--------------
oracle_confidence == HIGH  → one-hot on oracle label, weight 0.99
oracle_confidence == MEDIUM → 0.9 on oracle label, 0.1 uniform
oracle_confidence == LOW   → use LightGBM posterior, marginalized to 4 steady states
oracle absent              → use LightGBM posterior only
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .topology import LOG_TOPOLOGY, N_STATES, STATE_NAMES, IDX_TO_STATE, validate_sequence
from src.modeling.mode.p1_physics.oracle import CONFIDENCE_CODE, CODE_LABEL, LABEL_CODE
from src.modeling.mode.p3_rms.lgbm_classifier import ALL_CLASS_NAMES, STEADY_CLASS_NAMES

# Confidence weight per level
_CONF_WEIGHT = {
    CONFIDENCE_CODE["HIGH"]:   0.99,
    CONFIDENCE_CODE["MEDIUM"]: 0.90,
    CONFIDENCE_CODE["LOW"]:    0.00,   # use LightGBM instead
}

# Initial state probabilities (uninformative — machine may start in any mode)
_LOG_PI = np.log(np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float64))

# Transition matrix: slightly prefer staying in the same state (self-loop bias)
# Start from topology, then add self-loop bias and normalize
_SELF_LOOP_BIAS = 10.0  # relative weight of staying vs transitioning


def _build_log_trans() -> np.ndarray:
    """Build log-transition matrix with self-loop bias, topology-constrained."""
    A = np.array([
        [_SELF_LOOP_BIAS, 1, 1, 1],   # ST
        [1, _SELF_LOOP_BIAS, 0, 0],   # TU
        [1, 0, _SELF_LOOP_BIAS, 0],   # PU
        [1, 1, 1, _SELF_LOOP_BIAS],   # PH
    ], dtype=np.float64)
    # Apply topology mask
    from .topology import TOPOLOGY
    A = A * TOPOLOGY
    # Row-normalize
    row_sums = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(row_sums, 1e-300)
    return np.log(np.maximum(A, 1e-300))


LOG_TRANS = _build_log_trans()


# ---------------------------------------------------------------------------
# Emission computation
# ---------------------------------------------------------------------------


def _lgbm_to_steady_proba(lgbm_proba: np.ndarray) -> np.ndarray:
    """Marginalize LightGBM 13-class posteriors onto 4 steady states.

    Transition class probabilities are summed into their endpoint steady states
    (50/50 split between before and after).

    Parameters
    ----------
    lgbm_proba : (T, N_CLASSES) float32

    Returns
    -------
    (T, 4) float32 — rows sum to 1
    """
    T, n_lgbm_classes = lgbm_proba.shape
    out = np.zeros((T, 4), dtype=np.float64)

    # Direct steady-state columns
    for mode in STEADY_CLASS_NAMES:
        if mode in ALL_CLASS_NAMES:
            src_idx = ALL_CLASS_NAMES.index(mode)
            if src_idx >= n_lgbm_classes:
                continue  # class absent from training set
            dst_idx = STEADY_CLASS_NAMES.index(mode)
            out[:, dst_idx] += lgbm_proba[:, src_idx]

    # Transition columns: split evenly between before and after
    for trans_name in ALL_CLASS_NAMES:
        if "→" not in trans_name:
            continue
        src_idx = ALL_CLASS_NAMES.index(trans_name)
        if src_idx >= n_lgbm_classes:
            continue  # transition type absent from training set
        before_str, after_str = trans_name.split("→")
        for endpoint in [before_str.strip(), after_str.strip()]:
            if endpoint in STEADY_CLASS_NAMES:
                dst_idx = STEADY_CLASS_NAMES.index(endpoint)
                out[:, dst_idx] += 0.5 * lgbm_proba[:, src_idx]

    # Normalize
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / np.maximum(row_sums, 1e-300)
    return out.astype(np.float32)


def compute_log_emission(
    oracle_labels: np.ndarray,
    oracle_confidence: np.ndarray,
    lgbm_proba: np.ndarray | None = None,
) -> np.ndarray:
    """Compute (T, 4) log-emission matrix.

    Parameters
    ----------
    oracle_labels : (T,) int8 — LABEL_CODE values
    oracle_confidence : (T,) int8 — CONFIDENCE_CODE values
    lgbm_proba : (T, N_CLASSES) float32 or None

    Returns
    -------
    log_emit : (T, 4) float64 — log p(observation | state k) for k in {ST,TU,PU,PH}
    """
    T = len(oracle_labels)
    high_code   = CONFIDENCE_CODE["HIGH"]
    medium_code = CONFIDENCE_CODE["MEDIUM"]
    low_code    = CONFIDENCE_CODE["LOW"]

    # Start from uniform
    log_emit = np.full((T, N_STATES), np.log(1.0 / N_STATES), dtype=np.float64)

    # Build steady-state probability array for LightGBM rows
    if lgbm_proba is not None:
        lgbm_steady = _lgbm_to_steady_proba(lgbm_proba).astype(np.float64)

    for t in range(T):
        conf = int(oracle_confidence[t])
        lbl  = int(oracle_labels[t])

        if conf == high_code and lbl in (LABEL_CODE["ST"], LABEL_CODE["TU"],
                                          LABEL_CODE["PU"], LABEL_CODE["PH"]):
            # High confidence: one-hot with weight 0.99
            proba = np.full(N_STATES, 0.01 / (N_STATES - 1), dtype=np.float64)
            # Map LABEL_CODE to state index (they happen to be the same: ST=0,TU=1,PU=2,PH=3)
            state_idx = lbl  # ST=0, TU=1, PU=2, PH=3
            proba[state_idx] = 0.99
            log_emit[t] = np.log(proba)

        elif conf == medium_code and lbl in (LABEL_CODE["ST"], LABEL_CODE["TU"],
                                              LABEL_CODE["PU"], LABEL_CODE["PH"]):
            # Medium confidence: 0.9 on oracle, 0.1 spread
            proba = np.full(N_STATES, 0.1 / (N_STATES - 1), dtype=np.float64)
            state_idx = lbl
            proba[state_idx] = 0.9
            if lgbm_proba is not None:
                # Blend with LightGBM (0.9 oracle, 0.1 LightGBM)
                proba = 0.9 * proba + 0.1 * lgbm_steady[t]
            log_emit[t] = np.log(np.maximum(proba, 1e-300))

        else:
            # Low confidence or TRANSITION/UNKNOWN: use LightGBM only
            if lgbm_proba is not None:
                log_emit[t] = np.log(np.maximum(lgbm_steady[t], 1e-300))
            # else: stays at uniform

    return log_emit


# ---------------------------------------------------------------------------
# Vectorized Viterbi
# ---------------------------------------------------------------------------


def viterbi_decode(log_emit: np.ndarray) -> np.ndarray:
    """Standard Viterbi decoding with topology-constrained log-transition matrix.

    Parameters
    ----------
    log_emit : (T, N_STATES) float64

    Returns
    -------
    state_sequence : (T,) int32
    """
    T, N = log_emit.shape
    V = np.full((T, N), -np.inf, dtype=np.float64)
    psi = np.zeros((T, N), dtype=np.int32)

    V[0] = _LOG_PI + log_emit[0]

    for t in range(1, T):
        # trans_scores[i, j] = V[t-1, i] + LOG_TRANS[i, j]
        trans_scores = V[t - 1, :, None] + LOG_TRANS  # (N, N)
        best_prev = np.argmax(trans_scores, axis=0)    # (N,)
        V[t] = trans_scores[best_prev, np.arange(N)] + log_emit[t]
        psi[t] = best_prev

    # Backtrack
    seq = np.empty(T, dtype=np.int32)
    seq[T - 1] = int(np.argmax(V[T - 1]))
    for t in range(T - 2, -1, -1):
        seq[t] = psi[t + 1, seq[t + 1]]

    return seq


# ---------------------------------------------------------------------------
# Segment extraction and timeline
# ---------------------------------------------------------------------------


@dataclass
class ModeSegment:
    t_start_s: int
    t_end_s: int
    duration_s: int
    state_id: int
    label: str
    micro_event: bool


def extract_mode_segments(
    state_seq: np.ndarray,
    *,
    micro_event_threshold_s: int = 60,
) -> list[ModeSegment]:
    """Convert state sequence to list of contiguous mode segments."""
    segments = []
    T = len(state_seq)
    t = 0
    while t < T:
        s = int(state_seq[t])
        t_end = t
        while t_end + 1 < T and int(state_seq[t_end + 1]) == s:
            t_end += 1
        duration = t_end - t + 1
        segments.append(ModeSegment(
            t_start_s=t,
            t_end_s=t_end,
            duration_s=duration,
            state_id=s,
            label=IDX_TO_STATE[s],
            micro_event=duration < micro_event_threshold_s,
        ))
        t = t_end + 1
    return segments


# ---------------------------------------------------------------------------
# Mode timeline (JSON-serializable, same schema as DTSS output)
# ---------------------------------------------------------------------------


def build_mode_timeline(
    segments: list[ModeSegment],
    transition_segments: list,   # from p2_transitions.typing
    timestamps_ns: np.ndarray,
    *,
    layer2_typed: bool = True,
) -> list[dict[str, Any]]:
    """Build mode_timeline.json events from smoothed segments + L2 transition types.

    For each steady-state segment, label = state name (ST, TU, PU, PH).
    For each transition interval (between segments with different labels),
    look up the typed transition from Layer 2 if available.
    """
    timeline = []

    for seg in segments:
        t0_ns = int(timestamps_ns[seg.t_start_s]) if seg.t_start_s < len(timestamps_ns) else 0
        t1_ns = int(timestamps_ns[seg.t_end_s])   if seg.t_end_s < len(timestamps_ns) else 0

        timeline.append({
            "t_start_ns": t0_ns,
            "t_end_ns":   t1_ns,
            "t_start_s":  seg.t_start_s,
            "t_end_s":    seg.t_end_s,
            "duration_s": seg.duration_s,
            "state_id":   seg.state_id,
            "label":      seg.label,
            "micro_event": seg.micro_event,
        })

    # Annotate with Layer 2 transition types where available
    if layer2_typed and transition_segments:
        trans_map: dict[int, str] = {}
        for tseg in transition_segments:
            for t in range(tseg.t_start_s, tseg.t_end_s + 1):
                trans_map[t] = tseg.transition_type

        for event in timeline:
            if event["micro_event"]:
                # Look up transition type from Layer 2 for micro-events
                t_mid = (event["t_start_s"] + event["t_end_s"]) // 2
                trans_type = trans_map.get(t_mid)
                if trans_type:
                    event["transition_type"] = trans_type

    return timeline


def save_mode_timeline(timeline: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(timeline, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main smoother entry point
# ---------------------------------------------------------------------------


def smooth(
    oracle_labels: np.ndarray,
    oracle_confidence: np.ndarray,
    lgbm_proba: np.ndarray | None,
    *,
    micro_event_threshold_s: int = 60,
) -> tuple[np.ndarray, list[ModeSegment]]:
    """Run the full Layer 4 smoother.

    Returns
    -------
    state_seq : (T,) int32 — smoothed state indices
    segments : list[ModeSegment]
    """
    log_emit = compute_log_emission(oracle_labels, oracle_confidence, lgbm_proba)
    state_seq = viterbi_decode(log_emit)

    violations = validate_sequence(state_seq)
    if violations:
        raise RuntimeError(
            f"Topology violation in Viterbi output: {len(violations)} forbidden transitions. "
            f"First: t={violations[0][0]}, {IDX_TO_STATE[violations[0][1]]}→{IDX_TO_STATE[violations[0][2]]}"
        )

    segments = extract_mode_segments(state_seq, micro_event_threshold_s=micro_event_threshold_s)
    return state_seq, segments
