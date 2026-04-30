"""Single source of truth for the 4-state operating mode topology.

States: ST(0), TU(1), PU(2), PH(3)

Valid directed arcs (physics-constrained):
  ST → ST, TU, PU, PH    (hub state — can start any mode)
  TU → TU, ST            (turbine → standstill only)
  PU → PU, ST            (pump → standstill only)
  PH → PH, ST, TU, PU   (condenser can transition to any mode)

Forbidden direct arcs (must route through ST or PH):
  PU → TU
  TU → PU

Note: PH → TU and PH → PU are allowed (common in hydro operation).
"""

from __future__ import annotations

import numpy as np

STATE_NAMES = ["ST", "TU", "PU", "PH"]
N_STATES = 4

STATE_TO_IDX = {name: i for i, name in enumerate(STATE_NAMES)}
IDX_TO_STATE = {i: name for i, name in enumerate(STATE_NAMES)}

# Adjacency matrix: TOPOLOGY[i, j] = 1 if i→j is a valid arc
# Row = from-state, Column = to-state
#         ST  TU  PU  PH
TOPOLOGY = np.array([
    [1,   1,  1,  1],  # ST → any
    [1,   1,  0,  0],  # TU → ST, TU  (NOT PU, NOT PH directly)
    [1,   0,  1,  0],  # PU → ST, PU  (NOT TU, NOT PH directly)
    [1,   1,  1,  1],  # PH → any
], dtype=np.float64)

# Log-probability mask: 0.0 for allowed, -inf for forbidden
LOG_TOPOLOGY = np.where(TOPOLOGY > 0, 0.0, -np.inf)


def validate_sequence(state_seq: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (timestep, from_state, to_state) for any topology violations."""
    violations = []
    for t in range(1, len(state_seq)):
        i, j = int(state_seq[t - 1]), int(state_seq[t])
        if TOPOLOGY[i, j] == 0:
            violations.append((t, i, j))
    return violations


def steady_state_arc_str(from_state: int, to_state: int) -> str:
    """Human-readable arc label, e.g. 'TU→ST'."""
    return f"{IDX_TO_STATE[from_state]}→{IDX_TO_STATE[to_state]}"
