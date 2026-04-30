"""Layer 5 — Anomaly event generation and reporting.

Converts per-second alert_level arrays into discrete anomaly events with
timestamps, severity, mode context, and optional sub-mode context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.modeling.mode.p4_smoother.topology import IDX_TO_STATE
from .sub_modes import sub_mode_name


@dataclass
class AnomalyEvent:
    t_start_s: int
    t_end_s: int
    t_start_ns: int
    t_end_ns: int
    duration_s: int
    severity: str            # "watch" or "alert"
    mode: str                # ST, TU, PU, PH
    sub_mode: str            # TU-deep-part-load etc., or "n/a"
    peak_z_score: float
    mean_z_score: float


def detect_anomaly_events(
    z_scores: np.ndarray,
    alert_level: np.ndarray,
    smoothed_labels: np.ndarray,
    timestamps_ns: np.ndarray,
    sub_mode_labels: np.ndarray | None = None,
    *,
    min_event_duration_s: int = 5,
    merge_gap_s: int = 30,
) -> list[AnomalyEvent]:
    """Convert alert_level array to a list of anomaly events.

    Parameters
    ----------
    z_scores : (T,) float32
    alert_level : (T,) int8 — 0=normal, 1=watch, 2=alert
    smoothed_labels : (T,) int32
    timestamps_ns : (T,) int64
    sub_mode_labels : (T,) int8 or None
    min_event_duration_s : discard events shorter than this
    merge_gap_s : merge consecutive events within this gap
    """
    T = len(alert_level)
    events: list[AnomalyEvent] = []

    # Find contiguous alert runs (level >= 1)
    in_event = False
    ev_start = 0
    ev_level = 0

    def _finalize(t_start: int, t_end: int, level: int) -> AnomalyEvent | None:
        duration = t_end - t_start + 1
        if duration < min_event_duration_s:
            return None
        z_window = z_scores[t_start:t_end + 1]
        z_window = z_window[np.isfinite(z_window)]
        if len(z_window) == 0:
            return None
        mode_idx = int(np.bincount(smoothed_labels[t_start:t_end + 1].clip(0, 3),
                                    minlength=4).argmax())
        mode_str = IDX_TO_STATE.get(mode_idx, "UNKNOWN")
        sub_str = "n/a"
        if sub_mode_labels is not None:
            sub_window = sub_mode_labels[t_start:t_end + 1]
            sub_counts = np.bincount(np.maximum(sub_window, 0), minlength=4)
            dominant_sub = int(sub_counts.argmax()) if sub_window.max() >= 0 else -1
            sub_str = sub_mode_name(dominant_sub)

        t_start_ns = int(timestamps_ns[t_start]) if t_start < len(timestamps_ns) else 0
        t_end_ns   = int(timestamps_ns[t_end])   if t_end < len(timestamps_ns) else 0
        return AnomalyEvent(
            t_start_s=t_start,
            t_end_s=t_end,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            duration_s=duration,
            severity="alert" if level >= 2 else "watch",
            mode=mode_str,
            sub_mode=sub_str,
            peak_z_score=float(np.max(z_window)),
            mean_z_score=float(np.mean(z_window)),
        )

    for t in range(T):
        lv = int(alert_level[t])
        if lv >= 1 and not in_event:
            in_event = True
            ev_start = t
            ev_level = lv
        elif lv >= 1 and in_event:
            ev_level = max(ev_level, lv)
        elif lv == 0 and in_event:
            # Check merge: if next alert within merge_gap_s, continue
            t_next_alert = -1
            for tt in range(t, min(t + merge_gap_s, T)):
                if alert_level[tt] >= 1:
                    t_next_alert = tt
                    break
            if t_next_alert == -1:
                ev = _finalize(ev_start, t - 1, ev_level)
                if ev is not None:
                    events.append(ev)
                in_event = False

    if in_event:
        ev = _finalize(ev_start, T - 1, ev_level)
        if ev is not None:
            events.append(ev)

    return events


def save_anomaly_events(events: list[AnomalyEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in events], f, indent=2, default=str)


def anomaly_summary(
    events: list[AnomalyEvent],
    total_mode_seconds: dict[str, int],
) -> dict[str, Any]:
    """Compute per-mode anomaly rate and severity breakdown."""
    summary: dict[str, Any] = {
        "total_events": len(events),
        "watch_events": sum(1 for e in events if e.severity == "watch"),
        "alert_events": sum(1 for e in events if e.severity == "alert"),
        "per_mode": {},
    }
    for mode in ["ST", "TU", "PU", "PH"]:
        mode_events = [e for e in events if e.mode == mode]
        total_alert_s = sum(e.duration_s for e in mode_events if e.severity == "alert")
        total_s = total_mode_seconds.get(mode, 1)
        summary["per_mode"][mode] = {
            "n_events": len(mode_events),
            "total_alert_s": total_alert_s,
            "alert_rate_pct": round(100.0 * total_alert_s / max(total_s, 1), 4),
        }
    return summary
