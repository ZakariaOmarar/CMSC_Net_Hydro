"""Sliding-window anomaly detection with event extraction.

Motivation (supervisor discussion, 2026-05-13)
----------------------------------------------

The current V3 pipeline scores one anomaly value per 2 s window with
1 s stride (50 % overlap), producing one score every 1 s.  Two
problems with this:

  1. **Temporal precision is bounded by the stride.**  An anomaly
     shorter than 1 s may live entirely inside a single window's
     2 s span; the per-window score reflects the *averaged* energy
     over 2 s, diluting short impulses.  At the stride boundary, an
     impulse spanning two adjacent windows contributes to both
     scores, broadening the apparent event.
  2. **No event-level annotation.**  We report per-window alert
     rates; we do not report **when** an anomaly starts, **how long**
     it lasts, or **how many distinct events** occurred in a
     recording.  These are exactly the deployment-relevant facts
     (Chapter 7 streaming-inference subsection).

This module addresses both with an **inference-time sliding window**
of configurable stride (much shorter than training-time 1 s) plus
**hysteresis-thresholded event extraction**.

Design choices
--------------

* **Training-time stride remains 1 s, inference-time stride is
  configurable.**  V3 is trained on overlapping 2 s / 1 s windows
  (good statistical regularity, well-mixed batches).  At inference
  time we slide with a 100–250 ms stride to localise events to ~ 1/4
  of the training-time stride.  This decoupling is standard in
  acoustic event detection (DCASE Task 4 baselines all use coarser
  training cadence than inference).

* **Hysteresis thresholds** (Schmitt 1938).  Entering the alert
  state requires score > `high_threshold`; exiting requires score
  < `low_threshold` < `high_threshold`.  This prevents an event
  from being fragmented into many short sub-events by noise around
  a single threshold.  Defaults: `high = p95` (V3's headline
  threshold), `low = p90` (slightly looser; events stay open
  through small dips).

* **Minimum event duration**.  Events shorter than
  `min_duration_s` are discarded as alert-spike noise.  Default
  100 ms — shorter than any realistic mechanical event on a 375 rpm
  machine (one shaft rev = 160 ms).

* **Per-event statistics**.  Each event carries
  ``(t_start_s, t_end_s, duration_s, peak_score, mean_score,
  n_windows, peak_offset_s_in_event)``.  Chapter 6 reports these
  per recording and per cohort.

Citations
---------

* Schmitt, O. H. (1938). "A thermionic trigger." *J. Sci. Instrum.*
  15(1) — hysteresis thresholding.
* DCASE Task 4 baselines (Turpault et al., 2019; Serizel et al.,
  2020) — coarse-training / fine-inference cadence + event
  extraction methodology in acoustic event detection.
* Khamaisi et al. 2025 §3.5 — qualitative discussion of the
  ROW II 5-second transient that motivated this thesis's
  context-aware design; the same transient is a natural event
  for the methodology here to localise post-deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyEvent:
    """A single detected anomalous event in a recording's score timeline.

    Attributes:
      t_start_s: start time (s) — the centre time of the first window
        that crossed `high_threshold`.
      t_end_s: end time (s) — the centre time of the last window
        whose score remained above `low_threshold`, plus half the
        window duration to include the window's right edge.
      duration_s: ``t_end_s - t_start_s``.
      peak_score: max anomaly score within the event.
      peak_t_s: time (s) of the peak score window's centre.
      mean_score: mean anomaly score across all event windows.
      n_windows: number of contiguous sliding windows in the event.
      recording_id: which recording this event came from.
      dataset_id: which dataset (D1 / D2 / D3 / D4 / illwerke).
    """

    t_start_s: float
    t_end_s: float
    duration_s: float
    peak_score: float
    peak_t_s: float
    mean_score: float
    n_windows: int
    recording_id: str
    dataset_id: str


def detect_events_from_score_timeline(
    scores: np.ndarray,
    times_s: np.ndarray,
    *,
    high_threshold: float,
    low_threshold: float | None = None,
    min_duration_s: float = 0.1,
    max_gap_windows: int = 0,
    recording_id: str = "",
    dataset_id: str = "",
    window_seconds: float = 2.0,
) -> list[AnomalyEvent]:
    """Extract anomaly events from a per-window score timeline.

    Args:
      scores: ``(N,)`` per-window V3 anomaly score (`-log p(x | c)`).
      times_s: ``(N,)`` window-centre times in seconds, monotonically
        increasing.  The stride is inferred from ``times_s[1] -
        times_s[0]``.
      high_threshold: enter-alert threshold (typically the per-cluster
        p95 of healthy `c_t`).
      low_threshold: exit-alert threshold; defaults to
        `high_threshold` (no hysteresis).  For Chapter 6 we set
        `low = p90` of the same per-cluster healthy distribution.
      min_duration_s: minimum event duration; shorter alert runs are
        discarded as noise.
      max_gap_windows: number of below-low-threshold windows to
        tolerate inside an event before splitting.  Default 0 means
        any dip below `low_threshold` ends the event.
      recording_id / dataset_id: passthrough metadata for the
        returned `AnomalyEvent` objects.
      window_seconds: training-time window duration, used to compute
        the event's right-edge time.

    Returns:
      A list of `AnomalyEvent` objects, ordered by `t_start_s`.
      Empty when no event in the timeline meets the criteria.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    times_s = np.asarray(times_s, dtype=np.float64).ravel()
    if scores.shape != times_s.shape:
        raise ValueError("scores and times_s must have the same shape")
    if low_threshold is None:
        low_threshold = float(high_threshold)
    if low_threshold > high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be ≤ high_threshold "
            f"({high_threshold}) for hysteresis to make sense"
        )
    if scores.size < 2:
        return []

    events: list[AnomalyEvent] = []
    in_event = False
    event_start_i = -1
    gap_count = 0

    def _close_event(end_i: int) -> None:
        nonlocal in_event, event_start_i
        if event_start_i < 0:
            in_event = False
            return
        idx = np.arange(event_start_i, end_i + 1)
        sub_scores = scores[idx]
        sub_times = times_s[idx]
        peak_i = int(np.argmax(sub_scores))
        # Event interval = union of all alert-window coverages, where
        # each window centre c covers [c − window/2, c + window/2].
        # This way a single-window event reports duration = window_s
        # (the physical span of the 2 s window centred at peak_t),
        # not half.  Reviewer-facing semantics: "the anomaly was
        # active at some point during [t_start, t_end]".
        t_start = float(sub_times[0] - window_seconds / 2.0)
        t_end = float(sub_times[-1] + window_seconds / 2.0)
        duration = float(t_end - t_start)
        if duration >= min_duration_s:
            events.append(
                AnomalyEvent(
                    t_start_s=t_start,
                    t_end_s=t_end,
                    duration_s=duration,
                    peak_score=float(sub_scores[peak_i]),
                    peak_t_s=float(sub_times[peak_i]),
                    mean_score=float(sub_scores.mean()),
                    n_windows=int(sub_scores.size),
                    recording_id=recording_id,
                    dataset_id=dataset_id,
                )
            )
        in_event = False
        event_start_i = -1

    for i, s in enumerate(scores):
        if in_event:
            if s >= low_threshold:
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap_windows:
                    _close_event(i - 1 - gap_count + 1)  # = i - gap_count
                    gap_count = 0
        else:
            if s > high_threshold:
                in_event = True
                event_start_i = i
                gap_count = 0
    if in_event:
        _close_event(int(scores.size - 1))

    return events


def summarise_events(events: list[AnomalyEvent]) -> dict:
    """Per-cohort event statistics for Chapter 6 reporting.

    Returns a JSON-friendly dict with ``n_events``, percentile
    durations, peak-score distribution, and inter-event interval
    statistics — exactly the deployment-relevant numbers a Chapter 7
    streaming-inference subsection wants to quote.
    """
    if not events:
        return {
            "n_events": 0,
            "mean_duration_s": float("nan"),
            "median_duration_s": float("nan"),
            "p95_duration_s": float("nan"),
            "min_duration_s": float("nan"),
            "max_duration_s": float("nan"),
            "mean_peak_score": float("nan"),
            "median_peak_score": float("nan"),
            "max_peak_score": float("nan"),
            "events_per_recording": {},
        }
    durations = np.asarray([e.duration_s for e in events], dtype=np.float64)
    peaks = np.asarray([e.peak_score for e in events], dtype=np.float64)
    by_recording: dict[str, int] = {}
    for e in events:
        by_recording[e.recording_id] = by_recording.get(e.recording_id, 0) + 1
    return {
        "n_events": int(durations.size),
        "mean_duration_s": float(durations.mean()),
        "median_duration_s": float(np.median(durations)),
        "p95_duration_s": float(np.percentile(durations, 95)),
        "min_duration_s": float(durations.min()),
        "max_duration_s": float(durations.max()),
        "mean_peak_score": float(peaks.mean()),
        "median_peak_score": float(np.median(peaks)),
        "max_peak_score": float(peaks.max()),
        "events_per_recording": by_recording,
    }


# ---------------------------------------------------------------------------
# Sliding-window V3 inference
# ---------------------------------------------------------------------------


def sliding_window_v3_inference(
    v2_encoder,
    flow,
    segment,
    *,
    v2_cfg,
    inference_stride_s: float = 0.25,
    device: str = "auto",
):
    """Run V3 on a single paired segment at a finer-than-training stride.

    The training-time V2 / V3 pipeline uses ``window_seconds = 2.0``
    and ``window_stride_seconds = 1.0``.  At inference time we keep
    the 2 s window (V3 was trained for it) but slide with a much
    finer stride (e.g. 250 ms), producing one V3 score every 250 ms
    rather than every 1 s.

    Returns ``(times_s, scores, contexts)`` aligned over the segment.

    `segment` is a `_PairedSegment` (the in-memory feature stack;
    see `v2_ssl.py`).
    """
    import torch
    import torch.utils.data as tud

    from ...config import resolve_device
    from ..context.v2_ssl import (
        V2SSLConfig,
        _PairedGroupedBatchSampler,
        _PairedWindowedDataset,
        _collate,
    )
    from .v3_trainer import _extract_xc

    # Build a per-segment dataset with the finer stride.  We clone
    # the v2_cfg to override only `window_stride_seconds`.
    fine_cfg = V2SSLConfig(
        **{
            **v2_cfg.__dict__,
            "window_stride_seconds": float(inference_stride_s),
        }
    )
    ds = _PairedWindowedDataset([segment], fine_cfg)
    if len(ds) == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, flow.c_dim), dtype=np.float64),
        )
    sampler = _PairedGroupedBatchSampler(
        ds, fine_cfg.batch_size, shuffle=False, seed=0,
    )
    loader = tud.DataLoader(ds, batch_sampler=sampler, collate_fn=_collate)

    dev = resolve_device(device)
    x, c, _ = _extract_xc(v2_encoder, loader, dev)
    with torch.no_grad():
        scores = flow.anomaly_score(x.to(dev), c.to(dev)).cpu().numpy().astype(np.float64)
    # Window-centre times: start_ac / acoustic_fs + window_seconds / 2.
    # We recover the start indices from the dataset's `_refs` list.
    times = np.array(
        [
            (ref[1] / max(segment.acoustic_fs, 1e-9))
            + fine_cfg.window_seconds / 2.0
            for ref in ds._refs
        ],
        dtype=np.float64,
    )
    return times, scores, c.numpy().astype(np.float64)


__all__ = [
    "AnomalyEvent",
    "detect_events_from_score_timeline",
    "sliding_window_v3_inference",
    "summarise_events",
]
