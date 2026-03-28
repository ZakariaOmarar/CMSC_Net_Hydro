"""Typed data classes for the orchestration run manifest.

JobResult captures the outcome of one training or inference job launched by
train_all.py. StageRollup aggregates counts and timings across all jobs in a
stage. Both are serialized to train_all_manifest.json, providing a machine-
readable record of what ran, how long it took, and whether it succeeded —
useful for detecting flaky jobs and resuming partial runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JobStatus = Literal["ok", "failed", "dry_run", "skipped"]


@dataclass(frozen=True)
class JobResult:
    name: str
    module: str
    stage: str
    command: list[str]
    status: JobStatus
    returncode: int | None
    started_at_utc: str
    finished_at_utc: str
    duration_s: float
    orchestrator_peak_memory_mb: float | None
    output_dir: str | None
    attempt_count: int
    retried: bool
    skipped_reason: str | None


@dataclass
class StageRollup:
    n_jobs: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    n_dry_run: int = 0
    duration_s: float = 0.0
    orchestrator_peak_memory_mb: float = 0.0
