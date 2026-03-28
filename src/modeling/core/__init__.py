"""Core typed contracts and shared primitives for modeling."""

from .artifact_contracts import ARTIFACT_SCHEMA_VERSION
from .artifact_contracts import stamp_artifact_metadata
from .artifact_contracts import validate_artifact_metadata
from .run_manifest import JobResult
from .run_manifest import JobStatus
from .run_manifest import StageRollup

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "JobResult",
    "JobStatus",
    "StageRollup",
    "stamp_artifact_metadata",
    "validate_artifact_metadata",
]
