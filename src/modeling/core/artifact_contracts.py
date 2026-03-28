"""Shared artifact metadata contracts for schema versioning and compatibility checks.

Every serialized artifact (flow model, mode classifier, baseline) is stamped with
a schema version and type tag via stamp_artifact_metadata(). When loaded,
validate_artifact_metadata() verifies the artifact type and schema version against
the supported set.

This prevents silent breakage when an artifact written by an older run is loaded
by code that has changed the payload format — a critical guard given that model
checkpoints are large and re-running the full pipeline is expensive.
"""

from __future__ import annotations

from typing import Any

ARTIFACT_SCHEMA_VERSION = "2.0"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = {"2.0"}


def stamp_artifact_metadata(*, artifact_type: str) -> dict[str, str]:
    """Return the _meta dict to embed in a freshly saved artifact."""
    return {
        "artifact_type": str(artifact_type),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "compatibility": "Strict schema metadata required",
    }


def validate_artifact_metadata(
    *,
    blob: dict[str, Any],
    expected_type: str,
) -> str:
    meta = blob.get("_meta")
    if meta is None:
        raise ValueError("Invalid artifact metadata: missing _meta")
    if not isinstance(meta, dict):
        raise ValueError("Invalid artifact metadata: _meta must be an object")

    artifact_type = meta.get("artifact_type")
    schema_version = meta.get("schema_version")

    if str(artifact_type) != str(expected_type):
        raise ValueError(
            "Artifact type mismatch: "
            f"expected {expected_type!r}, got {artifact_type!r}"
        )

    schema = str(schema_version)
    if schema not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported artifact schema_version: "
            f"{schema!r}. Supported: {sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}"
        )
    return schema
