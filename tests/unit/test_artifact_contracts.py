from __future__ import annotations

import pytest

from src.modeling.core.artifact_contracts import ARTIFACT_SCHEMA_VERSION
from src.modeling.core.artifact_contracts import stamp_artifact_metadata
from src.modeling.core.artifact_contracts import validate_artifact_metadata


def test_validate_artifact_metadata_rejects_missing_meta() -> None:
    with pytest.raises(ValueError, match="missing _meta"):
        validate_artifact_metadata(
            blob={"state_dict": {}},
            expected_type="flow",
        )


def test_stamp_and_validate_artifact_metadata_roundtrip() -> None:
    meta = stamp_artifact_metadata(artifact_type="mode")
    schema = validate_artifact_metadata(
        blob={"_meta": meta, "state_dict": {}},
        expected_type="mode",
    )
    assert schema == ARTIFACT_SCHEMA_VERSION


def test_validate_artifact_metadata_rejects_type_mismatch() -> None:
    with pytest.raises(ValueError, match="Artifact type mismatch"):
        validate_artifact_metadata(
            blob={
                "_meta": {
                    "artifact_type": "mode",
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                }
            },
            expected_type="flow",
        )


def test_validate_artifact_metadata_rejects_unsupported_version() -> None:
    with pytest.raises(ValueError, match="Unsupported artifact schema_version"):
        validate_artifact_metadata(
            blob={
                "_meta": {
                    "artifact_type": "flow",
                    "schema_version": "99.0",
                }
            },
            expected_type="flow",
        )
