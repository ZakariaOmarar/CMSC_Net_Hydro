"""Unified loader for the three thesis test datasets (D1, D2, D3) and a future
Illwerke raw drop-in.

Provides:
  - `DatasetSpec` — declarative dataset registration (root, modality counts,
    position source, label scheme). Loaded from `configs/datasets/{id}.yaml`.
  - `TestDatasetSegment` — one loaded recording: a `DataSegment` plus per-channel
    3-D positions and parsed labels (mode, operating condition, spatial label).
  - `TestDatasetLoader` — wraps the existing `RecordingScanner` and
    `WavVibrationAdapter`, parameterised per-dataset, plus the `PositionRegistry`.

Constraint #3 (Illwerke ingestion-ready): adding a dataset is a YAML edit.
No code changes are needed for a fourth dataset that follows the
`recorded_<sensor>[_<extra>].wav` + `vibration_<sensor>[_<extra>].csv`
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import yaml

from ..data import DataSegment
from .adapters import WavVibrationAdapter
from .positions import PositionRegistry
from .scanner import RecordingGroup, RecordingScanner


_KNOWN_MODES = ("Pump", "Standstill", "Turbine", "RandomFault")
_D2_POS_RE = re.compile(
    r"^pos_\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)_(?P<context>.+)$"
)
_D3_HIT_RE = re.compile(
    r"^hit_between_(?P<a>[A-Za-z]+)_(?P<b>[A-Za-z]+)_(?P<speed>speed\d+)$"
)
_D3_SPEED_RE = re.compile(r"^speed\d+$")
# D4 RandomFault subfolders are bare `(x, y, z)` (no `pos_` prefix, optional spaces).
_D4_POS_RE = re.compile(
    r"^\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$"
)
# D4's RandomFault parent folder: e.g. `RandomFault_knock_unter_speed1`.
_D4_RF_PARENT_RE = re.compile(
    r"^RandomFault_[A-Za-z0-9_]*?_(?P<speed>speed\d+)$"
)


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative configuration for one dataset."""

    id: str  # "d1" | "d2" | "d3" | "d4" | "illwerke_raw"
    root: Path
    n_mics: int
    n_vibrations: int
    accel_target_sr: int  # we DO NOT downsample on the peak path; for raw, we keep the inferred ADC rate
    position_source: str  # "default" | path to file
    label_scheme: str  # "d1_mode" | "d2_mode_with_spatial" | "d3_speed_with_hit" | "d4_speed_with_random"
    vibration_format: str = "peak"  # "peak" (D1/D2/D3) | "raw" (D4 raw-waveform)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetSpec":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            id=str(data["id"]),
            root=Path(data["root"]),
            n_mics=int(data["n_mics"]),
            n_vibrations=int(data["n_vibrations"]),
            accel_target_sr=int(data["accel_target_sr"]),
            position_source=str(data["position_source"]),
            label_scheme=str(data["label_scheme"]),
            vibration_format=str(data.get("vibration_format", "peak")),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True)
class TestDatasetSegment:
    """A loaded recording with sensor positions and parsed labels.

    Conceptual model (clarified by the user, 2026-05):
      - `mode_label` is the operating mode, one of {"Pump", "Standstill",
        "Turbine"} when the campaign annotated it (D1 + D2 mode folders, plus
        D2 single-mode RandomFault folders).  D3 / D4 recordings have a real
        mode but the campaign did not record which one — `mode_label=None`
        marks them as "discoverable but unknown".  The `speed{N}` token in
        D3 / D4 is a **background-noise fan level**, not a mode.
      - `op_condition` carries the orthogonal operational variable (D3 /
        D4: `speed{N}` fan level).
      - `is_anomaly` is True for recordings inside a RandomFault / hit
        folder, False otherwise.  This is **independent** of `mode_label`:
        a recording can be in Turbine mode with anomalies (D2 single-mode
        RandomFault), or in unknown mode without anomalies (D3 / D4
        speed-bucket healthy), etc.
      - `spatial_label` is the metric anomaly position when known.  For D4
        sparse-anomaly recordings the label applies *only to the alert
        windows V3 surfaces*; the V4 trainer therefore gates D4 windows by
        V3 before applying the label.
    """

    __test__ = False  # not a pytest test class

    segment: DataSegment
    mic_positions: np.ndarray  # (Nm, 3) meters, row-aligned to segment.mic_data
    vib_positions: np.ndarray  # (Nv, 3) meters, row-aligned to segment.accel_data
    mic_ids: tuple[str, ...]
    vib_ids: tuple[str, ...]
    mode_label: str | None
    op_condition: str | None
    spatial_label: tuple[float, float, float] | None
    dataset_id: str
    recording_id: str
    source_dir: Path
    # Default-False so legacy test constructors (which predate the field)
    # keep working without per-test edits.  Production loader always sets it.
    is_anomaly: bool = False


class TestDatasetLoader:
    """Load all `TestDatasetSegment`s for one dataset spec."""

    __test__ = False  # not a pytest test class

    def __init__(self, spec: DatasetSpec) -> None:
        self._spec = spec
        if spec.position_source == "default":
            self._registry = PositionRegistry.for_dataset(spec.id)
        elif spec.position_source == "rowii":
            self._registry = PositionRegistry.for_dataset("illwerke")
        else:
            self._registry = PositionRegistry.for_dataset(
                spec.id, path=Path(spec.position_source)
            )

        # Per-dataset, allow exactly the configured channel counts.
        self._adapter = WavVibrationAdapter(
            expected_mic_count=spec.n_mics,
            expected_accel_count=spec.n_vibrations,
            allowed_mic_counts=(spec.n_mics,),
            accel_target_sr=spec.accel_target_sr,
            vibration_format=spec.vibration_format,
        )
        self._scanner = RecordingScanner(root_dir=spec.root)

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    @property
    def registry(self) -> PositionRegistry:
        return self._registry

    def list_segments(
        self, *, modes: Iterable[str] | None = None
    ) -> list[TestDatasetSegment]:
        """Load all recording groups; optionally filter by mode label.

        Walks the dataset tree recursively so deep folder layouts (D2's
        `RandomFault/pos_(x,y,z)_<context>/` subfolders) are picked up.
        """
        wanted_modes = None if modes is None else {m.lower() for m in modes}
        groups = self._scan_recursive(self._spec.root)
        segments: list[TestDatasetSegment] = []
        for g in groups:
            try:
                tds = self._load_one(g)
            except Exception:
                # Skip recordings that don't match the strict adapter (e.g., a
                # subfolder that has the wrong channel count for this spec).
                continue
            if wanted_modes is not None:
                m = (tds.mode_label or "").lower()
                if m not in wanted_modes:
                    continue
            segments.append(tds)
        return segments

    def _scan_recursive(self, root: Path) -> list[RecordingGroup]:
        """Walk `root` recursively, returning RecordingGroups from every level."""
        groups: list[RecordingGroup] = []
        seen: set[tuple[Path, str]] = set()

        def visit(directory: Path) -> None:
            if not directory.is_dir():
                return
            try:
                local = RecordingScanner(root_dir=directory).scan_groups()
            except FileNotFoundError:
                local = []
            for g in local:
                key = (g.source_dir.resolve(), g.recording_id)
                if key in seen:
                    continue
                seen.add(key)
                groups.append(g)
            for child in sorted(directory.iterdir()):
                if child.is_dir():
                    visit(child)

        visit(root)
        return groups

    def _load_one(self, group: RecordingGroup) -> TestDatasetSegment:
        # Pre-filter vibration files to the format the adapter will actually
        # read.  D4 directories carry both `vibration_*.csv` (peak) and
        # `vibration_raw_*.csv` (raw); the scanner globs both, so without
        # this filter `vib_ids` would include 8 entries (4 peak + 4 raw,
        # with duplicate sensor IDs after `raw_` is stripped) while the
        # adapter only loads 4 channels — collapsing the position registry
        # alignment.  Filtering here keeps `vib_ids` row-aligned with the
        # adapter's `accel_data`.
        if self._spec.vibration_format == "raw":
            filtered_vib_files = tuple(
                p for p in group.vibration_files if p.stem.startswith("vibration_raw_")
            )
        else:
            filtered_vib_files = tuple(
                p for p in group.vibration_files if not p.stem.startswith("vibration_raw_")
            )

        seg = self._adapter.read_recording_files(
            recording_dir=group.source_dir,
            mic_files=group.mic_files,
            vibration_files=filtered_vib_files,
            recording_id=group.recording_id,
        )
        mic_ids = tuple(_sensor_id(p.name, "recorded") for p in group.mic_files)
        vib_ids = tuple(_sensor_id(p.name, "vibration") for p in filtered_vib_files)

        mic_pos = np.stack([self._registry.lookup_mic(s) for s in mic_ids], axis=0)
        vib_pos = np.stack([self._registry.lookup_vibration(s) for s in vib_ids], axis=0)

        mode_label, op_cond, spatial, is_anomaly = _parse_labels(
            group.source_dir, group.recording_id, self._spec.label_scheme
        )

        return TestDatasetSegment(
            segment=seg,
            mic_positions=mic_pos.astype(np.float64),
            vib_positions=vib_pos.astype(np.float64),
            mic_ids=mic_ids,
            vib_ids=vib_ids,
            mode_label=mode_label,
            op_condition=op_cond,
            spatial_label=spatial,
            is_anomaly=is_anomaly,
            dataset_id=self._spec.id,
            recording_id=group.recording_id,
            source_dir=group.source_dir,
        )


# ----------------------------------------------------------------- helpers ---

def _sensor_id(filename: str, prefix: str) -> str:
    """Extract sensor ID from `<prefix>_<sensor>[_<extra>].(wav|csv)`.

    If the trailing token equals one of the four known mode names (D1
    convention), it is dropped.  If the first tail token is the literal
    `raw` (D4 raw-vibration files: `vibration_raw_D.csv`), it is stripped
    before joining so the position registry sees just `D`.  Otherwise the
    entire tail is the sensor ID (D3 stereo: `recorded_D_l.wav` → `D_l`).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts[0] != prefix:
        raise ValueError(f"unexpected filename {filename!r}, expected prefix {prefix!r}_*")
    tail = parts[1:]
    if tail and tail[0] == "raw":
        tail = tail[1:]
    if len(tail) >= 2 and tail[-1] in _KNOWN_MODES:
        tail = tail[:-1]
    return "_".join(tail)


_HEALTHY_MODES = ("Pump", "Standstill", "Turbine")


def _parse_d2_context_to_mode(context: str) -> str | None:
    """Parse a D2 RandomFault `_<context>` token → single mode label, or None.

    `<context>` is one of `turbine`, `pump`, `standstill`, or a multi-mode
    combination like `turbine_pump`.  The user's campaign protocol injects
    multi-mode anomalies in folders whose context contains *more than one*
    of the three mode tokens; those recordings are dropped from training
    (they conflate two operating regimes in a single recording-level label).
    Single-mode contexts return the mode in canonical case.
    """
    tokens = [t.strip().lower() for t in context.split("_") if t.strip()]
    mode_tokens = [t for t in tokens if t in {"pump", "standstill", "turbine"}]
    if len(mode_tokens) != 1:
        return None  # zero or multi-mode → skip recording
    return mode_tokens[0].capitalize()


def _parse_labels(
    source_dir: Path,
    recording_id: str,
    scheme: str,
) -> tuple[str | None, str | None, tuple[float, float, float] | None, bool]:
    """Return `(mode, op_condition, spatial_label_m, is_anomaly)`.

    `mode` is one of {"Pump", "Standstill", "Turbine"} when the campaign
    explicitly labels it, otherwise None.  `is_anomaly` is True iff the
    recording lives inside a RandomFault / hit folder.  D3 / D4 healthy
    speed buckets carry `mode=None` because the user's protocol does not
    record which of the three modes the unit was in (only the fan-noise
    level).  The pipeline discovers the mode via K-means on `c_t` at
    inference time.
    """
    folder = source_dir.name
    parent = source_dir.parent.name

    if scheme == "d1_mode":
        # D1 source_dir is the mode folder (Pump/Standstill/...) or `All/`,
        # with no spatial labels and no anomaly recordings.
        if folder in _HEALTHY_MODES:
            return folder, None, None, False
        if recording_id in _HEALTHY_MODES:
            return recording_id, None, None, False
        if folder == "RandomFault" or recording_id == "RandomFault":
            # Treat D1's bulk RandomFault folder as anomaly without a single
            # mode label (operating regime varies within the recording).
            return None, None, None, True
        return None, None, None, False

    if scheme == "d2_mode_with_spatial":
        # Healthy mode folders.
        if folder in _HEALTHY_MODES:
            return folder, None, None, False
        # RandomFault with spatial label and a context token.
        m = _D2_POS_RE.match(folder)
        if m is not None:
            xyz_cm = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            xyz_m = (xyz_cm[0] / 100.0, xyz_cm[1] / 100.0, xyz_cm[2] / 100.0)
            mode = _parse_d2_context_to_mode(m.group("context"))
            # Multi-mode (e.g. `_turbine_pump`) returns None → recording is
            # is_anomaly=True with no usable mode label.  The V4 trainer
            # filters these out of its supervised cohort.
            return mode, None, xyz_m, True
        if parent in _HEALTHY_MODES:
            return parent, None, None, False
        return None, None, None, False

    if scheme == "d3_speed_with_hit":
        # `speed{N}` is a fan-noise level, not a mode.  The actual mode is
        # one of {Pump, Standstill, Turbine} but the campaign does not
        # record which — leave mode_label=None and let K-means + Hungarian
        # discover it at inference time.
        m = _D3_HIT_RE.match(folder)
        if m is not None:
            return None, m.group("speed"), None, True  # spatial filled by orchestrator
        if _D3_SPEED_RE.match(folder):
            return None, folder, None, False
        return None, None, None, False

    if scheme == "d4_speed_with_random":
        # Same convention as D3: speed{N} is fan-noise level, not mode.
        # Within `RandomFault_knock_unter_speed{N}/(x, y, z)/` the anomalies
        # are sparse — recording-level spatial labels apply only to the
        # alert windows V3 surfaces; the V4 trainer therefore V3-gates D4
        # samples before applying the label.
        if _D3_SPEED_RE.match(folder):
            return None, folder, None, False
        m_pos = _D4_POS_RE.match(folder)
        if m_pos is not None:
            xyz_cm = (float(m_pos.group(1)), float(m_pos.group(2)), float(m_pos.group(3)))
            xyz_m = (xyz_cm[0] / 100.0, xyz_cm[1] / 100.0, xyz_cm[2] / 100.0)
            m_parent = _D4_RF_PARENT_RE.match(parent)
            speed = m_parent.group("speed") if m_parent is not None else None
            return None, speed, xyz_m, True
        return None, None, None, False

    raise ValueError(f"unknown label_scheme {scheme!r}")


__all__ = ["DatasetSpec", "TestDatasetSegment", "TestDatasetLoader"]
