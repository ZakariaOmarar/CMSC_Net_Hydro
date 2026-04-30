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


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative configuration for one dataset."""

    id: str  # "d1" | "d2" | "d3" | "illwerke_raw"
    root: Path
    n_mics: int
    n_vibrations: int
    accel_target_sr: int  # we DO NOT downsample to 4 Hz; keep native CSV rate
    position_source: str  # "default" | path to file
    label_scheme: str  # "d1_mode" | "d2_mode_with_spatial" | "d3_speed_with_hit"
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
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True)
class TestDatasetSegment:
    """A loaded recording with sensor positions and parsed labels."""

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
        seg = self._adapter.read_recording_files(
            recording_dir=group.source_dir,
            mic_files=group.mic_files,
            vibration_files=group.vibration_files,
            recording_id=group.recording_id,
        )
        mic_ids = tuple(_sensor_id(p.name, "recorded") for p in group.mic_files)
        vib_ids = tuple(_sensor_id(p.name, "vibration") for p in group.vibration_files)

        mic_pos = np.stack([self._registry.lookup_mic(s) for s in mic_ids], axis=0)
        vib_pos = np.stack([self._registry.lookup_vibration(s) for s in vib_ids], axis=0)

        mode_label, op_cond, spatial = _parse_labels(
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
            dataset_id=self._spec.id,
            recording_id=group.recording_id,
            source_dir=group.source_dir,
        )


# ----------------------------------------------------------------- helpers ---

def _sensor_id(filename: str, prefix: str) -> str:
    """Extract sensor ID from `<prefix>_<sensor>[_<extra>].(wav|csv)`.

    If the trailing token equals one of the four known mode names (D1
    convention), it is dropped.  Otherwise the entire tail is the sensor ID
    (D3 stereo: `recorded_D_l.wav` → sensor `D_l`).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts[0] != prefix:
        raise ValueError(f"unexpected filename {filename!r}, expected prefix {prefix!r}_*")
    tail = parts[1:]
    if len(tail) >= 2 and tail[-1] in _KNOWN_MODES:
        tail = tail[:-1]
    return "_".join(tail)


def _parse_labels(
    source_dir: Path, recording_id: str, scheme: str
) -> tuple[str | None, str | None, tuple[float, float, float] | None]:
    """Return `(mode, op_condition, spatial_label_m)` per the dataset's scheme."""
    folder = source_dir.name
    parent = source_dir.parent.name

    if scheme == "d1_mode":
        # source_dir is the mode folder (Pump/Standstill/...) OR `All/`.
        # In flat-grouped mode, recording_id encodes the mode (e.g., "Pump").
        if folder in _KNOWN_MODES:
            return folder, None, None
        if recording_id in _KNOWN_MODES:
            return recording_id, None, None
        return None, None, None

    if scheme == "d2_mode_with_spatial":
        # source_dir might be Pump/, Standstill/, Turbine/, or RandomFault/pos_(x,y,z)_<context>/.
        if folder in _KNOWN_MODES:
            return folder, None, None
        m = _D2_POS_RE.match(folder)
        if m is not None:
            xyz_cm = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            xyz_m = (xyz_cm[0] / 100.0, xyz_cm[1] / 100.0, xyz_cm[2] / 100.0)
            return "RandomFault", m.group("context"), xyz_m
        # Fallback: parent is the mode folder.
        if parent in _KNOWN_MODES:
            return parent, None, None
        return None, None, None

    if scheme == "d3_speed_with_hit":
        # source_dir is one of `speed{1,2,3}` or `hit_between_<a>_<b>_speed<n>`.
        m = _D3_HIT_RE.match(folder)
        if m is not None:
            return "RandomFault", m.group("speed"), None  # spatial inferred elsewhere
        if _D3_SPEED_RE.match(folder):
            return "Healthy", folder, None
        return None, None, None

    raise ValueError(f"unknown label_scheme {scheme!r}")


__all__ = ["DatasetSpec", "TestDatasetSegment", "TestDatasetLoader"]
