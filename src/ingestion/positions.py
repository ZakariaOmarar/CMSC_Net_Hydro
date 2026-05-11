"""Position registry for sensor channels across all test datasets and Illwerke.

Each dataset specifies sensor coordinates in its own native format and unit:
  - D1: no native positions. We synthesize a default geometry (4 mics on a small
        circle at one height, 4 vibrations on a circle at another height). The
        positions are not physically meaningful but they are *consistent* across
        runs, so the channel-agnostic encoder's positional embeddings remain
        well-defined.
  - D2: free-form text file (`node_position.txt`) with lines like
        `vibration_A:     (10, 0, 23)`  (centimeters). Note that the file uses
        a mix of ASCII colons and full-width colons (`：`).
  - D3: JSON list of `{"id": "Fr", "x": 0, "y": 0, "z": 1}` records. IDs use
        no underscore (e.g., "Dl" for D_l, "(V)F" for vibration F).
  - Illwerke: derived from `src.config.constants.mic_cartesian_positions()` and
        the accelerometer entries in `SENSOR_LAYOUT`.

All positions are normalised to METERS at the registry boundary so downstream
modules never have to think about units.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config.constants import (
    CASING_RADIUS_M,
    GENERATOR_LEVEL_Z_M,
    SENSOR_LAYOUT,
    TURBINE_LEVEL_Z_M,
    mic_cartesian_positions,
)


@dataclass(frozen=True)
class SensorPosition:
    """3-D position of a single sensor in meters."""

    id: str
    modality: str  # "mic" or "vibration"
    xyz: tuple[float, float, float]


class PositionRegistry:
    """Maps (modality, sensor_id) → 3-D position in meters.

    The lookup is forgiving on sensor-ID formatting:
      - underscores are stripped (D3 wav file `recorded_D_l.wav` → "D_l" ↔ "Dl")
      - vibration channel IDs are tried both bare ("D") and decorated ("(V)D")
      - leading "L1_"/"L2_" prefixes (Illwerke) match the SENSOR_LAYOUT names directly
    """

    def __init__(self, sensors: list[SensorPosition]) -> None:
        self._sensors = list(sensors)
        self._mic_index: dict[str, np.ndarray] = {}
        self._vib_index: dict[str, np.ndarray] = {}
        for s in self._sensors:
            xyz = np.asarray(s.xyz, dtype=np.float64)
            if s.modality == "mic":
                for k in self._id_aliases(s.id, modality="mic"):
                    self._mic_index[k] = xyz
            elif s.modality == "vibration":
                for k in self._id_aliases(s.id, modality="vibration"):
                    self._vib_index[k] = xyz
            else:
                raise ValueError(f"unknown modality {s.modality!r} for {s.id!r}")

    @staticmethod
    def _id_aliases(raw_id: str, modality: str) -> list[str]:
        candidates = {raw_id, raw_id.upper(), raw_id.replace("_", "")}
        if modality == "vibration":
            stripped = raw_id.replace("(V)", "").replace("_", "")
            candidates.add(stripped)
            candidates.add(stripped.upper())
            candidates.add(f"(V){stripped}")
        return sorted(candidates)

    @property
    def mic_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self._sensors if s.modality == "mic")

    @property
    def vibration_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self._sensors if s.modality == "vibration")

    def lookup_mic(self, sensor_id: str) -> np.ndarray:
        return self._lookup(self._mic_index, sensor_id, "mic")

    def lookup_vibration(self, sensor_id: str) -> np.ndarray:
        return self._lookup(self._vib_index, sensor_id, "vibration")

    @staticmethod
    def _lookup(
        index: dict[str, np.ndarray], sensor_id: str, modality: str
    ) -> np.ndarray:
        for k in PositionRegistry._id_aliases(sensor_id, modality):
            if k in index:
                return index[k].copy()
        raise KeyError(
            f"no {modality} position for sensor id {sensor_id!r} "
            f"(known: {sorted(index)})"
        )

    @classmethod
    def for_dataset(cls, dataset_id: str, **kwargs) -> "PositionRegistry":
        dataset_id = dataset_id.lower()
        if dataset_id == "d1":
            return cls(_d1_default_geometry())
        if dataset_id == "d2":
            return cls(_parse_d2_node_position_txt(Path(kwargs["path"])))
        if dataset_id == "d3":
            return cls(_parse_d3_position_json(Path(kwargs["path"])))
        if dataset_id == "d4":
            # D4 reuses the D3 sensor layout (same 9 mic + 4 accel names)
            # so the D3 JSON parser handles it directly.
            return cls(_parse_d3_position_json(Path(kwargs["path"])))
        if dataset_id in ("illwerke", "illwerke_raw"):
            return cls(_illwerke_geometry())
        raise ValueError(f"unknown dataset id: {dataset_id!r}")


# ---------------------------------------------------------------------- D1 ---

def _d1_default_geometry() -> list[SensorPosition]:
    """Synthesize a placeholder geometry for D1 (which has no native positions).

    Mics B, C, D, E are placed on a 1.0 m circle at z=2.0 m at 0/90/180/270°.
    Vibrations B, C, D, E are placed on a 0.6 m circle at z=0.0 m, offset 45°.
    These are NOT physical positions; they exist so position embeddings are
    well-defined and consistent across runs.
    """
    sensors: list[SensorPosition] = []
    mic_radius, mic_z = 1.0, 2.0
    vib_radius, vib_z = 0.6, 0.0
    for ch_id, deg in zip("BCDE", (0, 90, 180, 270)):
        theta = math.radians(deg)
        sensors.append(
            SensorPosition(
                id=ch_id,
                modality="mic",
                xyz=(mic_radius * math.cos(theta), mic_radius * math.sin(theta), mic_z),
            )
        )
    for ch_id, deg in zip("BCDE", (45, 135, 225, 315)):
        theta = math.radians(deg)
        sensors.append(
            SensorPosition(
                id=ch_id,
                modality="vibration",
                xyz=(vib_radius * math.cos(theta), vib_radius * math.sin(theta), vib_z),
            )
        )
    return sensors


# ---------------------------------------------------------------------- D2 ---

# Lines look like: `vibration_A:     (10, 0, 23)` or `microfone_D：   (0, 41, 15)`
# Note: file uses mixed colon characters. `microfone` (sic) is the literal spelling.
_D2_LINE_RE = re.compile(
    r"^(?P<kind>vibration|microfone|microphone)_(?P<id>[^\s:：]+)\s*[:：]\s*"
    r"\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<z>-?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)


def _parse_d2_node_position_txt(path: Path) -> list[SensorPosition]:
    if not path.exists():
        raise FileNotFoundError(f"D2 position file not found: {path}")

    sensors: list[SensorPosition] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            m = _D2_LINE_RE.match(line)
            if m is None:
                continue
            kind = m.group("kind").lower()
            sid = m.group("id").upper()
            xyz_cm = (float(m.group("x")), float(m.group("y")), float(m.group("z")))
            xyz_m = tuple(v / 100.0 for v in xyz_cm)  # cm → m
            modality = "vibration" if kind == "vibration" else "mic"
            sensors.append(SensorPosition(id=sid, modality=modality, xyz=xyz_m))
    if not sensors:
        raise ValueError(f"no sensor positions parsed from {path}")
    return sensors


# ---------------------------------------------------------------------- D3 ---

def _parse_d3_position_json(path: Path) -> list[SensorPosition]:
    """Parse `position.json`.  IDs starting with `(V)` are vibrations; the rest
    are mics.

    **Unit convention** (corrected 2026-05): the JSON values are
    centimetres on the ~ 10 cm bench-top prototype scale, matching D2's
    `node_position.txt` format and the spatial-label folder convention
    (`(x, y, z)` in cm).  The parser therefore divides by 100 to return
    metres — without this conversion, mic positions land at 0–11 m
    (eleven *meters*, the ROW II reference scale) instead of 0–0.11 m,
    which silently breaks SRP-PHAT and the V4 spatial-feature pipeline
    (V0 SRP-PHAT errors of 4–8 m on a 10 cm prototype were the original
    symptom).
    """
    if not path.exists():
        raise FileNotFoundError(f"D3 position file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    sensors: list[SensorPosition] = []
    for r in records:
        sid = str(r["id"])
        xyz_cm = (float(r["x"]), float(r["y"]), float(r["z"]))
        xyz_m = tuple(v / 100.0 for v in xyz_cm)  # cm → m
        if sid.startswith("(V)"):
            sensors.append(SensorPosition(id=sid, modality="vibration", xyz=xyz_m))
        else:
            sensors.append(SensorPosition(id=sid, modality="mic", xyz=xyz_m))
    return sensors


# ----------------------------------------------------------------- Illwerke ---

def _illwerke_geometry() -> list[SensorPosition]:
    sensors: list[SensorPosition] = []
    for xyz in mic_cartesian_positions():
        # Use the SENSOR_LAYOUT name (e.g., "L1_0") as the canonical id, looked up
        # by channel index.
        pass
    mic_records = (
        SENSOR_LAYOUT["microphones"]["level_1"]
        + SENSOR_LAYOUT["microphones"]["level_2"]
    )
    mic_records_sorted = sorted(mic_records, key=lambda r: r["channel"])
    mic_xyz = mic_cartesian_positions()
    for r, xyz in zip(mic_records_sorted, mic_xyz):
        sensors.append(SensorPosition(id=str(r["name"]), modality="mic", xyz=xyz))

    accel_radius = CASING_RADIUS_M * 0.9  # accelerometers slightly inboard of mic ring
    for r in SENSOR_LAYOUT["accelerometers"]:
        theta = math.radians(float(r["angle_deg"]))
        xyz = (
            accel_radius * math.cos(theta),
            accel_radius * math.sin(theta),
            float(r["z_m"]),
        )
        sensors.append(SensorPosition(id=str(r["name"]), modality="vibration", xyz=xyz))

    # Sanity: heights used (silences flake8 about unused TURBINE/GENERATOR imports)
    _ = (TURBINE_LEVEL_Z_M, GENERATOR_LEVEL_Z_M)

    return sensors


__all__ = ["SensorPosition", "PositionRegistry"]
