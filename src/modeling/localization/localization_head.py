"""Localization Head — Context-Aware Source Position Estimator.

Invoked ONLY when the Detection Head flags is_anomaly = True.
Estimates the (x, y) position of an anomalous sound source in the machine hall
from raw microphone waveforms extracted directly from DataSegment.mic_data.

--------------------------------------------------------------------------------
Design choices and why:
--------------------------------------------------------------------------------

Option A (LocalizationCNN) — 1D CNN over stacked GCC-PHAT vectors.
  Simple, fast to converge, good first baseline. Use when compute is tight or
  you have very few calibration recordings (< 20 windows per position).

Option B (LocalizationTransformer) — Transformer encoder over 6 GCC-PHAT tokens
  with geometry-aware pair positional embeddings and optional cross-attention to
  an SRP-PHAT prior map. Preferred for production because:
    - Self-attention learns which pairs are reliable for each source position,
      handling reverberation-corrupted pairs adaptively.
    - Geometry-aware positional embeddings encode the physical mic layout,
      so the model understands how each pair samples the spatial field.
    - Cross-attention to the SRP-PHAT map anchors learned features in the
      physics-based beamforming prior, which prevents confident wrong predictions
      in regions where the GCC signal is noisy.

Both architectures use FiLM conditioning on context c (from the Detection Head)
to adapt spatial priors to the current operating mode (Turbine / Pump, etc.).
This is important because dominant noise sources and their spatial signatures
differ between modes.

--------------------------------------------------------------------------------
Hardware constraints (ROW II early dataset):
--------------------------------------------------------------------------------

  4 mics, single horizontal ring, same elevation.
  C(4,2) = 6 pairs  → meaningful X/Y variation, no vertical baseline.
  Output is (x, y) only.  Z localization requires a second elevation ring.

  Pair indexing used throughout this module (canonical order):
    pair 0: (mic0, mic1)    pair 3: (mic1, mic2)
    pair 1: (mic0, mic2)    pair 4: (mic1, mic3)
    pair 2: (mic0, mic3)    pair 5: (mic2, mic3)

  GCC-PHAT resolution floor:
    fs = 16000 Hz  →  1 sample = 0.021 m at 343 m/s.
    In practice reverberation dominates; report errors in absolute meters.
    NTP jitter is irrelevant for offline analysis of recorded files.

--------------------------------------------------------------------------------
Known limitations (document in thesis):
--------------------------------------------------------------------------------

  LIMITATION 1: Z axis not available.
    All 4 mics share one elevation.  Z needs a second ring.

  LIMITATION 2: Vibration sensors do not contribute to GCC-PHAT.
    CSVs contain pre-computed amplitude at ~4 Hz, not raw waveforms.
    No inter-sensor delay information is available from them.
    Vibration contributes only through context vector c (FiLM conditioning).

  LIMITATION 3: 6 pairs → arc spatial ambiguities.
    Single-elevation ring gives arc-shaped power maps in X/Y.
    The network mitigates but does not eliminate this ambiguity.

  LIMITATION 4: Accuracy bounded by sample rate, not synchronization.
    Theoretical floor ~0.021 m.  Practical limit set by reverberation.

  LIMITATION 5: Invoke only on anomalous windows.
    Running continuously during normal operation produces meaningless positions.

--------------------------------------------------------------------------------
Classical baseline to beat (from Haller & Lanzlinger pysoundlocalization):
    Single clean source:     0.009 m mean error
    Polyphonic conditions:   9.46 m mean error (worst case > 136 m)
    Target: beat 9.46 m under polyphonic conditions.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "localization_head requires PyTorch. Install with: pip install torch"
    ) from exc


# ---------------------------------------------------------------------------
# Turbine geometry — Rodund 2 | Voith | Drawing 2821-028589-ROW Rev F
# ---------------------------------------------------------------------------
# All positions in metres.  Origin: turbine rotation axis at upper-mic height.
# x-axis: positive toward the entrance opening.
# y-axis: positive 90° CCW from entrance.
# z-axis: positive upward.
# Angle convention: 0° = entrance (+x); CCW positive; CW negative.
#
# BARRIER_RADIUS_M must be confirmed on site (estimated 5.5 m from axis to wall).
# ---------------------------------------------------------------------------

# -- Turbine parameters -------------------------------------------------------
TURBINE: dict = {
    "name": "Francis Turbine — Rodund 2",
    "type": "pump-turbine (reversible Francis)",
    "manufacturer": "Voith",
    "drawing_number": "2821-028589-ROW",
    "drawing_revision": "F",
    "drawing_date": "2019-04-04",
    "facility_id": "27011",
    "rpm": 375,
    "runner_blades": 7,
    "guide_vanes": 20,
    "stator_slots": 288,
    "rotor_poles": 16,
}

# -- Characteristic frequencies (Hz) -----------------------------------------
FREQUENCIES_HZ: dict = {
    "shaft_hz": 6.25,
    "runner_blade_passing_hz": 43.75,
    "guide_vane_passing_hz": 125.0,
    "rotor_pole_passing_hz": 100.0,
    "electrical_network_hz": 50.0,
}

# -- Vertical levels ----------------------------------------------------------
BARRIER_RADIUS_M: float = 5.5  # *** CONFIRM ON SITE ***
Z_UPPER_MIC: float = 0.00  # m — reference level
Z_LOWER_MIC: float = -0.80  # m — 80 cm below upper mics
Z_ACCEL_SENSOR: float = -0.90  # m — 10 cm below lower mics
SPEED_OF_SOUND_MS: float = 343.0  # m/s at ~20 °C

# -- Individual sensor XYZ positions (metres) ---------------------------------
# Computed from arc-distance table on drawing 2821-028589-ROW, R = 5.5 m.
_SENSOR_XYZ_RAW: dict[str, list[float]] = {
    # Upper microphones  (z = Z_UPPER_MIC = 0.00 m)
    "MIC_UP_2": [5.345904528398038, 1.2927895316923597, 0.0],
    "MIC_UP_4": [4.732751574776851, 2.8019747556762953, 0.0],
    "MIC_UP_6": [5.5, 0.0, 0.0],
    "MIC_UP_8": [2.269476735317571, -5.009937659078434, 0.0],
    # Lower (bottom) microphones  (z = Z_LOWER_MIC = -0.80 m)
    "MIC_BOT_1": [5.1039378506168624, 2.0493458510072227, -0.8],
    "MIC_BOT_3": [-1.5397624963450893, 5.2800692661033475, -0.8],
    "MIC_BOT_5": [5.07359032360746, -2.123365542763833, -0.8],
    "MIC_BOT_7": [5.394303917240584, -1.0730728066831692, -0.8],
    "MIC_BOT_9": [2.269476735317571, -5.009937659078434, -0.8],
    # Accelerometers  (z = Z_ACCEL_SENSOR = -0.90 m)
    "ACCEL_1": [5.5, 0.0, -0.9],
    "ACCEL_2": [4.732751574776851, 2.8019747556762953, -0.9],
    "ACCEL_3": [3.2960976627194523, 4.402924050879752, -0.9],
    "ACCEL_4": [2.269476735317571, -5.009937659078434, -0.9],
}

SENSOR_XYZ: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) for k, v in _SENSOR_XYZ_RAW.items()
}

# -- Convenience arrays -------------------------------------------------------
# Upper mics in label order → channel indices 0–3 for the 4-mic early dataset.
#   ch 0 = MIC_UP_2,  ch 1 = MIC_UP_4,  ch 2 = MIC_UP_6,  ch 3 = MIC_UP_8
MIC_UPPER_XY: np.ndarray = np.array(
    [SENSOR_XYZ[k][:2] for k in ("MIC_UP_2", "MIC_UP_4", "MIC_UP_6", "MIC_UP_8")],
    dtype=np.float64,
)  # shape (4, 2)

# All 9 mics in numerical label order (BOT/UP interleaved: 1,2,3,4,5,6,7,8,9).
#   ch 0 = MIC_BOT_1, ch 1 = MIC_UP_2, ch 2 = MIC_BOT_3, ch 3 = MIC_UP_4,
#   ch 4 = MIC_BOT_5, ch 5 = MIC_UP_6, ch 6 = MIC_BOT_7, ch 7 = MIC_UP_8,
#   ch 8 = MIC_BOT_9
MIC_ALL_XYZ: np.ndarray = np.array(
    [
        SENSOR_XYZ[k]
        for k in (
            "MIC_BOT_1",
            "MIC_UP_2",
            "MIC_BOT_3",
            "MIC_UP_4",
            "MIC_BOT_5",
            "MIC_UP_6",
            "MIC_BOT_7",
            "MIC_UP_8",
            "MIC_BOT_9",
        )
    ],
    dtype=np.float64,
)  # shape (9, 3)

# 4 accelerometers in label order → channel indices 0–3.
#   ch 0 = ACCEL_1,  ch 1 = ACCEL_2,  ch 2 = ACCEL_3,  ch 3 = ACCEL_4
ACCEL_XYZ: np.ndarray = np.array(
    [SENSOR_XYZ[k] for k in ("ACCEL_1", "ACCEL_2", "ACCEL_3", "ACCEL_4")],
    dtype=np.float64,
)  # shape (4, 3)

# Max Euclidean distance between any pair in each mic set (used for TDOA range).
#   Upper-4   : MIC_UP_4 ↔ MIC_UP_8  = 8.1911 m
#   All-9     : MIC_BOT_3 ↔ MIC_BOT_9 ≈ 10.97 m
_MAX_DIST_UPPER4_M: float = 8.1911
_MAX_DIST_ALL9_M: float = 10.9729


# ==============================================================================
#  SECOND TEST DATASET GEOMETRY — Bench-top prototype setup
# ==============================================================================
# Source: data/second_test_dataset/node_position.txt (unit: cm → converted to m)
#
# 5 microphones : D, E, F, G, I  (channel order matches sorted filename: D=0 … I=4)
# 5 vibration sensors: A, B, C, D, E  (channel order A=0 … E=4)
#
# Coordinate system: arbitrary lab frame, origin at corner of the test box.
#   x-axis : width of the box
#   y-axis : depth (length) of the box
#   z-axis : height above the table surface
#
# Fault injection positions (folder names in data/second_test_dataset/RandomFault/)
# correspond to the vibration sensor positions below (in integer cm, rounded).
# ==============================================================================

# Raw positions from node_position.txt (cm)
_S2_VIB_XYZ_CM: dict[str, list[float]] = {
    "vibration_A": [10.0, 0.0, 23.0],
    "vibration_B": [15.5, 6.0, 15.0],
    "vibration_C": [0.0, 17.0, 12.0],
    "vibration_D": [0.0, 40.0, 15.0],
    "vibration_E": [15.5, 30.0, 16.0],
}
_S2_MIC_XYZ_CM: dict[str, list[float]] = {
    "mic_D": [0.0, 41.0, 15.0],
    "mic_E": [0.0, 31.0, 16.0],
    "mic_F": [10.0, 0.0, 24.0],
    "mic_G": [15.5, 5.0, 15.0],
    "mic_I": [0.0, 10.0, 15.0],
}

# Converted to metres — used throughout the localization pipeline
S2_VIB_XYZ_M: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) / 100.0 for k, v in _S2_VIB_XYZ_CM.items()
}
S2_MIC_XYZ_M: dict[str, np.ndarray] = {
    k: np.array(v, dtype=np.float64) / 100.0 for k, v in _S2_MIC_XYZ_CM.items()
}

# Ordered arrays (channel index = alphabetical / label order as returned by glob sort)
#   mic channels: D=0, E=1, F=2, G=3, I=4
#   vib channels: A=0, B=1, C=2, D=3, E=4
S2_MIC_XYZ: np.ndarray = np.array(
    [S2_MIC_XYZ_M[k] for k in ("mic_D", "mic_E", "mic_F", "mic_G", "mic_I")],
    dtype=np.float64,
)  # shape (5, 3)

S2_VIB_XYZ: np.ndarray = np.array(
    [
        S2_VIB_XYZ_M[k]
        for k in (
            "vibration_A",
            "vibration_B",
            "vibration_C",
            "vibration_D",
            "vibration_E",
        )
    ],
    dtype=np.float64,
)  # shape (5, 3)

# Ground truth fault positions (cm, integer) → label as written in folder names
S2_FAULT_POSITIONS_CM: dict[str, np.ndarray] = {
    "pos_(10,0,23)": np.array([10.0, 0.0, 23.0]),  # vibration_A
    "pos_(15,6,15)": np.array([15.0, 6.0, 15.0]),  # vibration_B (≈15.5)
    "pos_(0,17,12)": np.array([0.0, 17.0, 12.0]),  # vibration_C
    "pos_(0,40,15)": np.array([0.0, 40.0, 15.0]),  # vibration_D
    "pos_(15,30,15)": np.array([15.0, 30.0, 15.0]),  # vibration_E (≈15.5, 16→15)
}
# Same set in metres for direct comparison with localization output
S2_FAULT_POSITIONS_M: dict[str, np.ndarray] = {
    k: v / 100.0 for k, v in S2_FAULT_POSITIONS_CM.items()
}

# Max inter-mic distance for TDOA range (mic_D ↔ mic_F ≈ diagonal of the box)
_S2_MAX_MIC_DIST_M: float = float(
    max(
        np.linalg.norm(S2_MIC_XYZ[i] - S2_MIC_XYZ[j])
        for i in range(len(S2_MIC_XYZ))
        for j in range(i + 1, len(S2_MIC_XYZ))
    )
)

# Mic-pair indices for second dataset: C(5,2) = 10 pairs
MIC_PAIRS_S2: list[tuple[int, int]] = [
    (i, j) for i in range(5) for j in range(i + 1, 5)
]
N_PAIRS_S2: int = len(MIC_PAIRS_S2)  # 10


# ---------------------------------------------------------------------------
# Canonical mic-pair indices (C(4,2) = 6 pairs, fixed order)
# Used for the early 4-mic dataset (upper ring only).
# ---------------------------------------------------------------------------

MIC_PAIRS: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
N_PAIRS: int = len(MIC_PAIRS)  # 6

# Canonical mic-pair indices for the full 9-mic dataset (C(9,2) = 36 pairs).
MIC_PAIRS_9: list[tuple[int, int]] = [(i, j) for i in range(9) for j in range(i + 1, 9)]
N_PAIRS_9: int = len(MIC_PAIRS_9)  # 36


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalizationConfig:
    """Frozen configuration for the localization head.

    Args:
        fs: Audio sample rate in Hz (must match DataSegment.mic_sample_rate, 16000).
        c_air: Speed of sound in m/s.
        max_mic_distance: Maximum Euclidean distance (metres) between any mic
            pair in the array. Sets the TDOA search range.
            Early 4-mic dataset (MIC_UP_2/4/6/8):  8.1911 m  → 383 samples at 16 kHz
            Full  9-mic dataset (all mics):        10.9729 m  → 512 samples at 16 kHz
            Use _MAX_DIST_UPPER4_M or _MAX_DIST_ALL9_M constants directly.
        d_ctx: Dimensionality of the context vector c produced by the Detection
            Head's LightweightContextEncoder (must match at runtime).
        d_model: Internal feature dimension for the Transformer. Must be even.
        n_heads: Number of self-attention heads. d_model must be divisible by n_heads.
        n_layers: Number of Transformer encoder layers.
        dropout: Dropout applied inside Transformer and FiLM.
        use_srp_cross_attn: If True, LocalizationTransformer adds a cross-attention
            sublayer that attends to the SRP-PHAT prior map. Requires mic_xy and
            grid parameters at build time.
    """

    fs: int = 16000
    c_air: float = 343.0
    max_mic_distance: float = (
        _MAX_DIST_UPPER4_M  # 8.1911 m — upper-4 mic ring (early dataset)
    )
    d_ctx: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    use_srp_cross_attn: bool = False

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be > 0")
        if self.c_air <= 0.0:
            raise ValueError("c_air must be > 0")
        if self.max_mic_distance <= 0.0:
            raise ValueError("max_mic_distance must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

    @property
    def max_delay_samples(self) -> int:
        """Maximum TDOA in samples = floor(max_mic_distance / c_air * fs)."""
        return int(self.max_mic_distance / self.c_air * self.fs)

    @property
    def gcc_length(self) -> int:
        """Length L of each GCC-PHAT vector = 2 * max_delay_samples + 1."""
        return 2 * self.max_delay_samples + 1


# ---------------------------------------------------------------------------
# GCC-PHAT computation (pure NumPy, runs outside the autograd graph)
# ---------------------------------------------------------------------------


def gcc_phat(
    x_i: np.ndarray,
    x_j: np.ndarray,
    max_delay_samples: int,
    n_fft: int | None = None,
) -> np.ndarray:
    """Compute the GCC-PHAT cross-correlation vector for one mic pair.

    Args:
        x_i: 1D float array of audio samples for mic i.
        x_j: 1D float array of audio samples for mic j.
        max_delay_samples: Half-width of the output in samples.
        n_fft: FFT size. Defaults to len(x_i); increase to next power-of-two
            for efficiency on non-power-of-two window lengths.

    Returns:
        gcc: float32 array of shape (L,), where L = 2 * max_delay_samples + 1.
            Index max_delay_samples corresponds to zero lag.
    """
    if n_fft is None:
        n_fft = len(x_i)

    Xi = np.fft.rfft(x_i, n=n_fft)
    Xj = np.fft.rfft(x_j, n=n_fft)
    G = Xi * np.conj(Xj)
    G_phat = G / (np.abs(G) + 1e-8)  # PHAT whitening
    gcc_full = np.fft.irfft(G_phat, n=n_fft)
    gcc_full = np.fft.fftshift(gcc_full)  # center at zero lag

    center = len(gcc_full) // 2
    gcc = gcc_full[center - max_delay_samples : center + max_delay_samples + 1]
    return gcc.astype(np.float32)  # shape (L,)


def compute_gcc_phat_stack(
    mic_data: np.ndarray,
    config: LocalizationConfig,
    n_fft: int | None = None,
) -> np.ndarray:
    """Compute GCC-PHAT for all 6 canonical mic pairs.

    Args:
        mic_data: float array of shape (n_mics, n_samples). Taken directly
            from DataSegment.mic_data.
        config: LocalizationConfig holding fs, max_delay_samples, etc.
        n_fft: Optional FFT size override.

    Returns:
        acoustic_gcc: float32 array of shape (6, L).

    Raises:
        ValueError: If mic_data has fewer than 4 channels.
    """
    if mic_data.shape[0] < 4:
        raise ValueError(
            f"Localization requires 4 mic channels; got {mic_data.shape[0]}. "
            "For n_mics < 4, source localization is not possible. "
            "Detection (CNF anomaly score) still works independently."
        )

    L = config.gcc_length
    rows: list[np.ndarray] = []
    for i, j in MIC_PAIRS:
        rows.append(gcc_phat(mic_data[i], mic_data[j], config.max_delay_samples, n_fft))
    stack = np.stack(rows, axis=0)  # (6, L)
    assert stack.shape == (N_PAIRS, L), f"GCC stack shape mismatch: {stack.shape}"
    return stack


# ---------------------------------------------------------------------------
# SRP-PHAT map (physics-based spatial prior)
# ---------------------------------------------------------------------------


def srp_phat_map(
    acoustic_gcc: np.ndarray,
    mic_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    fs: float,
    c_air: float = 343.0,
) -> np.ndarray:
    """Compute a Steered-Response Power PHAT map over a 2-D candidate grid.

    The SRP map is used as a physics-based prior for the Transformer's
    cross-attention sublayer. It captures the coarse spatial hypothesis that
    purely data-driven attention can refine.

    Args:
        acoustic_gcc: float array of shape (6, L).
        mic_xy: float array of shape (4, 2) — mic (x, y) positions in metres.
        grid_x: 1D array of candidate x values (metres).
        grid_y: 1D array of candidate y values (metres).
        fs: Audio sample rate (Hz).
        c_air: Speed of sound (m/s).

    Returns:
        srp: float32 array of shape (len(grid_x), len(grid_y)).
            Arc ambiguities from the single-elevation ring will be visible.
    """
    max_delay_samples = acoustic_gcc.shape[1] // 2  # (L-1)//2
    srp = np.zeros((len(grid_x), len(grid_y)), dtype=np.float32)

    for ix, gx in enumerate(grid_x):
        for iy, gy in enumerate(grid_y):
            pos = np.array([gx, gy])
            score = 0.0
            for k, (i, j) in enumerate(MIC_PAIRS):
                d_i = float(np.linalg.norm(mic_xy[i] - pos))
                d_j = float(np.linalg.norm(mic_xy[j] - pos))
                expected_delay = (d_i - d_j) / c_air * fs
                idx = int(round(expected_delay)) + max_delay_samples
                if 0 <= idx < acoustic_gcc.shape[1]:
                    score += float(acoustic_gcc[k, idx])
            srp[ix, iy] = score

    return srp


# ---------------------------------------------------------------------------
# Shared FiLM layer (same design as detection_head.FiLM)
# ---------------------------------------------------------------------------


class FiLM(nn.Module):
    """Feature-wise Linear Modulation — injects operating-mode context c into h.

    Uses the (1 + gamma') * h + beta parameterisation so that at initialisation
    (weights ≈ 0, bias = 0) the layer is an identity, which stabilises early
    training.
    """

    def __init__(self, d_ctx: int, feature_dim: int) -> None:
        super().__init__()
        self._gen = nn.Linear(d_ctx, 2 * feature_dim)
        nn.init.normal_(self._gen.weight, std=0.01)
        nn.init.zeros_(self._gen.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self._gen(c)
        gamma_prime, beta = params.chunk(2, dim=-1)
        return (1.0 + gamma_prime) * h + beta


# ---------------------------------------------------------------------------
# Option A: LocalizationCNN
# ---------------------------------------------------------------------------


class LocalizationCNN(nn.Module):
    """1D CNN over stacked GCC-PHAT vectors for source position regression.

    Architecture:
        3 × Conv1d blocks (6→32→64→128 channels, kernel sizes 7/5/3, same padding)
        AdaptiveAvgPool1d(1) — collapse temporal dimension
        FiLM conditioning with context c
        2-layer MLP head → (x, y) in metres

    Suitable as a fast baseline when you have limited calibration data.
    Approximate parameter count: ~60 K (d_model=128, L=187).

    Args:
        config: LocalizationConfig.
    """

    def __init__(self, config: LocalizationConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self._conv1 = nn.Conv1d(N_PAIRS, 32, kernel_size=7, padding=3)
        self._conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self._conv3 = nn.Conv1d(64, d, kernel_size=3, padding=1)
        self._pool = nn.AdaptiveAvgPool1d(1)
        self._film = FiLM(config.d_ctx, d)
        self._drop = nn.Dropout(p=config.dropout)
        self._head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 2),  # (x, y)
        )

    def forward(
        self,
        gcc: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gcc: (batch, 6, L) — stacked GCC-PHAT vectors.
            c:   (batch, d_ctx) — operating-mode context from Detection Head.

        Returns:
            pos: (batch, 2) — predicted (x, y) position in metres.
        """
        x = F.relu(self._conv1(gcc))  # (batch, 32, L)
        x = F.relu(self._conv2(x))  # (batch, 64, L)
        x = F.relu(self._conv3(x))  # (batch, d_model, L)
        x = self._pool(x).squeeze(-1)  # (batch, d_model)
        x = self._film(x, c)
        x = self._drop(x)
        return self._head(x)  # (batch, 2)


# ---------------------------------------------------------------------------
# Option B: LocalizationTransformer (preferred)
# ---------------------------------------------------------------------------


class _GeometryAwarePairEmbedding(nn.Module):
    """Projects the physical geometry of each mic pair into d_model space.

    For pair (i, j) with known mic positions mic_xy:
        geom_feat = [x_midpoint, y_midpoint, pair_length, pair_angle]  (4-D)
    A linear projection maps this to d_model, so the Transformer's attention
    mechanism can exploit the physical layout when deciding which pairs to trust.

    Args:
        mic_xy: (4, 2) float Tensor — mic positions in metres, fixed at build.
        d_model: Embedding dimension.
    """

    def __init__(self, mic_xy: torch.Tensor, d_model: int) -> None:
        super().__init__()
        # Compute and buffer the 4-D geometry feature for each of 6 pairs
        geom = self._make_geom_features(mic_xy)  # (6, 4)
        self.register_buffer("_geom", geom)
        self._proj = nn.Linear(4, d_model)

    @staticmethod
    def _make_geom_features(mic_xy: torch.Tensor) -> torch.Tensor:
        rows: list[list[float]] = []
        for i, j in MIC_PAIRS:
            pi = mic_xy[i]
            pj = mic_xy[j]
            mid_x = float((pi[0] + pj[0]) / 2)
            mid_y = float((pi[1] + pj[1]) / 2)
            length = float(torch.norm(pj - pi).item())
            angle = float(
                math.atan2(
                    float((pj[1] - pi[1]).item()),
                    float((pj[0] - pi[0]).item()),
                )
            )
            rows.append([mid_x, mid_y, length, angle])
        return torch.tensor(rows, dtype=torch.float32)  # (6, 4)

    def forward(self) -> torch.Tensor:
        """Returns pair embeddings of shape (6, d_model)."""
        return self._proj(self._geom)  # type: ignore[arg-type]


class LocalizationTransformer(nn.Module):
    """Transformer encoder over 6 GCC-PHAT tokens for source position regression.

    Design rationale:
        - Treats each of the 6 GCC-PHAT vectors as one token (sequence length = 6).
        - Geometry-aware pair embeddings (midpoint, length, angle → Linear → d_model)
          are added at the token level, so attention can exploit physical layout.
        - Self-attention learns which pairs are reliable for each source position,
          naturally handling reverberation-corrupted pairs without explicit masking.
        - FiLM conditioning on c (operating-mode context) adapts spatial priors
          per mode, applied after the final encoder layer.
        - Optional cross-attention to a flattened SRP-PHAT map (physics prior):
          Q = Transformer output, K = V = SRP grid tokens with (x, y) PE.
          Grounds learned features in the beamforming prior.

    Args:
        config: LocalizationConfig.
        mic_xy: (4, 2) float Tensor — mic x/y positions in metres. Required here
            rather than at forward time so pair embeddings are fixed at build.
        srp_grid_x: Optional 1D array of candidate x positions for SRP prior.
        srp_grid_y: Optional 1D array of candidate y positions for SRP prior.
            Both required when config.use_srp_cross_attn = True.
    """

    def __init__(
        self,
        config: LocalizationConfig,
        mic_xy: torch.Tensor,
        srp_grid_x: np.ndarray | None = None,
        srp_grid_y: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # --- GCC-PHAT token projector: maps each (L,) vector → d_model ---
        self._gcc_proj = nn.Linear(config.gcc_length, d)

        # --- Geometry-aware pair positional embedding ---
        self._pair_pe = _GeometryAwarePairEmbedding(mic_xy, d)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            batch_first=True,  # (batch, seq, d_model)
            norm_first=True,  # pre-LN for training stability
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # --- Optional cross-attention to SRP-PHAT prior map ---
        self._use_srp = config.use_srp_cross_attn
        if self._use_srp:
            if srp_grid_x is None or srp_grid_y is None:
                raise ValueError(
                    "srp_grid_x and srp_grid_y are required when use_srp_cross_attn=True"
                )
            n_grid = len(srp_grid_x) * len(srp_grid_y)
            self._srp_feat_proj = nn.Linear(1, d)  # SRP scalar → d_model
            self._srp_xy_proj = nn.Linear(2, d)  # grid (x,y) PE
            self._cross_attn = nn.MultiheadAttention(
                embed_dim=d,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            # Pre-compute and register grid coordinates as buffer
            gx = torch.tensor(srp_grid_x, dtype=torch.float32)
            gy = torch.tensor(srp_grid_y, dtype=torch.float32)
            grid_xy = torch.stack(
                torch.meshgrid(gx, gy, indexing="ij"), dim=-1
            ).reshape(
                -1, 2
            )  # (n_grid, 2)
            self.register_buffer("_srp_grid_xy", grid_xy)

        # --- FiLM: adapts pooled representation to operating mode ---
        self._film = FiLM(config.d_ctx, d)
        self._drop = nn.Dropout(p=config.dropout)

        # --- Regression head: d_model → (x, y) in metres ---
        self._head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2),
        )

    def forward(
        self,
        gcc: torch.Tensor,
        c: torch.Tensor,
        srp_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            gcc: (batch, 6, L) — stacked GCC-PHAT vectors.
            c:   (batch, d_ctx) — operating-mode context from Detection Head.
            srp_map: (batch, n_grid_x, n_grid_y) SRP power map. Required when
                config.use_srp_cross_attn = True, ignored otherwise.

        Returns:
            pos: (batch, 2) — predicted (x, y) position in metres.
        """
        # Project each GCC-PHAT vector to d_model and add pair embedding
        tokens = self._gcc_proj(gcc)  # (batch, 6, d_model)
        pair_pe = self._pair_pe()  # (6, d_model)
        tokens = tokens + pair_pe.unsqueeze(0)  # broadcast over batch

        # Self-attention: learn inter-pair reliability without explicit masking
        tokens = self._encoder(tokens)  # (batch, 6, d_model)

        # Optional cross-attention to SRP-PHAT prior
        if self._use_srp:
            if srp_map is None:
                raise ValueError("srp_map required when use_srp_cross_attn=True")
            batch = gcc.shape[0]
            srp_flat = srp_map.reshape(batch, -1, 1)  # (batch, n_grid, 1)
            srp_feat = self._srp_feat_proj(srp_flat)  # (batch, n_grid, d_model)
            srp_pe = self._srp_xy_proj(
                self._srp_grid_xy
            )  # (n_grid, d_model)  # type: ignore[arg-type]
            kv = srp_feat + srp_pe.unsqueeze(0)  # (batch, n_grid, d_model)
            tokens, _ = self._cross_attn(tokens, kv, kv)  # (batch, 6, d_model)

        # Pool over the 6 pair tokens (mean) → single representation
        x = tokens.mean(dim=1)  # (batch, d_model)

        # FiLM: inject operating-mode context
        x = self._film(x, c)
        x = self._drop(x)

        return self._head(x)  # (batch, 2)


# ---------------------------------------------------------------------------
# Training losses
# ---------------------------------------------------------------------------


def supervised_localization_loss(
    pred_xy: torch.Tensor,
    true_xy: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted and labelled (x, y) positions.

    Use when calibration recordings with known source positions are available.
    Even 5–10 recordings per position are sufficient given the informative
    GCC-PHAT features.

    Args:
        pred_xy: (batch, 2) — model output.
        true_xy: (batch, 2) — ground-truth source positions in metres.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred_xy, true_xy)


def geometric_consistency_loss(
    pred_xy: torch.Tensor,
    acoustic_gcc: torch.Tensor,
    mic_xy: torch.Tensor,
    max_delay_samples: int,
    c_air: float = 343.0,
    fs: float = 16000.0,
) -> torch.Tensor:
    """Self-supervised loss: predicted TDOA must match the GCC-PHAT peak.

    Use when no labelled source positions exist. The predicted (x, y) position
    implies a unique TDOA for each mic pair through physics. This loss penalises
    disagreement between the implied TDOA and the measured GCC-PHAT peak.

    Derivation:
        TDOA_ij = (||pred − mic_i|| − ||pred − mic_j||) / c_air * fs   [samples]
        Measured peak from GCC-PHAT: argmax(gcc[k]) − max_delay_samples

    Args:
        pred_xy: (batch, 2) — predicted position in metres.
        acoustic_gcc: (batch, 6, L) — GCC-PHAT stack.
        mic_xy: (4, 2) Tensor — mic positions in metres.
        max_delay_samples: Half-width of GCC vector.
        c_air: Speed of sound (m/s).
        fs: Audio sample rate (Hz).

    Returns:
        Scalar mean MSE over all 6 pairs.
    """
    total = torch.zeros(1, device=pred_xy.device, dtype=pred_xy.dtype)
    for k, (i, j) in enumerate(MIC_PAIRS):
        pi = mic_xy[i]  # (2,)
        pj = mic_xy[j]  # (2,)

        # Physics-implied TDOA from predicted position
        d_i = torch.norm(pred_xy - pi.unsqueeze(0), dim=-1)  # (batch,)
        d_j = torch.norm(pred_xy - pj.unsqueeze(0), dim=-1)
        tdoa_expected = (d_i - d_j) / c_air * fs  # (batch,) in samples

        # Measured TDOA from GCC-PHAT peak (differentiable soft-argmax)
        gcc_k = acoustic_gcc[:, k, :]  # (batch, L)
        weights = torch.softmax(gcc_k * 10.0, dim=-1)  # sharpened softmax
        lags = torch.arange(gcc_k.shape[-1], device=gcc_k.device, dtype=gcc_k.dtype)
        tdoa_measured = (weights * lags).sum(dim=-1) - float(max_delay_samples)

        total = total + F.mse_loss(tdoa_expected, tdoa_measured)

    return total / float(N_PAIRS)


# ---------------------------------------------------------------------------
# Convenience: extract GCC-PHAT from DataSegment (no FeatureFrame needed)
# ---------------------------------------------------------------------------


def extract_gcc_from_segment(
    mic_data: np.ndarray,
    config: LocalizationConfig,
) -> torch.Tensor:
    """Compute GCC-PHAT stack from raw mic_data and return as a (1, 6, L) Tensor.

    Convenience wrapper for single-sample inference. For batch inference
    (e.g. during training), call compute_gcc_phat_stack() directly and stack.

    Args:
        mic_data: (n_mics, n_samples) float array — DataSegment.mic_data.
        config: LocalizationConfig.

    Returns:
        gcc_tensor: float32 Tensor of shape (1, 6, L), ready to pass to
            LocalizationCNN.forward() or LocalizationTransformer.forward().
    """
    gcc_np = compute_gcc_phat_stack(mic_data, config)
    return torch.from_numpy(gcc_np).unsqueeze(0)  # (1, 6, L)
