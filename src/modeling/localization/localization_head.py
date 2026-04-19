"""Localization Head — LocalizationCNNS2 for the bench-top second test dataset.

Transformer-based 3-D source localizer using GCC-PHAT features from a 5-mic array.
Takes per-window GCC stacks from 10 mic pairs and an SRP-PHAT prior as input,
and outputs a 3-D position estimate (x, y, z) in metres.

Public API:
  LocalizationCNNS2              — main model class
  supervised_localization_loss_s2 — supervised regression loss
  geometric_consistency_loss_s2  — physics-based self-supervised loss
  compute_gcc_stack_s2_multiwindow — multi-window GCC-PHAT preprocessing
  gcc_phat                       — single-pair GCC-PHAT utility

Geometry constants (second dataset, bench-top prototype):
  MIC_PAIRS_S2, N_PAIRS_S2, S2_MIC_XYZ, S2_VIB_XYZ, S2_FAULT_POSITIONS_M

Geometry constants (ROW II turbine, for thesis reference):
  TURBINE, FREQUENCIES_HZ, SENSOR_XYZ, MIC_UPPER_XY, MIC_ALL_XYZ, ACCEL_XYZ
"""

from __future__ import annotations

import math
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


# ==============================================================================
# Second test dataset — LocalizationCNNS2
# ==============================================================================
# For the bench-top prototype: 5 mics, 10 pairs, 3-D output (x, y, z).
# No FiLM context required — standalone module, does not depend on DetectionHead.
#
# Training modes:
#   Supervised  : supervised_localization_loss_s2  (ground-truth position labels)
#   Self-supervised: geometric_consistency_loss_s2 (no labels required, physics-only)
#   Combined    : supervised + lambda * geometric_consistency (recommended)
# ==============================================================================

# Derived constants for the second dataset
_S2_MAX_DELAY_SAMPLES: int = int(_S2_MAX_MIC_DIST_M / 343.0 * 16000)
_S2_GCC_LENGTH: int = 2 * _S2_MAX_DELAY_SAMPLES + 1


def compute_gcc_stack_s2_multiwindow(
    mic_data: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    c: float = 343.0,
) -> np.ndarray:
    """Multi-window averaged GCC-PHAT stack for the 5-mic second dataset.

    Short-window PHAT normalization followed by time-averaging reduces
    reverberation noise compared to computing GCC on the full signal once.
    Each window gets its own PHAT whitening, so transient artefacts stay local.

    Args:
        mic_data: (5, N_samples) float array — raw mic waveforms.
        fs: Sample rate in Hz.
        window_s: Analysis window length in seconds.
        hop_s: Hop between successive windows in seconds.
        c: Speed of sound in m/s.

    Returns:
        gcc_avg: float32 array of shape (10, L_s2).
    """
    window_samples = int(window_s * fs)
    hop_samples = max(1, int(hop_s * fs))
    max_delay = _S2_MAX_DELAY_SAMPLES
    n_samples = mic_data.shape[1]

    stacks: list[np.ndarray] = []
    start = 0
    while start + window_samples <= n_samples:
        frame = mic_data[:, start : start + window_samples]
        rows = [gcc_phat(frame[i], frame[j], max_delay) for i, j in MIC_PAIRS_S2]
        stacks.append(np.stack(rows, axis=0))  # (10, L)
        start += hop_samples

    if not stacks:
        # Recording shorter than one window — process full signal as-is
        rows = [gcc_phat(mic_data[i], mic_data[j], max_delay) for i, j in MIC_PAIRS_S2]
        return np.stack(rows, axis=0)

    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)  # (10, L)


# ---------------------------------------------------------------------------
# Geometry-aware pair embeddings for 10 pairs in 3-D space
# ---------------------------------------------------------------------------


class _GeometryAwarePairEmbeddingS2(nn.Module):
    """Encodes physical geometry of each of 10 mic pairs into d_model space.

    Features per pair: [mid_x, mid_y, mid_z, length, angle_xy, angle_z]  (6-D).
    A linear projection maps to d_model so the Transformer attention can
    exploit the 3-D physical layout of the second dataset's sensor array.

    Args:
        mic_xyz: (5, 3) float Tensor — mic positions in metres (S2_MIC_XYZ).
        d_model: Embedding dimension.
    """

    def __init__(self, mic_xyz: torch.Tensor, d_model: int) -> None:
        super().__init__()
        geom = self._make_geom_features(mic_xyz)  # (10, 6)
        self.register_buffer("_geom", geom)
        self._proj = nn.Linear(6, d_model)

    @staticmethod
    def _make_geom_features(mic_xyz: torch.Tensor) -> torch.Tensor:
        rows: list[list[float]] = []
        for i, j in MIC_PAIRS_S2:
            pi = mic_xyz[i]  # (3,)
            pj = mic_xyz[j]
            mid_x = float(((pi[0] + pj[0]) / 2).item())
            mid_y = float(((pi[1] + pj[1]) / 2).item())
            mid_z = float(((pi[2] + pj[2]) / 2).item())
            dx = float((pj[0] - pi[0]).item())
            dy = float((pj[1] - pi[1]).item())
            dz = float((pj[2] - pi[2]).item())
            length = float(torch.norm(pj - pi).item())
            angle_xy = math.atan2(dy, dx)
            angle_z = math.atan2(dz, math.sqrt(dx**2 + dy**2))
            rows.append([mid_x, mid_y, mid_z, length, angle_xy, angle_z])
        return torch.tensor(rows, dtype=torch.float32)  # (10, 6)

    def forward(self) -> torch.Tensor:
        """Returns pair embeddings of shape (10, d_model)."""
        return self._proj(self._geom)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LocalizationCNNS2: the combined SRP+neural estimator for the second dataset
# ---------------------------------------------------------------------------


class LocalizationCNNS2(nn.Module):
    """Transformer-based 3-D source localizer for the second bench-top dataset.

    Architecture:
        - Projects each of 10 GCC-PHAT vectors (L_s2,) → d_model
        - Adds geometry-aware pair embeddings (3-D midpoint / length / angles)
        - Concatenates SRP-PHAT peak position as an explicit spatial prior token
        - 2-layer Transformer encoder over 11 tokens (10 pairs + 1 SRP prior)
        - Mean-pool → MLP head → (x, y, z) in metres

    The SRP-PHAT prior token anchors the neural output close to the physics-
    based estimate; the self-attention then refines it by weighting pairs
    selectively based on the geometry and the observed GCC peaks.

    Args:
        d_model: Internal feature dimension (must be divisible by n_heads=4).
        dropout: Dropout rate.
    """

    # Fixed mic geometry for second dataset
    _MIC_XYZ: np.ndarray = S2_MIC_XYZ  # (5, 3)

    def __init__(
        self,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # L_s2 is determined at runtime from the gcc input; store for reference
        self._max_delay_samples = _S2_MAX_DELAY_SAMPLES

        mic_t = torch.tensor(self._MIC_XYZ, dtype=torch.float32)
        self._pair_pe = _GeometryAwarePairEmbeddingS2(mic_t, d_model)

        # Project each GCC-PHAT vector (L_s2,) → d_model
        self._gcc_proj = nn.Linear(_S2_GCC_LENGTH, d_model)

        # SRP prior token: encode the 3-D SRP-PHAT peak position → d_model
        self._srp_prior_proj = nn.Linear(3, d_model)

        # Transformer over 10 pair tokens + 1 SRP prior token = 11 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self._drop = nn.Dropout(p=dropout)

        # Regression head: d_model → (x, y, z) in metres
        self._head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, 3),
        )

    def forward(
        self,
        gcc: torch.Tensor,
        srp_prior_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gcc:            (batch, 10, L_s2) — multi-window averaged GCC-PHAT stack.
            srp_prior_xyz:  (batch, 3) — SRP-PHAT peak position in metres.

        Returns:
            pos: (batch, 3) — refined (x, y, z) position in metres.
        """
        # GCC tokens: (batch, 10, L_s2) → (batch, 10, d_model) with pair PE
        tokens = self._gcc_proj(gcc)  # (batch, 10, d_model)
        pair_pe = self._pair_pe()  # (10, d_model)
        tokens = tokens + pair_pe.unsqueeze(0)  # broadcast over batch

        # SRP prior token: (batch, 3) → (batch, 1, d_model)
        srp_token = self._srp_prior_proj(srp_prior_xyz).unsqueeze(1)  # (batch, 1, d)

        # Concatenate: 10 pair tokens + 1 SRP prior = 11 tokens
        all_tokens = torch.cat([tokens, srp_token], dim=1)  # (batch, 11, d_model)

        # Self-attention: pairs attend to each other AND to the SRP prior
        out = self._encoder(all_tokens)  # (batch, 11, d_model)

        # Pool only over the 10 pair tokens (ignore the SRP prior token)
        x = out[:, :10, :].mean(dim=1)  # (batch, d_model)
        x = self._drop(x)
        return self._head(x)  # (batch, 3)


# ---------------------------------------------------------------------------
# Losses for LocalizationCNNS2
# ---------------------------------------------------------------------------


def supervised_localization_loss_s2(
    pred_xyz: torch.Tensor,
    true_xyz: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for 3-D position regression (second dataset).

    Args:
        pred_xyz: (batch, 3) — predicted (x, y, z) in metres.
        true_xyz: (batch, 3) — ground-truth (x, y, z) in metres.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred_xyz, true_xyz)


def geometric_consistency_loss_s2(
    pred_xyz: torch.Tensor,
    gcc_stack: torch.Tensor,
    mic_xyz: torch.Tensor,
    max_delay_samples: int,
    c_air: float = 343.0,
    fs: float = 16000.0,
) -> torch.Tensor:
    """Self-supervised 3-D consistency: predicted position → implied TDOAs must
    match the GCC-PHAT peaks across all 10 mic pairs.

    No ground-truth position labels required. Physics-derived TDOAs for the
    predicted (x,y,z) are matched to the measured GCC-PHAT soft peaks via
    differentiable soft-argmax, enabling gradient-based training without labels.

    Args:
        pred_xyz:          (batch, 3) — predicted (x, y, z) in metres.
        gcc_stack:         (batch, 10, L) — GCC-PHAT stack (10 pairs).
        mic_xyz:           (5, 3) float Tensor — mic positions in metres.
        max_delay_samples: Half-width of the GCC vector.
        c_air:             Speed of sound (m/s).
        fs:                Sample rate (Hz).

    Returns:
        Scalar mean MSE over all 10 pairs.
    """
    total = torch.zeros(1, device=pred_xyz.device, dtype=pred_xyz.dtype)
    for k, (i, j) in enumerate(MIC_PAIRS_S2):
        pi = mic_xyz[i]  # (3,)
        pj = mic_xyz[j]  # (3,)

        d_i = torch.norm(pred_xyz - pi.unsqueeze(0), dim=-1)  # (batch,)
        d_j = torch.norm(pred_xyz - pj.unsqueeze(0), dim=-1)
        tdoa_expected = (d_i - d_j) / c_air * fs  # (batch,) in samples

        gcc_k = gcc_stack[:, k, :]  # (batch, L)
        weights = torch.softmax(gcc_k * 10.0, dim=-1)  # sharpened soft-argmax
        lags = torch.arange(gcc_k.shape[-1], device=gcc_k.device, dtype=gcc_k.dtype)
        tdoa_measured = (weights * lags).sum(dim=-1) - float(max_delay_samples)

        total = total + F.mse_loss(tdoa_expected, tdoa_measured)

    return total / float(N_PAIRS_S2)
