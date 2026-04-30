"""Layer 3 — RMS Feature Extraction.

Produces a (T, ~40) physics-motivated feature matrix from the 6 active vibration
channels (RmsGenVib180X/Y/Z, RmsTurVib180X/Y/Z).

All rolling operations use vectorized prefix-sum cumsum — no Python loops over T.
Handles T = 604,800 (7-day campaign) well within 30 seconds on CPU.

Feature layout
--------------
0–17  log1p(rolling_mean(x², w)) for 6 ch × windows {5, 60, 300} s
18–20 log(E_gen_axis / (E_tur_axis + ε)) per axis X/Y/Z at 60 s
21    Δ₁  of log(E_total_60s)
22    Δ₁₀ of log(E_total_60s)
23    Stationarity: rolling_var(rolling_mean(E_60s, 300s))
24–29 Per-channel kurtosis at 60 s
30–32 3 leading PCs of 60 s sliding covariance across 6 channels
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Expected active channel order after dropping dead sensors
ACTIVE_CHANNEL_NAMES = [
    "RmsGenVib180X",
    "RmsGenVib180Y",
    "RmsGenVib180Z",
    "RmsTurVib180X",
    "RmsTurVib180Y",
    "RmsTurVib180Z",
]
N_ACTIVE = len(ACTIVE_CHANNEL_NAMES)   # 6

# Indices within ACTIVE_CHANNEL_NAMES
_GEN_AXES = [0, 1, 2]  # X, Y, Z for Generator
_TUR_AXES = [3, 4, 5]  # X, Y, Z for Turbine

FEATURE_DIM = 33  # 18 + 3 + 2 + 1 + 6 + 3


# ---------------------------------------------------------------------------
# Prefix-sum vectorized rolling operations
# ---------------------------------------------------------------------------


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of x (T,) using prefix sums. O(T), no loops."""
    T = len(x)
    cs = np.zeros(T + 1, dtype=np.float64)
    cs[1:] = np.cumsum(x.astype(np.float64))
    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = t_arr - t0_arr + 1
    return ((cs[t_arr + 1] - cs[t0_arr]) / n).astype(np.float32)


def _rolling_mean_2d(X: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling mean of (T, C) — applies _rolling_mean per column."""
    T, C = X.shape
    cs = np.zeros((T + 1, C), dtype=np.float64)
    cs[1:] = np.cumsum(X.astype(np.float64), axis=0)
    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = (t_arr - t0_arr + 1).astype(np.float64)[:, None]
    return ((cs[t_arr + 1] - cs[t0_arr]) / n).astype(np.float32)


def _rolling_var_scalar(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling variance of a 1-D array using prefix sums."""
    T = len(x)
    x64 = x.astype(np.float64)
    cs1 = np.zeros(T + 1)
    cs2 = np.zeros(T + 1)
    cs1[1:] = np.cumsum(x64)
    cs2[1:] = np.cumsum(x64 ** 2)
    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = (t_arr - t0_arr + 1).astype(np.float64)
    mean = (cs1[t_arr + 1] - cs1[t0_arr]) / n
    mean2 = (cs2[t_arr + 1] - cs2[t0_arr]) / n
    return np.maximum(0.0, mean2 - mean ** 2).astype(np.float32)


def _rolling_kurtosis(x: np.ndarray, window: int) -> np.ndarray:
    """Excess kurtosis via 4th central moment. Vectorized with prefix sums."""
    T = len(x)
    x64 = x.astype(np.float64)
    cs = [np.zeros(T + 1) for _ in range(5)]
    for k in range(1, 5):
        cs[k][1:] = np.cumsum(x64 ** k)

    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = (t_arr - t0_arr + 1).astype(np.float64)

    def _sum(k: int) -> np.ndarray:
        return cs[k][t_arr + 1] - cs[k][t0_arr]

    m1 = _sum(1) / n
    m2 = _sum(2) / n - m1 ** 2
    m4_raw = (
        _sum(4) / n
        - 4 * m1 * _sum(3) / n
        + 6 * m1 ** 2 * _sum(2) / n
        - 3 * m1 ** 4
    )
    var_safe = np.maximum(m2, 1e-30)
    excess = m4_raw / var_safe ** 2 - 3.0
    # Suppress kurtosis when variance is near-zero (flat/stopped signal) — the
    # formula amplifies measurement noise to astronomically large values at m2→0.
    excess = np.where(m2 < 1e-4, 0.0, excess)
    return np.clip(excess, -50.0, 200.0).astype(np.float32)


def _sliding_cov_pcs(X: np.ndarray, window: int, n_pcs: int = 3) -> np.ndarray:
    """Approximate leading PCs of the 60-s sliding covariance across 6 channels.

    Computes the rolling covariance matrix (6×6) at each timestep using prefix
    sums, then extracts the leading `n_pcs` eigenvector projections.

    For efficiency, the eigenvector is computed on the *global* covariance (fit on
    training data), and the per-timestep feature is the projection of the rolling
    covariance onto those static eigenvectors (the Frobenius dot-product).
    This avoids a per-timestep eigendecomposition (O(T × C³)) and is sufficient
    for downstream classification.
    """
    T, C = X.shape
    X64 = X.astype(np.float64)

    # Build prefix-sum arrays for cross-products
    cov_cs = np.zeros((T + 1, C, C), dtype=np.float64)
    mean_cs = np.zeros((T + 1, C), dtype=np.float64)
    for i in range(T):
        cov_cs[i + 1] = cov_cs[i] + np.outer(X64[i], X64[i])
        mean_cs[i + 1] = mean_cs[i] + X64[i]

    t_arr = np.arange(T, dtype=np.int64)
    t0_arr = np.maximum(0, t_arr - window + 1)
    n = (t_arr - t0_arr + 1).astype(np.float64)

    # Rolling covariance matrix at each t: E[xx^T] - E[x]E[x]^T
    sum_xx = cov_cs[t_arr + 1] - cov_cs[t0_arr]        # (T, C, C)
    sum_x  = mean_cs[t_arr + 1] - mean_cs[t0_arr]      # (T, C)
    mean   = sum_x / n[:, None]                          # (T, C)
    cov_t  = sum_xx / n[:, None, None] - mean[:, :, None] * mean[:, None, :]  # (T, C, C)

    # Global covariance eigenvectors (computed once on the full array)
    global_cov = np.cov(X64.T) + np.eye(C) * 1e-9
    _, evecs = np.linalg.eigh(global_cov)
    top_evecs = evecs[:, -n_pcs:]  # (C, n_pcs), leading eigenvectors

    # Projection: for each t, project cov_t onto each eigenvector
    # proj[t, k] = evec_k^T @ cov_t[t] @ evec_k  (scalar)
    out = np.zeros((T, n_pcs), dtype=np.float32)
    for k in range(n_pcs):
        v = top_evecs[:, k]  # (C,)
        out[:, k] = np.einsum("tij,i,j->t", cov_t, v, v).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Head residualization and robust scaling (imported from rms_temporal.py)
# ---------------------------------------------------------------------------


def _apply_head_residualization(
    X_active: np.ndarray,
    net_head: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Apply global polynomial head-residualization.

    coeffs: (N_active, 5) — one row per channel, basis [h², h, t², t, 1].
    """
    T = X_active.shape[0]
    h = (net_head - net_head.mean()) / (net_head.std() + 1e-9)
    t_norm = np.linspace(0, 1, T)
    basis = np.column_stack([h ** 2, h, t_norm ** 2, t_norm, np.ones(T)])  # (T, 5)
    fit = basis @ coeffs.T  # (T, N_active)
    return (X_active - fit).astype(np.float32)


def _apply_robust_scale(
    X: np.ndarray,
    median: np.ndarray,
    iqr: np.ndarray,
) -> np.ndarray:
    return ((X - median) / np.maximum(iqr, 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature transform dataclass
# ---------------------------------------------------------------------------


@dataclass
class RMSFeatureTransform:
    """Fitted transform parameters — no PCA."""
    active_channel_indices: list[int]     # indices in original 16-channel array
    active_channel_names: list[str]       # 6 names
    head_poly_coeffs: np.ndarray          # (6, 5)
    robust_median: np.ndarray             # (6,)
    robust_iqr: np.ndarray               # (6,)
    upper_head_channel: str
    lower_level_channel: str
    windows: tuple[int, ...] = (5, 60, 300)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            active_channel_indices=np.array(self.active_channel_indices),
            active_channel_names=np.array(self.active_channel_names),
            head_poly_coeffs=self.head_poly_coeffs,
            robust_median=self.robust_median,
            robust_iqr=self.robust_iqr,
            upper_head_channel=np.array(self.upper_head_channel),
            lower_level_channel=np.array(self.lower_level_channel),
            windows=np.array(self.windows),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RMSFeatureTransform":
        d = np.load(path, allow_pickle=False)
        return cls(
            active_channel_indices=d["active_channel_indices"].tolist(),
            active_channel_names=d["active_channel_names"].tolist(),
            head_poly_coeffs=d["head_poly_coeffs"],
            robust_median=d["robust_median"],
            robust_iqr=d["robust_iqr"],
            upper_head_channel=str(d["upper_head_channel"]),
            lower_level_channel=str(d["lower_level_channel"]),
            windows=tuple(int(w) for w in d["windows"]),
        )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_rms_transform(
    rms: np.ndarray,
    channel_names_rms: list[str],
    net_head: np.ndarray,
    *,
    forced_drop_channels: list[str],
    upper_head_channel: str = "Oberwasserpegel",
    lower_level_channel: str = "UW_Pegel_Rodund",
    head_poly_degree: int = 2,
    train_fraction: float = 0.75,
    windows: tuple[int, ...] = (5, 60, 300),
) -> "RMSFeatureTransform":
    """Fit head-residualization polynomial + robust scaler on training slice."""
    T = rms.shape[0]
    T_train = int(T * train_fraction)

    # Resolve active channel indices
    drop_set = set(forced_drop_channels)
    active_idx = [i for i, n in enumerate(channel_names_rms) if n not in drop_set]
    active_names = [channel_names_rms[i] for i in active_idx]
    X_active = rms[:T_train, active_idx].astype(np.float64)

    # Head residualization: global polynomial [h², h, t², t, 1] per channel
    h = net_head[:T_train]
    h_norm = (h - h.mean()) / (h.std() + 1e-9)
    t_norm = np.linspace(0, 1, T_train)
    basis = np.column_stack([h_norm ** 2, h_norm, t_norm ** 2, t_norm, np.ones(T_train)])
    coeffs = np.linalg.lstsq(basis, X_active, rcond=None)[0].T  # (n_active, 5)

    X_resid = X_active - basis @ coeffs.T

    # Robust scaling
    q25 = np.percentile(X_resid, 25, axis=0)
    q75 = np.percentile(X_resid, 75, axis=0)
    median = np.median(X_resid, axis=0)
    iqr = q75 - q25

    return RMSFeatureTransform(
        active_channel_indices=active_idx,
        active_channel_names=active_names,
        head_poly_coeffs=coeffs.astype(np.float64),
        robust_median=median.astype(np.float64),
        robust_iqr=iqr.astype(np.float64),
        upper_head_channel=upper_head_channel,
        lower_level_channel=lower_level_channel,
        windows=windows,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_rms_features(
    rms: np.ndarray,
    net_head: np.ndarray,
    transform: RMSFeatureTransform,
    *,
    cavitation_kurtosis_threshold: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply transform and compute the (T, 33) feature matrix.

    Returns
    -------
    X_feat : (T, 33) float32
    cav_mask : (T,) bool — cavitation-flagged timesteps (kurtosis > threshold)
    """
    X_active = rms[:, transform.active_channel_indices].astype(np.float64)
    T, C = X_active.shape
    h = net_head.astype(np.float64)

    X_resid = _apply_head_residualization(
        X_active.astype(np.float32), h.astype(np.float32), transform.head_poly_coeffs
    )
    X_scaled = _apply_robust_scale(X_resid, transform.robust_median, transform.robust_iqr)

    windows = transform.windows  # (5, 60, 300)
    w_60 = windows[1]

    # --- Features 0–17: log-energy at 3 scales per channel ---
    log_energy = np.zeros((T, C, len(windows)), dtype=np.float32)
    for wi, w in enumerate(windows):
        E_w = _rolling_mean_2d(X_scaled ** 2, w)  # (T, C)
        log_energy[:, :, wi] = np.log1p(E_w)
    # Reshape to (T, 18): iterate window first, then channel
    feat_log_energy = log_energy.reshape(T, C * len(windows))  # (T, 18)

    # --- Features 18–20: Gen/Tur energy ratio per axis at 60 s ---
    E_60 = np.exp(log_energy[:, :, 1]) - 1.0  # (T, 6) — undo log1p
    eps = 1e-9
    feat_ratio = np.zeros((T, 3), dtype=np.float32)
    for ax in range(3):
        ratio_raw = np.log(E_60[:, _GEN_AXES[ax]] / (E_60[:, _TUR_AXES[ax]] + eps))
        feat_ratio[:, ax] = np.clip(ratio_raw, -10.0, 10.0)

    # --- Features 21–22: energy rate of change ---
    E_total_60 = np.log1p(np.sum(E_60, axis=1))  # (T,) scalar log-energy
    delta1  = np.diff(E_total_60, prepend=E_total_60[0]).astype(np.float32)
    delta10 = np.diff(E_total_60, n=1, prepend=np.full(1, E_total_60[0]))
    # 10-step diff: use shift by 10
    delta10 = np.zeros(T, dtype=np.float32)
    delta10[10:] = E_total_60[10:] - E_total_60[:-10]
    delta10[:10] = 0.0

    # --- Feature 23: stationarity score ---
    mean_60 = _rolling_mean(E_total_60, w_60)
    stationarity = _rolling_var_scalar(mean_60.astype(np.float64), windows[2])

    # --- Features 24–29: per-channel kurtosis at 60 s ---
    feat_kurtosis = np.zeros((T, C), dtype=np.float32)
    for c in range(C):
        feat_kurtosis[:, c] = _rolling_kurtosis(X_scaled[:, c].astype(np.float64), w_60)

    # --- Features 30–32: leading PCs of 60 s sliding covariance ---
    feat_pcs = _sliding_cov_pcs(X_scaled, w_60, n_pcs=3)

    # Cavitation mask: kurtosis > threshold on any channel
    cav_mask = np.any(feat_kurtosis > cavitation_kurtosis_threshold, axis=1)

    # Assemble
    X_feat = np.concatenate([
        feat_log_energy,     # (T, 18)
        feat_ratio,          # (T, 3)
        delta1[:, None],     # (T, 1)
        delta10[:, None],    # (T, 1)
        stationarity[:, None],  # (T, 1)
        feat_kurtosis,       # (T, 6)
        feat_pcs,            # (T, 3)
    ], axis=1).astype(np.float32)

    # Hard clip to keep features in a bounded range for downstream models.
    # Kurtosis and ratio features are already bounded above, but this is a
    # belt-and-suspenders guard against any remaining outliers.
    X_feat = np.clip(X_feat, -50.0, 200.0)

    return X_feat, cav_mask
