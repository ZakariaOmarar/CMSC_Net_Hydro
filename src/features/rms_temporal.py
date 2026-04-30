"""Multi-scale temporal feature engineering for Illwerke RMS streams.

Pipeline (applied in order):
  1. Channel selection              — drop forced_drop_channels (uninstalled sensors)
  2. Net-head residualization       — global polynomial; removes slow head drift
  3. Cavitation masking             — kurtosis-based impulse flags
  4. RobustScaler                   — median/IQR normalisation
  5. Multi-scale rolling statistics — τ ∈ {10, 60, 300, 1800} s
  6. Temporal derivatives           — Δ₁x, Δ₁₀x
  7. Cross-sensor correlations      — 60-s rolling Pearson per sensor pair
  8. Stationarity score             — scalar S[t] ∈ [0,1]
  9. PCA                            — retain 95 % variance

All fit parameters are stored in ``FeatureTransform`` and saved to a .npz
artifact so that inference uses identical transforms without refitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.polynomial import polynomial as npoly


# ---------------------------------------------------------------------------
# Public transform artifact
# ---------------------------------------------------------------------------


@dataclass
class FeatureTransform:
    """All fitted parameters needed to reproduce the feature matrix."""

    keep_mask: np.ndarray
    """(C_raw,) bool — True for channels that are NOT flat."""
    head_poly_coeffs: np.ndarray
    """(C_kept, 5) float64 — one row per kept channel; basis = [h², h, t², t, 1]."""
    robust_median: np.ndarray
    """(C_kept,) float64 — per-channel median after head residualization."""
    robust_iqr: np.ndarray
    """(C_kept,) float64 — per-channel IQR (set to 1.0 where < 1e-6)."""
    pca_components: np.ndarray
    """(n_components, n_features_total) float64 — principal axes."""
    pca_mean: np.ndarray
    """(n_features_total,) float64 — feature mean before PCA."""
    pca_explained_variance_ratio: np.ndarray
    """(n_components,) float64."""
    n_kept_channels: int
    cavitation_kurtosis_threshold: float = 8.0
    stationarity_window: int = 300
    rolling_windows: tuple = (10, 60, 300, 1800)
    corr_window: int = 60

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            keep_mask=self.keep_mask,
            head_poly_coeffs=self.head_poly_coeffs,
            robust_median=self.robust_median,
            robust_iqr=self.robust_iqr,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            pca_explained_variance_ratio=self.pca_explained_variance_ratio,
            n_kept_channels=np.array(self.n_kept_channels),
            cavitation_kurtosis_threshold=np.array(self.cavitation_kurtosis_threshold),
            stationarity_window=np.array(self.stationarity_window),
            rolling_windows=np.array(self.rolling_windows, dtype=np.int32),
            corr_window=np.array(self.corr_window),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureTransform":
        d = np.load(path)
        return cls(
            keep_mask=d["keep_mask"],
            head_poly_coeffs=d["head_poly_coeffs"],
            robust_median=d["robust_median"],
            robust_iqr=d["robust_iqr"],
            pca_components=d["pca_components"],
            pca_mean=d["pca_mean"],
            pca_explained_variance_ratio=d["pca_explained_variance_ratio"],
            n_kept_channels=int(d["n_kept_channels"]),
            cavitation_kurtosis_threshold=float(d["cavitation_kurtosis_threshold"]),
            stationarity_window=int(d["stationarity_window"]),
            rolling_windows=(
                tuple(int(x) for x in d["rolling_windows"])
                if "rolling_windows" in d
                else (10, 60, 300, 1800)
            ),
            corr_window=int(d["corr_window"]) if "corr_window" in d else 60,
        )


# ---------------------------------------------------------------------------
# Step 2 — Net-head residualization (global polynomial, NOT rolling)
# ---------------------------------------------------------------------------


def compute_net_head(
    allg_data: np.ndarray,
    allg_channels: list[str],
    upper_channel: str = "Oberwasserpegel",
    lower_channel: str = "UW_Pegel_Rodund",
) -> np.ndarray:
    """Compute gross hydraulic head (T,) in metres.

    Both sensors record reservoir levels directly in metres above sea level (masl).
    Net head = upper reservoir level (masl) − lower reservoir level (masl).
    For ROWII this yields ~315–384 m, consistent with a high-head Francis machine.

    Note: Do NOT apply any mA-to-mWs conversion here — the Gantner UDBF files
    store both channels in engineering units (masl), not as raw 4–20 mA signals.
    Applying a mA conversion produces ~4 800 m for the lower level, causing
    catastrophic ill-conditioning in the downstream polynomial regression.
    """

    def _ch(name: str) -> int:
        try:
            return allg_channels.index(name)
        except ValueError:
            for i, ch in enumerate(allg_channels):
                if name.lower() in ch.lower():
                    return i
            raise KeyError(
                f"Channel '{name}' not found in Allg_M1. "
                f"Available: {allg_channels[:10]}…"
            )

    upper_masl = allg_data[:, _ch(upper_channel)].astype(np.float64)
    lower_masl = allg_data[:, _ch(lower_channel)].astype(np.float64)
    return upper_masl - lower_masl


def fit_head_residualization(
    X_rms: np.ndarray,
    net_head: np.ndarray,
    degree: int = 2,
) -> np.ndarray:
    """Fit global polynomial in (net_head, t) for each RMS channel.

    Returns poly_coeffs (C, 5) — one row per channel over the basis
    [h², h, t², t, 1].  The global fit captures slow multi-day reservoir
    drainage trends without absorbing sub-hour operational mode variance.
    """
    T, C = X_rms.shape
    t_norm = np.linspace(0.0, 1.0, T)
    h = net_head.astype(np.float64)

    # Basis: [h², h, t², t, 1]
    A = np.column_stack([h**2, h, t_norm**2, t_norm, np.ones(T)])

    coeffs = np.zeros((C, 5), dtype=np.float64)
    for c in range(C):
        result, *_ = np.linalg.lstsq(A, X_rms[:, c].astype(np.float64), rcond=None)
        coeffs[c] = result
    return coeffs


def apply_head_residualization(
    X_rms: np.ndarray,
    net_head: np.ndarray,
    poly_coeffs: np.ndarray,
) -> np.ndarray:
    """Apply pre-fitted polynomial to residualize X_rms against net head."""
    T = X_rms.shape[0]
    t_norm = np.linspace(0.0, 1.0, T)
    h = net_head.astype(np.float64)
    A = np.column_stack([h**2, h, t_norm**2, t_norm, np.ones(T)])
    trend = A @ poly_coeffs.T  # (T, C)
    return (X_rms.astype(np.float64) - trend).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Cavitation masking
# ---------------------------------------------------------------------------


def _kurtosis_stable(block: np.ndarray, min_std: float = 1e-6) -> np.ndarray:
    """Excess kurtosis per column, numerically stable.

    Avoids scipy's catastrophic-cancellation path by:
    1. Working in float64.
    2. Centring the block before raising to powers (zero-mean eliminates the
       large constant that causes cancellation in the shifted formula).
    3. Returning 0.0 for near-constant channels — physically correct because
       a flat RMS signal carries no impulsive energy and cannot indicate
       cavitation regardless of its nominal kurtosis value.
    """
    b = block.astype(np.float64)
    mu = b.mean(axis=0)
    b_c = b - mu  # centred: cancellation-free
    std = b_c.std(axis=0)
    active = std > min_std  # mask of channels with real signal
    kurt = np.zeros(b.shape[1], dtype=np.float64)
    if active.any():
        b_a = b_c[:, active]
        s_a = std[active]
        n = b.shape[0]
        m4 = (b_a**4).mean(axis=0)
        m2 = s_a**2
        # Fisher excess kurtosis = m4/m2² − 3
        kurt[active] = m4 / (m2**2) - 3.0
    return kurt


def detect_cavitation_windows(
    X_rms: np.ndarray,
    kurtosis_threshold: float = 8.0,
    window_s: int = 10,
) -> np.ndarray:
    """Return bool mask (T,) — True where cavitation impulses are detected.

    Kurtosis > threshold in any channel within a sliding window flags that
    window as cavitation-contaminated.  Flagged timesteps are NOT dropped;
    they are zero-weighted in clustering and encoder losses.

    Uses a numerically stable centred kurtosis estimator to avoid the
    catastrophic-cancellation warning that scipy raises on near-constant
    (standstill / uninstalled sensor) windows.
    """
    T = X_rms.shape[0]
    mask = np.zeros(T, dtype=bool)
    for t in range(0, T, window_s):
        block = X_rms[t : t + window_s]
        if block.shape[0] < 2:
            continue
        k = _kurtosis_stable(block)
        if float(np.nanmax(np.abs(k))) > kurtosis_threshold:
            mask[t : t + window_s] = True
    return mask


# ---------------------------------------------------------------------------
# Step 4 — RobustScaler
# ---------------------------------------------------------------------------


def fit_robust_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel median and IQR (set to 1.0 where < 1e-6)."""
    median = np.median(X, axis=0).astype(np.float64)
    q75 = np.percentile(X, 75, axis=0).astype(np.float64)
    q25 = np.percentile(X, 25, axis=0).astype(np.float64)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return median, iqr


def apply_robust_scaler(
    X: np.ndarray,
    median: np.ndarray,
    iqr: np.ndarray,
) -> np.ndarray:
    return ((X.astype(np.float64) - median) / iqr).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 — Multi-scale rolling statistics
# ---------------------------------------------------------------------------


def _rolling_stats(
    X: np.ndarray,
    window: int,
) -> np.ndarray:
    """Return (T, C*3) array: rolling mean, std, log1p(max-min) for each channel."""
    T, C = X.shape
    out_mean = np.empty((T, C), dtype=np.float32)
    out_std = np.empty((T, C), dtype=np.float32)
    out_range = np.empty((T, C), dtype=np.float32)

    half = window // 2
    X64 = X.astype(np.float64)
    for t in range(T):
        t0 = max(0, t - half)
        t1 = min(T, t + half + 1)
        block = X64[t0:t1]
        out_mean[t] = block.mean(axis=0)
        out_std[t] = block.std(axis=0)
        out_range[t] = np.log1p(block.max(axis=0) - block.min(axis=0))

    return np.concatenate([out_mean, out_std, out_range], axis=1)


def build_multiscale_features(
    X: np.ndarray,
    windows: Sequence[int] = (10, 60, 300, 1800),
) -> np.ndarray:
    """Concatenate rolling stats at multiple timescales → (T, C*3*len(windows))."""
    parts = [_rolling_stats(X, w) for w in windows]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Step 6 — Temporal derivatives
# ---------------------------------------------------------------------------


def build_derivatives(X: np.ndarray) -> np.ndarray:
    """Return (T, C*2): first and tenth-order differences, zero-padded at edges."""
    T, C = X.shape
    d1 = np.zeros_like(X)
    d10 = np.zeros_like(X)
    d1[1:] = X[1:] - X[:-1]
    d10[10:] = X[10:] - X[:-10]
    return np.concatenate([d1, d10], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 7 — Cross-sensor correlations
# ---------------------------------------------------------------------------

_SENSOR_GROUPS = [
    (slice(0, 4), "GenMic"),
    (slice(4, 8), "GenVib"),
    (slice(8, 12), "TurbMic"),
    (slice(12, 16), "TurbVib"),
]

_CORR_PAIRS = [
    (0, 2, "GenMic_TurbMic"),
    (1, 3, "GenVib_TurbVib"),
    (0, 1, "GenMic_GenVib"),
    (2, 3, "TurbMic_TurbVib"),
]


def build_cross_correlations(
    X_raw: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """Rolling Pearson correlation between sensor-group means → (T, 4)."""
    T = X_raw.shape[0]
    group_means = np.stack(
        [X_raw[:, sl].mean(axis=1) for sl, _ in _SENSOR_GROUPS], axis=1
    )  # (T, 4)

    out = np.zeros((T, len(_CORR_PAIRS)), dtype=np.float32)
    half = window // 2
    for t in range(T):
        t0, t1 = max(0, t - half), min(T, t + half + 1)
        block = group_means[t0:t1]  # (win, 4)
        if block.shape[0] < 4:
            continue
        corr_mat = np.corrcoef(block.T)  # (4, 4)
        for col, (i, j, _) in enumerate(_CORR_PAIRS):
            out[t, col] = float(corr_mat[i, j]) if np.isfinite(corr_mat[i, j]) else 0.0
    return out


# ---------------------------------------------------------------------------
# Step 8 — Stationarity score
# ---------------------------------------------------------------------------


def compute_stationarity_score(
    X: np.ndarray,
    window: int = 300,
) -> np.ndarray:
    """Return S[t] ∈ [0,1]: 1 = very stable, 0 = highly transient."""
    T, C = X.shape
    rolling_std = np.zeros(T, dtype=np.float64)
    half = window // 2
    for t in range(T):
        t0, t1 = max(0, t - half), min(T, t + half + 1)
        rolling_std[t] = X[t0:t1].std(axis=0).mean()

    p99 = np.percentile(rolling_std, 99)
    if p99 < 1e-9:
        return np.ones(T, dtype=np.float32)
    score = 1.0 - np.clip(rolling_std / p99, 0.0, 1.0)
    return score.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 9 — PCA
# ---------------------------------------------------------------------------


def fit_pca(
    X: np.ndarray,
    variance_ratio: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA; returns (components, mean, explained_variance_ratio)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance_ratio, svd_solver="full")
    pca.fit(X)
    return (
        pca.components_.astype(np.float64),
        pca.mean_.astype(np.float64),
        pca.explained_variance_ratio_.astype(np.float64),
    )


def apply_pca(
    X: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    return ((X.astype(np.float64) - mean) @ components.T).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level: fit and apply
# ---------------------------------------------------------------------------


def fit_feature_transform(
    X_rms: np.ndarray,
    net_head: np.ndarray,
    allg_data: np.ndarray | None = None,
    *,
    channel_names: Sequence[str] | None = None,
    forced_drop_channels: Sequence[str] | None = None,
    head_poly_degree: int = 2,
    cavitation_kurtosis_threshold: float = 8.0,
    rolling_windows: Sequence[int] = (10, 60, 300, 1800),
    corr_window: int = 60,
    stationarity_window: int = 300,
    pca_variance_ratio: float = 0.95,
) -> FeatureTransform:
    """Fit all transform parameters on the provided training data.

    Parameters
    ----------
    channel_names       : Names of the RMS channels in column order.  Required
                          when ``forced_drop_channels`` is provided.
    forced_drop_channels: Channel names to exclude (uninstalled / disconnected
                          sensors).  All other channels are kept.
    """

    # 1. Build keep_mask solely from forced_drop_channels.
    keep_mask = np.ones(X_rms.shape[1], dtype=bool)
    if forced_drop_channels and channel_names is not None:
        drop_set = set(forced_drop_channels)
        for i, name in enumerate(channel_names):
            if name in drop_set:
                keep_mask[i] = False
        present = set(channel_names)
        missing = [n for n in forced_drop_channels if n not in present]
        if missing:
            warnings.warn(
                f"forced_drop_channels: {len(missing)} name(s) not found in "
                f"channel_names and were ignored: {missing}. "
                "Check spelling against campaign.channel_names_rms."
            )
    elif forced_drop_channels and channel_names is None:
        warnings.warn(
            "forced_drop_channels provided but channel_names is None — "
            "name-based exclusion skipped.  Pass channel_names_rms from the campaign."
        )
    X_kept = X_rms[:, keep_mask]

    # 2. Net-head residualization — fit global polynomial
    poly_coeffs = fit_head_residualization(X_kept, net_head, degree=head_poly_degree)
    X_resid = apply_head_residualization(X_kept, net_head, poly_coeffs)

    # 4. RobustScaler (fit on residualized data)
    median, iqr = fit_robust_scaler(X_resid)
    X_scaled = apply_robust_scaler(X_resid, median, iqr)

    # 5-8. Build all feature columns — pass X_kept (masked) so dropped channels
    #       are excluded from cross-correlations as well as rolling stats.
    X_full = _build_feature_matrix(
        X_raw_16ch=X_kept,
        X_scaled=X_scaled,
        rolling_windows=rolling_windows,
        corr_window=corr_window,
        stationarity_window=stationarity_window,
    )

    # 9. PCA
    components, pca_mean, evr = fit_pca(X_full, variance_ratio=pca_variance_ratio)

    return FeatureTransform(
        keep_mask=keep_mask,
        head_poly_coeffs=poly_coeffs,
        robust_median=median,
        robust_iqr=iqr,
        pca_components=components,
        pca_mean=pca_mean,
        pca_explained_variance_ratio=evr,
        n_kept_channels=int(np.sum(keep_mask)),
        cavitation_kurtosis_threshold=cavitation_kurtosis_threshold,
        stationarity_window=stationarity_window,
        rolling_windows=tuple(rolling_windows),
        corr_window=corr_window,
    )


def apply_feature_transform(
    X_rms: np.ndarray,
    net_head: np.ndarray,
    transform: FeatureTransform,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fitted FeatureTransform to new data.

    Returns
    -------
    X_pca          : (T, D) float32 — PCA-projected feature matrix
    cavitation_mask: (T,) bool      — True where cavitation was detected
    stationarity   : (T,) float32   — S[t] ∈ [0,1]
    """
    X_kept = X_rms[:, transform.keep_mask]
    X_resid = apply_head_residualization(X_kept, net_head, transform.head_poly_coeffs)

    # Cavitation mask on raw kept channels (before scaling)
    cav_mask = detect_cavitation_windows(
        X_resid, kurtosis_threshold=transform.cavitation_kurtosis_threshold
    )

    X_scaled = apply_robust_scaler(
        X_resid, transform.robust_median, transform.robust_iqr
    )

    # Pass X_kept (masked) so dropped channels are excluded from cross-correlations.
    X_full = _build_feature_matrix(
        X_raw_16ch=X_kept,
        X_scaled=X_scaled,
        rolling_windows=transform.rolling_windows,
        corr_window=transform.corr_window,
        stationarity_window=transform.stationarity_window,
    )

    stationarity = compute_stationarity_score(
        X_scaled, window=transform.stationarity_window
    )

    X_pca = apply_pca(X_full, transform.pca_components, transform.pca_mean)
    return X_pca, cav_mask, stationarity


def _build_feature_matrix(
    X_raw_16ch: np.ndarray,
    X_scaled: np.ndarray,
    rolling_windows: Sequence[int],
    corr_window: int,
    stationarity_window: int,
) -> np.ndarray:
    """Assemble the full feature matrix before PCA."""
    parts: list[np.ndarray] = []

    # 5. Multi-scale rolling statistics on scaled kept channels
    parts.append(build_multiscale_features(X_scaled, windows=rolling_windows))

    # 6. Temporal derivatives on scaled kept channels
    parts.append(build_derivatives(X_scaled))

    # 7. Cross-sensor correlations on raw 16-channel data
    if X_raw_16ch.shape[1] == 16:
        parts.append(build_cross_correlations(X_raw_16ch, window=corr_window))

    return np.concatenate(parts, axis=1).astype(np.float32)
