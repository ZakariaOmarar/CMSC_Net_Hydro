"""V0-style anomaly baseline — per-cluster KDE on V2 c_t vectors.

A simple, strong baseline the V3 conditional flow must beat to earn its
complexity.  The audit (2026-05-22, see plan `i-would-like-you-distributed-pizza`)
flagged that no such baseline existed: V3's val NLL was reported without
reference to what a non-deep density estimator achieves on the same
`(x = mean_pool(fused), c = PMA-pooled c_t)` pairs.

Design:
  1. K-means(`n_clusters`) on the V3 healthy training-cohort `c_t` vectors —
     same `n_clusters` (default 3) as `PerClusterThresholds` so the buckets
     line up with V3's threshold structure.
  2. Per-cluster `scipy.stats.gaussian_kde` fit on the bucketed `x_for_v3`
     vectors (V3's pre-flow input — mean-pool of fused tokens; *not* the
     PMA-pooled c_t).  Scott's rule for the bandwidth (the SciPy default);
     no per-cluster tuning to keep the comparison apples-to-apples.
  3. Score = `-kde.logpdf(x).mean()` per window — the same orientation as
     V3's `anomaly_score = -log p(x|c)`, so lower NLL is "more in-distribution"
     in both models.

Reported metric: mean held-out NLL on the same `val_eval` cohort V3 uses.
A V3 vs KDE delta close to zero means V3 has not extracted information
beyond what a fixed-kernel density estimator already captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans


@dataclass
class KDEResult:
    """Outcome of fitting KDE-on-c_t on a healthy training cohort."""

    centroids: np.ndarray  # (K, c_dim)
    val_nll_mean: float    # mean held-out -log p(x)
    val_nll_per_cluster: dict[int, float]
    n_per_cluster_train: np.ndarray  # (K,)
    n_per_cluster_val: np.ndarray    # (K,)
    n_clusters_used: int             # may be < requested if cohort small


def fit_and_score_kde_on_ct(
    x_train: np.ndarray,
    c_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    *,
    n_clusters: int = 3,
    seed: int = 42,
) -> KDEResult:
    """K-means(c_train) → per-cluster KDE on x_train → mean held-out NLL.

    Mirrors V3's evaluation: `x_val` is scored against the per-cluster KDE
    whose centroid is closest to each `c_val[i]`.  When a cluster has < 2
    training points its KDE cannot be fit (singular covariance); those
    windows fall back to the pooled KDE over `x_train`.
    """
    if x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]:
        raise ValueError(
            f"x_train/x_val shape mismatch: {x_train.shape} vs {x_val.shape}"
        )
    n_clusters_eff = max(1, min(int(n_clusters), c_train.shape[0]))
    km = KMeans(n_clusters=n_clusters_eff, random_state=seed, n_init=10).fit(c_train)
    centroids = km.cluster_centers_.astype(np.float64)
    train_labels = km.labels_
    # Nearest-centroid assignment for val (KMeans.predict refits internally —
    # using argmin over euclidean distances is cheaper and explicit).
    d_val = np.linalg.norm(c_val[:, None, :] - centroids[None, :, :], axis=-1)
    val_labels = d_val.argmin(axis=1)

    # Pooled fallback KDE for clusters too small to fit individually.
    pooled_kde = gaussian_kde(x_train.T) if x_train.shape[0] >= 2 else None

    per_cluster_kde: dict[int, gaussian_kde] = {}
    n_train_per: list[int] = []
    for k in range(n_clusters_eff):
        mask = train_labels == k
        n_train_per.append(int(mask.sum()))
        if mask.sum() >= 2:
            try:
                per_cluster_kde[k] = gaussian_kde(x_train[mask].T)
            except Exception:
                # Singular covariance, identical samples, etc.  Fall through
                # to pooled.
                pass

    n_val_per: list[int] = []
    nll_per_cluster: dict[int, float] = {}
    nll_accum = np.zeros(x_val.shape[0], dtype=np.float64)
    for k in range(n_clusters_eff):
        mask = val_labels == k
        n_val_per.append(int(mask.sum()))
        if not mask.any():
            continue
        kde = per_cluster_kde.get(k, pooled_kde)
        if kde is None:
            nll_accum[mask] = float("nan")
            nll_per_cluster[k] = float("nan")
            continue
        log_p = kde.logpdf(x_val[mask].T)
        nll_accum[mask] = -log_p
        nll_per_cluster[k] = float(np.mean(-log_p))

    finite = np.isfinite(nll_accum)
    val_nll_mean = float(np.mean(nll_accum[finite])) if finite.any() else float("nan")

    return KDEResult(
        centroids=centroids,
        val_nll_mean=val_nll_mean,
        val_nll_per_cluster=nll_per_cluster,
        n_per_cluster_train=np.asarray(n_train_per, dtype=np.int64),
        n_per_cluster_val=np.asarray(n_val_per, dtype=np.int64),
        n_clusters_used=n_clusters_eff,
    )


__all__ = ["KDEResult", "fit_and_score_kde_on_ct"]
