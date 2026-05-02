"""Cluster-purity evaluation against ground-truth labels.

The plan's RQ1 sanity gate (V1) and headline metric (V2) both follow the same
recipe:
  1. K-means(k=N_classes) on the embedding vectors.
  2. Hungarian-match cluster indices to ground-truth label indices so the
     reported number is invariant to cluster naming.
  3. Cluster purity = sum_c max_y count(c, y) / N.

This is **evaluation only**.  Mode labels do not enter the V1 / V2 training
loops; they only appear here, and the K-means + Hungarian step makes that
explicit (cluster IDs are arbitrary integers up until the matching step).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_purity(
    cluster_idx: np.ndarray,
    label_idx: np.ndarray,
    n_clusters: int,
    n_labels: int,
) -> tuple[float, dict[int, int], np.ndarray]:
    """Hungarian-matched cluster purity.

    Returns
    -------
    purity : float in [0, 1]
        Sum over predicted clusters of the count of the most-common label in
        that cluster, divided by total samples — after Hungarian assignment of
        cluster indices to label indices.
    mapping : dict[cluster_idx, label_idx]
        The optimal cluster → label assignment.
    confusion : (n_clusters, n_labels) int array
        Pre-mapping confusion matrix, rows = clusters, cols = labels.
    """
    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    if cluster_idx.shape != label_idx.shape:
        raise ValueError(
            f"cluster_idx and label_idx must have the same shape; "
            f"got {cluster_idx.shape} and {label_idx.shape}"
        )
    n = int(cluster_idx.shape[0])
    if n == 0:
        return 0.0, {}, np.zeros((n_clusters, n_labels), dtype=np.int64)

    confusion = np.zeros((n_clusters, n_labels), dtype=np.int64)
    for c, y in zip(cluster_idx, label_idx):
        confusion[int(c), int(y)] += 1

    cost = -confusion.astype(np.int64)
    if n_clusters >= n_labels:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # linear_sum_assignment requires the rect matrix to have rows ≤ cols
        row_ind, col_ind = linear_sum_assignment(cost.T)
        row_ind, col_ind = col_ind, row_ind

    matched = int(confusion[row_ind, col_ind].sum())
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return float(matched / n), mapping, confusion


def _normalised_mutual_information(
    cluster_idx: np.ndarray, label_idx: np.ndarray
) -> float:
    """Symmetric normalised mutual information (NMI), matching sklearn's default
    'arithmetic' average."""
    cluster_idx = np.asarray(cluster_idx, dtype=np.int64)
    label_idx = np.asarray(label_idx, dtype=np.int64)
    n = int(cluster_idx.shape[0])
    if n == 0:
        return 0.0

    cu = np.unique(cluster_idx)
    yu = np.unique(label_idx)
    contingency = np.zeros((cu.size, yu.size), dtype=np.float64)
    cu_idx = {int(v): i for i, v in enumerate(cu)}
    yu_idx = {int(v): i for i, v in enumerate(yu)}
    for c, y in zip(cluster_idx, label_idx):
        contingency[cu_idx[int(c)], yu_idx[int(y)]] += 1.0

    p_xy = contingency / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.where(p_xy > 0, p_xy / (p_x @ p_y + 1e-12), 1.0))
    mi = float(np.where(p_xy > 0, p_xy * log_term, 0.0).sum())

    h_x = float(-(p_x * np.log(np.where(p_x > 0, p_x, 1.0))).sum())
    h_y = float(-(p_y * np.log(np.where(p_y > 0, p_y, 1.0))).sum())
    denom = 0.5 * (h_x + h_y)
    if denom <= 0:
        return 0.0
    return mi / denom


def cluster_purity_and_nmi(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int = 4,
    seed: int = 42,
) -> dict:
    """K-means(k=n_clusters) on embeddings, Hungarian-matched against `labels`.

    Returns a dict with `purity`, `nmi`, the cluster→label `mapping`, and the
    pre-mapping `confusion` matrix.
    """
    from sklearn.cluster import KMeans

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, D); got {embeddings.shape}")
    if embeddings.shape[0] != len(labels):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but {len(labels)} labels"
        )

    label_set = sorted(set(labels))
    label_to_idx = {y: i for i, y in enumerate(label_set)}
    label_idx = np.array([label_to_idx[y] for y in labels], dtype=np.int64)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_idx = km.fit_predict(embeddings)

    purity, mapping, confusion = hungarian_purity(
        cluster_idx, label_idx, n_clusters=n_clusters, n_labels=len(label_set)
    )
    nmi = _normalised_mutual_information(cluster_idx, label_idx)

    return {
        "purity": purity,
        "nmi": nmi,
        "n_clusters": n_clusters,
        "n_labels": len(label_set),
        "label_set": tuple(label_set),
        "mapping": mapping,
        "confusion": confusion,
        "cluster_idx": cluster_idx.astype(np.int64),
    }


__all__ = ["cluster_purity_and_nmi", "hungarian_purity"]
