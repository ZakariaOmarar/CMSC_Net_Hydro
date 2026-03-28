"""Low-level training and scoring helpers for the conditional flow model.

This module owns the stateless, reusable building blocks consumed by train.py:
- Data iteration (feature_batch_iterator, flow_batches_with_context).
- Feature assembly (build_flat_features).
- Contrastive pre-training of the context encoder (pretrain_context_encoder).
- Score collection over full datasets (collect_scores_with_context,
  collect_context_vectors).
- Cluster-based score statistics for threshold calibration (cluster_score_stats).
- A pure-NumPy k-means implementation used when scikit-learn is unavailable.

Separating these primitives from the orchestration in train.py makes each
component independently testable and reusable across training variants.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - surfaced on usage
    raise ImportError(
        "flow_train_core requires PyTorch. Install with: pip install torch"
    ) from exc

from .detection_head import ConditionalRealNVP, LightweightContextEncoder, nt_xent_loss


def build_flat_features(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    x = np.concatenate([z, c], axis=1)
    return np.asarray(x, dtype=np.float32)


def feature_batch_iterator(
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        x_t = torch.from_numpy(x[batch_idx]).to(device=device, dtype=torch.float32)
        yield x_t


def augment_feature_views(
    x_t: torch.Tensor,
    *,
    noise_std: float = 0.01,
    drop_prob: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_t.ndim != 2:
        raise ValueError("Expected x_t shape (batch, feature_dim)")

    def _view(inp: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(inp) * float(noise_std)
        keep = (torch.rand_like(inp) > float(drop_prob)).to(dtype=inp.dtype)
        return inp * keep + noise

    return _view(x_t), _view(x_t)


def pretrain_context_encoder(
    context_encoder: LightweightContextEncoder,
    *,
    x_train: np.ndarray,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    seed: int,
    epoch: int,
    temperature: float,
) -> float:
    context_encoder.train()
    losses: list[float] = []

    for x_t in feature_batch_iterator(
        x_train,
        batch_size=batch_size,
        device=device,
        shuffle=True,
        seed=seed + epoch,
    ):
        if int(x_t.shape[0]) < 2:
            continue
        view_a, view_b = augment_feature_views(x_t)
        emb_a = context_encoder(view_a)
        emb_b = context_encoder(view_b)

        optimizer.zero_grad(set_to_none=True)
        loss = nt_xent_loss(emb_a, emb_b, temperature=float(temperature))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        raise ValueError("Context pretraining produced no valid batches")
    return float(np.mean(losses))


def flow_batches_with_context(
    *,
    x: np.ndarray,
    context_encoder: LightweightContextEncoder,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
):
    context_encoder.eval()
    for x_t in feature_batch_iterator(
        x,
        batch_size=batch_size,
        device=device,
        shuffle=shuffle,
        seed=seed,
    ):
        with torch.no_grad():
            c_t = context_encoder(x_t)
        yield x_t, c_t


def collect_scores_with_context(
    flow: ConditionalRealNVP,
    context_encoder: LightweightContextEncoder,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    flow.eval()
    context_encoder.eval()
    out: list[np.ndarray] = []

    with torch.no_grad():
        for x_t in feature_batch_iterator(
            x,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            seed=0,
        ):
            c_t = context_encoder(x_t)
            scores = flow.anomaly_score(x_t, c_t).detach().cpu().numpy()
            out.append(scores)

    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def collect_context_vectors(
    context_encoder: LightweightContextEncoder,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    context_encoder.eval()
    out: list[np.ndarray] = []

    with torch.no_grad():
        for x_t in feature_batch_iterator(
            x,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            seed=0,
        ):
            c_t = context_encoder(x_t).detach().cpu().numpy().astype(np.float32)
            out.append(c_t)

    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _kmeans_cluster_labels(
    x: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    max_iter: int = 50,
) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must have shape (n_samples, n_features) with n_samples > 0")
    if n_clusters <= 1 or x.shape[0] == 1:
        return np.zeros((x.shape[0],), dtype=np.int32)

    k = int(min(n_clusters, x.shape[0]))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(x.shape[0], size=k, replace=False)
    centroids = x[init_idx].astype(np.float32)
    labels = np.zeros((x.shape[0],), dtype=np.int32)

    for _ in range(max_iter):
        d2 = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for i in range(k):
            sel = labels == i
            if np.any(sel):
                centroids[i] = np.mean(x[sel], axis=0)

    return labels


def cluster_score_stats(
    context_vecs: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    max_clusters: int,
) -> dict[str, dict[str, float]]:
    if context_vecs.ndim != 2 or context_vecs.shape[0] == 0 or scores.size == 0:
        return {}

    n_clusters = max(1, min(int(max_clusters), int(context_vecs.shape[0])))
    labels = _kmeans_cluster_labels(
        context_vecs,
        n_clusters=n_clusters,
        seed=seed,
    )

    stats: dict[str, dict[str, float]] = {}
    for cid in range(n_clusters):
        vals = scores[labels == cid]
        if vals.size == 0:
            continue
        stats[f"cluster_{cid}"] = {
            "count": float(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p95": float(np.percentile(vals, 95.0)),
            "p99": float(np.percentile(vals, 99.0)),
        }
    return stats
