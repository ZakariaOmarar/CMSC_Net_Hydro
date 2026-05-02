"""Context-vector training and evaluation for the V0–V5 chained system.

V1 / V2 share two pieces of machinery:
  - `cluster_metric`: K-means(k=4) + Hungarian-match against folder labels →
    cluster purity + NMI.  Pure evaluation, never used at training time, per
    the plan's label-leakage invariant.
  - `v1_ssl`:         per-modality SimCLR-style contrastive trainer.  V2 will
    add a cross-attention block + Latent Masked Modeling on top of V1's
    weights.
"""

from .cluster_metric import cluster_purity_and_nmi, hungarian_purity
from .v1_ssl import V1Result, V1SSLConfig, evaluate_sanity_gate, train_v1_per_modality

__all__ = [
    "V1Result",
    "V1SSLConfig",
    "cluster_purity_and_nmi",
    "evaluate_sanity_gate",
    "hungarian_purity",
    "train_v1_per_modality",
]
