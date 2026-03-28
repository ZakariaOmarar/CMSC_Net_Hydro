"""FeatureFrame — the named-feature container passed between extractors and the latent builder.

Each analysis window produces one FeatureFrame per extractor. The latent builder
merges them and calls to_flat_array() to produce the z vector stored in the .npz
latent cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureFrame:
    """Immutable container of named features extracted from one DataSegment window.

    Features may be scalars (float) or 1D arrays (e.g. spectral band energies,
    MFCC means). All are serialized together by to_flat_array() for downstream
    model input.

    Extractors each return a FeatureFrame for their slice of the feature space;
    the latent builder merges them with FeatureFrame.merge() before flattening
    into the z vector that is written to the .npz latent cache.
    """

    features: dict[str, float | np.ndarray]
    start_time: datetime
    duration_s: float
    segment_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.features.keys())

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_flat_array(self) -> np.ndarray:
        """Flatten all features into a single 1D float64 array in sorted-name order.

        This canonical ordering is critical for reproducibility: the same feature
        names always map to the same positions in the z vector, regardless of
        insertion order.
        """
        values: list[float] = []
        for name in self.feature_names:
            value = self.features[name]
            if isinstance(value, np.ndarray):
                values.extend(value.flatten().tolist())
            else:
                values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    def merge(self, other: "FeatureFrame") -> "FeatureFrame":
        """Combine two FeatureFrames into one, with other's features taking precedence on collision.

        Timing metadata is inherited from self.
        """
        merged = {**self.features, **other.features}
        return FeatureFrame(
            features=merged,
            start_time=self.start_time,
            duration_s=self.duration_s,
            segment_metadata={**self.segment_metadata, **other.segment_metadata},
        )
