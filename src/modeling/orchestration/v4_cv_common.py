"""Shared scaffolding for the post-hoc V4 cross-validation drivers.

`v4_lopo_cv` (leave-one-position-out) and `v4_cross_dataset` (dataset transfer)
both reuse a V2 encoder trained by `full_run` and run on the *same* labelled
cohort — D2/D3/D4/D5, knock-interval-restricted. The pieces that are identical
between them live here so they cannot drift. `v4_loocv` deliberately uses a
narrower cohort (no D5, no knock-interval restriction) and keeps its own gather.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..localization import V4_CANDIDATE_GRID, precompute_v4_knock_event_samples
from .full_run import _d3_spatial_overrides, resolved_loader

if TYPE_CHECKING:
    from ..context.v2_fusion import V2FusionEncoder
    from ..context.v2_ssl import V2SSLConfig
    from ..localization import V4Sample

ChannelMode = Literal["both", "srp_only", "tdoa_only", "vibration_only_learned"]

# The four V4 channel-ablation paradigms (acoustic SRP, accel TDOA, learned
# vibration-only, and the full fusion) compared in the RQ3 localization tables.
CHANNEL_MODES: tuple[ChannelMode, ...] = (
    "both", "srp_only", "tdoa_only", "vibration_only_learned",
)


def load_or_precompute_cv_samples(
    encoder: V2FusionEncoder,
    v2_cfg: V2SSLConfig,
    *,
    samples_cache: Path | None,
    burst_aware_srp: bool = True,  # retained for call-site compat; ignored
    log_prefix: str,
) -> list[V4Sample]:
    """Return the shared V4Sample list for a cross-validation driver.

    Loads from ``samples_cache`` when it exists; otherwise gathers the D2/D3/D4/D5
    labelled recordings and precomputes **per-knock** V4 samples (each detected
    knock localized on its own transient-centred crop — the multi-seed-confirmed
    RQ3 win).  ``log_prefix`` tags the progress prints (e.g. "V4 LOPO").

    NOTE: this builder replaced the older fixed-window
    ``precompute_v4_samples`` path; a ``samples_cache`` written by the old
    builder must be regenerated (delete the pickle) to pick up per-knock samples.
    The ``burst_aware_srp`` argument is retained only for call-site
    compatibility and is no longer used (per-knock cropping is inherent).
    """
    if samples_cache is not None and Path(samples_cache).exists():
        with Path(samples_cache).open("rb") as fh:
            samples = pickle.load(fh)
        print(f"{log_prefix}: loaded {len(samples)} cached V4 samples from {samples_cache}")
        return samples

    print(f"{log_prefix}: gathering labeled segments + precomputing V4 samples ...")
    D2 = resolved_loader("d2.yaml")
    D3 = resolved_loader("d3.yaml")
    D4 = resolved_loader("d4.yaml")
    D5 = resolved_loader("d5.yaml")
    d2_labeled = [
        s for s in D2.list_segments()
        if s.is_anomaly and s.spatial_label is not None and s.mode_label is not None
    ]
    d3_segs = D3.list_segments()
    overrides = _d3_spatial_overrides(d3_segs)
    d3_labeled = [s for s in d3_segs if s.recording_id in overrides]
    d4_labeled = [s for s in D4.list_segments() if s.is_anomaly and s.spatial_label is not None]
    d5_labeled = [s for s in D5.list_segments() if s.is_anomaly and s.spatial_label is not None]
    all_labeled = d2_labeled + d3_labeled + d4_labeled + d5_labeled
    print(
        f"  D2={len(d2_labeled)}, D3={len(d3_labeled)}, D4={len(d4_labeled)}, "
        f"D5={len(d5_labeled)}, total={len(all_labeled)} labeled recordings"
    )
    t0 = time.time()
    samples = precompute_v4_knock_event_samples(
        encoder, all_labeled,
        v2_cfg=v2_cfg, grid=V4_CANDIDATE_GRID,
        spatial_label_overrides=overrides,
    )
    print(f"  precomputed {len(samples)} per-knock V4 samples in {time.time() - t0:.1f}s")
    if samples_cache is not None:
        with Path(samples_cache).open("wb") as fh:
            pickle.dump(samples, fh)
    return samples
