"""Root entrypoint for the third test dataset full pipeline.

Trains all anomaly and mode models on the third test dataset recordings
(9-mic, 4-accel bench-top prototype v2), then trains the native
LocalizationCNNS3 model and runs per-fault fault inference + 4-method
fault localisation:

    1. SRP-PHAT (9 mics, 36 pairs, hierarchical 3-D grid)
    2. TDOA triangulation (36-pair non-linear L-BFGS-B)
    3. LocalizationCNNS3 — native Transformer trained on S3 fault data
    4. LocalizationCNNS2 — zero-shot transfer (5-mic subset of S3)

Usage:
    python train_third_dataset.py [options]

Key options:
    --config-dir              configs/third             (default)
    --artifacts-root          results/third             (default)
    --data-root               data/third_test_dataset   (default)
    --latent-root             artifacts/latents_third   (default)
    --latent-root-fault       artifacts/latents_third_fault (default)
    --s2-localization-artifact  results/second/localization_cnn_s2.pt (default)
    --dry-run                 Print jobs without executing
    --continue-on-error       Keep running after a failed job
    --resume-from-manifest    <path>  Skip already-successful jobs
    --grid-resolution-m       SRP-PHAT grid spacing in metres (default 0.02)
    --loc-epochs              LocalizationCNNS3 training epochs (default 600)
    --loc-geo-weight          Geometric consistency loss weight (default 0.25)
"""

from __future__ import annotations

from src.modeling.orchestration.third_dataset_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
