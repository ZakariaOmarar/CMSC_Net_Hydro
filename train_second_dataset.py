"""Root entrypoint for the second test dataset full pipeline.

Trains all anomaly and mode models on the second test dataset recordings,
then runs per-position fault inference and GCC-PHAT / SRP-PHAT localization.

Usage:
    python train_second_dataset.py [options]

Key options:
    --config-dir       configs/second  (default)
    --artifacts-root   results/second  (default)
    --data-root        data/second_test_dataset  (default)
    --latent-root      artifacts/latents_second  (default)
    --latent-root-fault artifacts/latents_second_fault  (default)
    --dry-run          Print jobs without executing
    --continue-on-error  Keep running after a failed job
    --resume-from-manifest <path>  Skip already-successful jobs
    --grid-resolution-m  SRP-PHAT grid spacing in metres (default 0.02)
"""

from __future__ import annotations

from src.modeling.orchestration.second_dataset_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
