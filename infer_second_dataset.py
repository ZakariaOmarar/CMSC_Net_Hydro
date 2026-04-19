"""Root entrypoint for inference-only orchestration on the second test dataset.

Runs anomaly inference and SRP-PHAT fault localization for every fault position
using the already-trained model artifacts.  Model training is skipped.

Usage:
    python infer_second_dataset.py [options]

Key options:
    --config-dir        configs/second              (default)
    --artifacts-root    results/second              (default)
    --data-root         data/second_test_dataset    (default)
    --latent-root       artifacts/latents_second    (default)
    --latent-root-fault artifacts/latents_second_fault (default)
    --dry-run           Print jobs without executing
    --continue-on-error Keep running after a failed job
    --resume-from-manifest <path>  Skip already-successful jobs
"""

from __future__ import annotations

import sys

from src.modeling.orchestration.second_dataset_eval import main

_INFER_STAGES = ["anomaly_infer", "localization"]

if __name__ == "__main__":
    # Inject --stages so that only inference and localization jobs are executed.
    argv = sys.argv[1:] + ["--stages"] + _INFER_STAGES
    raise SystemExit(main(argv))
