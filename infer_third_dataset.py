"""Root entrypoint for inference-only orchestration on the third test dataset.

Runs anomaly inference and 4-method fault localisation for every fault
folder using already-trained model artifacts.  Model training is skipped.

Usage:
    python infer_third_dataset.py [options]

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
"""

from __future__ import annotations

import sys

from src.modeling.orchestration.third_dataset_eval import main

_INFER_STAGES = ["anomaly_infer", "localization"]

if __name__ == "__main__":
    # Inject --stages so that only inference and localisation jobs are executed.
    argv = sys.argv[1:] + ["--stages"] + _INFER_STAGES
    raise SystemExit(main(argv))
