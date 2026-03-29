"""Root entrypoint for inference-only orchestration."""

from __future__ import annotations

from src.modeling.orchestration.infer_all import main


if __name__ == "__main__":
    raise SystemExit(main())
