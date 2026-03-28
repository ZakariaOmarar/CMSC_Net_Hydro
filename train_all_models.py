"""Root entrypoint for full-model orchestration."""

from __future__ import annotations

from src.modeling.orchestration.train_all import main


if __name__ == "__main__":
    raise SystemExit(main())
