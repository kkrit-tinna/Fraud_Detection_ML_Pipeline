"""Run the whole pipeline in one command (same as `make pipeline`).

    python -m fraud_detection.pipeline
"""
from __future__ import annotations

from . import etl, monitor, train


def main() -> None:
    etl.run()
    train.run()
    monitor.replay()


if __name__ == "__main__":
    main()
