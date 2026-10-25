"""Logging setup: short messages on the console, timestamped ones in a log file."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):  # idempotent when called twice
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        root.addHandler(to_file)

    # Third-party libraries stay quiet unless something goes wrong.
    # (httpx: Qdrant and Hugging Face clients log every HTTP request at INFO.)
    for name in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
