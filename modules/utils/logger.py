"""
logger.py
=========
Centralized logging factory. Every module obtains its logger via
`get_logger(__name__)` to guarantee consistent formatting and a single
place to control log level / handlers / file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    from config.config import settings  # local import avoids circular imports

    log_dir: Path = settings.paths.log_dir
    log_file = log_dir / "cad_revision_ai.log"

    root = logging.getLogger("cad_revision_ai")
    root.setLevel(settings.log_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Filesystem might be read-only in some deployment contexts; degrade gracefully.
        pass

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `cad_revision_ai` root logger."""
    _configure_root()
    return logging.getLogger(f"cad_revision_ai.{name}")
