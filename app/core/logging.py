"""Structured logging setup (M4 — core)."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a simple structured format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Silence verbose third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
