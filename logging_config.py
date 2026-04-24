"""Centralised logging for the insider-trading pipeline.

Usage:

    from logging_config import get_logger
    log = get_logger(__name__)
    log.info("started")
    log.error("fetch failed for %s", ticker)
    try:
        ...
    except Exception:
        log.exception("unhandled error in step X")

Honours LOG_LEVEL and LOG_FILE from .env. Safe to import multiple times —
handlers are only attached once per process.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stderr handler so errors show up next to the existing print() output
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    stream.setLevel(level)
    root.addHandler(stream)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(fmt)
            file_handler.setLevel(level)
            root.addHandler(file_handler)
        except OSError:
            # File logging is best-effort; fall back to stderr only.
            root.warning("could not open LOG_FILE=%s, continuing without file logging", log_file)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)
