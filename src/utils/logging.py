"""Loguru setup with one log file per functional channel.

Usage::

    from src.utils.logging import get_logger
    log = get_logger("training")
    log.info("Epoch done")

Channels map to files in ``logs/``: ``training.log``, ``capture.log``,
``dataset.log``. Every message at ERROR level or above also goes to
``errors.log`` whatever its channel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

    from src.config.schema import LoggingConfig

CHANNELS: tuple[str, ...] = ("training", "capture", "dataset")
_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[channel]:<8} | {message}"


def setup_logging(log_dir: Path, config: LoggingConfig | None = None) -> None:
    """Configure console + per-channel file sinks. Safe to call several times."""
    level = config.level if config else "INFO"
    rotation = config.rotation if config else "10 MB"
    retention = config.retention if config else "30 days"

    logger.remove()
    logger.configure(extra={"channel": "app"})
    logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        logger.add(
            log_dir / f"{channel}.log",
            level="DEBUG",
            format=_FORMAT,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,  # thread/process safe: capture and training run concurrently
            filter=lambda record, ch=channel: record["extra"].get("channel") == ch,
        )
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format=_FORMAT,  # loguru appends the traceback itself
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


def get_logger(channel: str) -> Logger:
    """Return a logger bound to ``channel`` (training, capture, dataset...)."""
    return logger.bind(channel=channel)
