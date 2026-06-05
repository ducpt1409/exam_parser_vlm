"""Logger dùng chung (loguru)."""
from __future__ import annotations

import sys

from loguru import logger

from src.core.config import settings

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>"
    ),
)

__all__ = ["logger"]
