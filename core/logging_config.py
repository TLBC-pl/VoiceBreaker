import logging
from typing import Final

from core.config import config


def setup_logging() -> None:

    log_level: Final[int] = logging.DEBUG if config.DEBUG else logging.INFO
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
