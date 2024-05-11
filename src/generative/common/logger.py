"""Basic logger for generative package."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Build and return logger.

    Args:
    ----
        name: name of module using the logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=(
            "%(levelname)-8s :: %(asctime)s :: %(name)s ::"
            "%(funcName)s:%(lineno)d :: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
