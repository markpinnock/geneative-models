"""Module containing factory for loss functions."""

import enum
from typing import Callable

import tensorflow as tf

from generative.common.registry import Registry, Categories


@enum.unique
class LossesEnum(str, enum.Enum):
    """Loss function types."""

    BINARY_CROSSENTROPY = "binary_crossentropy"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value
