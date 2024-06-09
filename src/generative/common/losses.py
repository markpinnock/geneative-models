"""Module containing factory for loss functions."""

import enum
from typing import Callable

import tensorflow as tf


@enum.unique
class LossesEnum(str, enum.Enum):
    """Loss function types."""

    BINARY_CROSSENTROPY = "binary_crossentropy"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value


class LossMeta(type):
    """Metaclass for loss function factory."""

    _losses: dict[str, type[tf.keras.losses.Loss]]

    def __contains__(cls, x: str) -> bool:
        """Check if loss function in dictionary.

        Args:
            x: loss function to check for
        """
        return x in cls._losses

    def __getitem__(cls, x: str) -> type[tf.keras.losses.Loss]:
        """Return loss function from dictionary.

        Args:
            x: loss function to return
        """
        return cls._losses[x]


class Loss(metaclass=LossMeta):
    """Activation function factory."""

    _losses = {
        LossesEnum.BINARY_CROSSENTROPY: tf.keras.losses.BinaryCrossentropy,
    }
