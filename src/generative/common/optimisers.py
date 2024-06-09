"""Module containing factory for optimisers."""

import enum
from typing import Callable

import tensorflow as tf


@enum.unique
class OptimisersEnum(str, enum.Enum):
    """Optimiser types."""

    ADAM = "adam"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value


class OptimiserMeta(type):
    """Metaclass for optimiser factory."""

    _optimisers: dict[str, type[tf.keras.optimizers.Optimizer]]

    def __contains__(cls, x: str) -> bool:
        """Check if optimiser in dictionary.

        Args:
            x: optimiser to check for
        """
        return x in cls._optimisers

    def __getitem__(cls, x: str) -> type[tf.keras.optimizers.Optimizer]:
        """Return optimiser from dictionary.

        Args:
            x: optimiser to return
        """
        return cls._optimisers[x]


class Optimiser(metaclass=OptimiserMeta):
    """Optimiser factory."""

    _optimisers = {
        OptimisersEnum.ADAM: tf.keras.optimizers.Adam,
    }
