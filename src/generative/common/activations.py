"""Module containing factory for activation functions."""

import enum
from typing import Callable

import tensorflow as tf


@enum.unique
class ActivationsEnum(str, enum.Enum):
    """Activation function types."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    LINEAR = "linear"
    TANH = "tanh"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value


class ActivationMeta(type):
    """Metaclass for activation function factory."""

    def __contains__(cls, x: str) -> bool:
        """Check if activation in dictionary.

        Args:
            x: activation to check for
        """
        return x in cls._activations

    def __getitem__(cls, x: str) -> Callable:
        """Return activation from dictionary.

        Args:
            x: activation to return
        """
        return cls._activations[x]


class Activation(metaclass=ActivationMeta):
    """Activation function factory.

    Notes:
        - Leaky relu has alpha hard-coded to 0.2
    """

    _activations = {
        ActivationsEnum.LEAKY_RELU: lambda x: tf.keras.activations.leaky_relu(
            x,
            negative_slope=0.2,
        ),
        ActivationsEnum.LINEAR: tf.keras.activations.linear,
        ActivationsEnum.RELU: tf.keras.activations.relu,
        ActivationsEnum.TANH: tf.keras.activations.tanh,
    }
