"""Module for declaring constants."""

import enum


@enum.unique
class Datasets(str, enum.Enum):
    """Dataset types."""

    CIFAR10 = "cifar10"
    MNIST = "mnist"


@enum.unique
class Normalisation(str, enum.Enum):
    """Normalisation types."""

    _01 = "01"  # Normalise to range [0, 1]
    _11 = "-11"  # Normalise to range [-1, 1]
