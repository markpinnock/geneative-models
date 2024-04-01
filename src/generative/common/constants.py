"""Module for declaring constants."""

import enum


@enum.unique
class Datasets(str, enum.Enum):
    """Datasets enum."""

    CIFAR10 = "cifar10"
    MNIST = "mnist"
