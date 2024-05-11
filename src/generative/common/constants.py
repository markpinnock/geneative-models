"""Module for declaring constants."""

import enum

EPSILON = 1e-12


@enum.unique
class Datasets(str, enum.Enum):
    """Dataset types."""

    CIFAR10 = "cifar10"
    MNIST = "mnist"


@enum.unique
class DataSplits(str, enum.Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALID = "validation"
    TEST = "test"


@enum.unique
class Normalisation(str, enum.Enum):
    """Normalisation types."""

    ZERO_ONE = "01"  # Normalise to range [0, 1]
    NEG_ONE_ONE = "-11"  # Normalise to range [-1, 1]
