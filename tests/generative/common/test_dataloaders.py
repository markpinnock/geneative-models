import numpy as np
import pytest
from generative.common.constants import Normalisation
from generative.common.dataloaders import add_channel_dim, normalise, resize_dataset


@pytest.mark.parametrize(
    "data_dims,expected_dims",
    [
        ([8, 4, 4], [8, 4, 4, 1]),
        ([8, 4, 4, 1], [8, 4, 4, 1]),
        ([8, 4, 4, 3], [8, 4, 4, 3]),
    ],
)
def test_add_channel_dim(data_dims: list[int], expected_dims: list[int]) -> None:
    """Test adding channel dimension."""
    dataset = np.zeros(data_dims)
    dataset = add_channel_dim(dataset)

    assert list(dataset.shape) == expected_dims


@pytest.mark.parametrize("data_dims", [[4, 4], [1, 4, 4, 2, 1]])
def test_add_channel_dim_fail(data_dims: list[int]) -> None:
    dataset = np.zeros(data_dims)

    with pytest.raises(ValueError):
        _ = add_channel_dim(dataset)


@pytest.mark.parametrize(
    "normalisation,expected_min_max",
    [
        (Normalisation.ZERO_ONE, [0.0, 1.0]),
        (Normalisation.NEG_ONE_ONE, [-1.0, 1.0]),
    ],
)
def test_normalise(normalisation: str, expected_min_max: list[float]) -> None:
    """Test normalisation."""
    dataset = np.repeat(np.reshape(np.arange(0, 256), [4, 8, 8, 1]), 3, axis=3)
    dataset = normalise(normalisation, dataset)

    assert dataset.min() == expected_min_max[0]
    assert dataset.max() == expected_min_max[1]


def test_resize_dataset() -> None:
    """Test resizing dataset."""
