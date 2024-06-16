"""Configuration for pytest."""

from pathlib import Path
import pytest
from typing import Any

import matplotlib.image as plt
import numpy as np

TEST_IMG_DIMS_3D = [8, 4, 4]
TEST_IMG_DIMS_4D = [8, 4, 4, 3]


@pytest.fixture(scope="session", params=[TEST_IMG_DIMS_3D, TEST_IMG_DIMS_4D])
def create_test_dataset_file(tmp_path_factory: pytest.TempPathFactory, request: Any) -> Path:
    """Create a test dataset for both black and white and RGB images."""
    tmp_dir = tmp_path_factory.mktemp("dataset_file")
    save_dir = tmp_dir / "dataset.npz"
    dataset = np.zeros(request.param)
    dataset[:, 0, 0] = 255
    np.savez(save_dir, x_train=dataset)

    return save_dir


@pytest.fixture(scope="session", params=[TEST_IMG_DIMS_3D, TEST_IMG_DIMS_4D])
def create_test_dataset_folder(tmp_path_factory: pytest.TempPathFactory, request: Any) -> Path:
    """Create a test dataset folder for both black and white and RGB images."""
    tmp_dir = tmp_path_factory.mktemp("dataset_folder")

    for i in range(8):
        img = np.zeros(request.param, dtype="uint8")
        img[:, 0, 0] = 255
        plt.imsave(tmp_dir / f"{i}.png", img)

    return tmp_dir
