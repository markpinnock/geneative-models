"""Utility functions for image processing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from generative.common.constants import Normalisation


def save_images(
    image_tensor: npt.NDArray[np.float32],
    norm: str,
    save_path: Path,
) -> None:
    """Save images to disk.

    Args:
    ----
        image_tensor: tensor of images from model
        norm: normalisation type
        save_path: path to save images
    """
    if norm == Normalisation.ZERO_ONE:
        image_tensor = np.clip(image_tensor, 0, 1)
        image_tensor = np.round(image_tensor * 255).astype(int)

    elif norm == Normalisation.NEG_ONE_ONE:
        image_tensor = np.clip(image_tensor, -1, 1)
        image_tensor = (image_tensor + 1) / 2
        image_tensor = np.round(image_tensor * 255.0).astype(int)

    else:
        raise ValueError(f"Normalisation {norm} not supported")

    _ = plt.figure(figsize=(16, 16))

    for i in range(image_tensor.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image_tensor[i, :, :, :], cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
