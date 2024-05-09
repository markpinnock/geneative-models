"""Functions for accessing dataloaders."""

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from generative.common.constants import Normalisation
from generative.common.logger import get_logger
from omegaconf import DictConfig

logger = get_logger(__file__)


def add_channel_dim(dataset: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Add a channel dimension if needed.

    Args:
    ----
        dataset: numpy array of images

    Returns:
    -------
        dataset: uint8 NDArray

    """
    if dataset.ndim == 3:
        dataset = dataset[:, :, :, np.newaxis]

    if dataset.ndim != 4:
        msg = f"Incorrect dataset dims: {dataset.shape}"
        raise ValueError(msg)

    return dataset


def normalise(
    normalisation: str,
    dataset: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float64]:
    """Normalise images to either [0, 1] or [-1, 1].

    Args:
    ----
        normalisation: either `01` or `-11`
        dataset: numpy array of images

    Returns:
    -------
        dataset: uint8 NDArray

    """
    # Normalise to [0, 1]
    min_val, max_val = dataset.min(), dataset.max()
    dataset = (dataset - min_val) / (max_val - min_val)

    # Normalise to [-1, 1] if needed
    if normalisation == Normalisation.NEG_ONE_ONE:
        dataset = (dataset * 2) - 1

    return dataset


def resize_dataset(img_dims: list[int], dataset: tf.Tensor) -> tf.Tensor:
    """Resize dataset to [H, W].

    Notes:
    -----
        - If img_dims field is None, use default dataset size

    Args:
    ----
        img_dims: list of img_dims e.g. [H, W]
        dataset: numpy array of images

    Returns:
    -------
        dataset: uint8 NDArray

    """
    # Get image dims from either config or dataset
    if img_dims is not None:
        if len(img_dims) != 2:
            msg = f"Incorrect image dims {img_dims}"
            raise ValueError(msg)

        # Resize images if needed
        dataset = tf.image.resize(dataset, img_dims)

    return tf.cast(dataset, tf.float16)


def get_dataset_from_file(cfg: DictConfig, split: str) -> tf.data.Dataset:
    """Get dataset from a single file.

    Args:
    ----
        cfg: config
        split: one of `train`, `valid` or `test`

    Returns:
    -------
        dataset: tf.data.Dataset

    """
    # Get data directory
    if cfg.data_dir is None:
        data_path = Path(__file__).parents[3] / "datasets"
    else:
        data_path = Path(cfg.data_dir)

    # Load dataset from .npz
    try:
        data_path = data_path / cfg.dataset_name / f"{cfg.dataset_name}.npz"
        dataset_np = np.load(data_path)[f"x_{split}"]
    except FileNotFoundError:
        logger.exception("File not found at %s", data_path)
        sys.exit(1)

    # Add channel dimension if needed
    dataset_np = add_channel_dim(dataset_np)

    # Normalise and convert to tensor
    dataset_np = normalise(dataset_np, cfg.normalisation)
    dataset_tf = tf.convert_to_tensor(dataset_np)

    # Resize dataset if needed
    dataset_tf = resize_dataset(cfg.img_dims, dataset_tf)
    dataset_size, height, width = dataset_tf.shape[0:4]
    cfg.img_dims = [height, width]

    dataset = tf.data.Dataset.from_tensor_slices(dataset_tf)

    return dataset.shuffle(dataset_size).batch(32)