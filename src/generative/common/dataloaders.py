"""Functions for accessing dataloaders."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from generative.common.constants import Normalisation
from generative.common.logger import get_logger
from omegaconf import DictConfig

logger = get_logger(__file__)


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
    if dataset_np.ndim == 3:
        dataset_np = dataset_np[:, :, :, np.newaxis]

    if dataset_np.ndim != 4:
        logger.error("Incorrect dataset dims: %s", dataset_np.shape)
        sys.exit(1)

    # Normalise to [0, 1]
    min_val, max_val = dataset_np.min(), dataset_np.max()
    dataset_np = (dataset_np - min_val) / (max_val - min_val)

    # Normalise to [-1, 1] if needed
    if cfg.normalisation == Normalisation.NEG_ONE_ONE:
        dataset_np = (dataset_np * 2) - 1

    dataset_tf = tf.convert_to_tensor(dataset_np.astype(np.float16))

    # Get image dims from either config or dataset
    if cfg.img_dims is not None:
        if len(cfg.img_dims) != 2:
            logger.error("Incorrect image dims %s", cfg.img_dims)
            sys.exit(1)

        new_dims = tuple(cfg.img_dims)

    else:
        new_dims = dataset_np.shape[1:3]
        cfg.img_dims = new_dims

    # Resize image if needed
    old_dims = dataset_np.shape[1:3]

    if old_dims != new_dims:
        dataset_tf = tf.image.resize(dataset_tf, new_dims)

    dataset_size = dataset_tf.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(dataset_tf)

    return dataset.shuffle(dataset_size).batch(32)
