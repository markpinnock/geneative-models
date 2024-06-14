"""Functions for accessing dataloaders."""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from omegaconf import DictConfig

from generative.common.activations import ActivationsEnum
from generative.common.constants import EPSILON, Normalisation
from generative.common.logger import get_logger
from generative.common.losses import LossTypes

logger = get_logger(__file__)

FROM_FILE = "from_file"
FROM_FOLDER = "from_folder"


def add_channel_dim(dataset: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Add a channel dimension if needed.

    Args:
    ----
        dataset: numpy array of images

    Returns:
    -------
        dataset: uint8 NDArray (size [N, H, W, C])

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
        dataset: float64 NDArray

    """
    # Normalise to [0, 1]
    min_val, max_val = dataset.min(), dataset.max()
    dataset_float = dataset.astype(np.float64)
    dataset_float = (dataset_float - min_val) / (max_val - min_val + EPSILON)

    # Normalise to [-1, 1] if needed
    if normalisation == Normalisation.NEG_ONE_ONE:
        dataset_float = (dataset_float * 2) - 1

    return dataset_float


def normalise_tf(normalisation: str, x: tf.Tensor) -> tf.Tensor:
    """Normalise images to either [0, 1] or [-1, 1].

    Args:
    ----
        normalisation: either `01` or `-11`
        dataset: tensor of images

    Returns:
    -------
        dataset: tf.Tensor

    """
    # Normalise to [0, 1]
    min_val, max_val = tf.reduce_min(x), tf.reduce_max(x)
    x = (x - min_val) / (max_val - min_val + EPSILON)

    # Normalise to [-1, 1] if needed
    if normalisation == Normalisation.NEG_ONE_ONE:
        x = (x * 2) - 1

    return x


def resize_dataset(img_dims: list[int], dataset: tf.Tensor) -> tf.Tensor:
    """Resize [N, H_old, W_old, C] dataset to [N, H_new, W_new, C].

    Notes:
    -----
        - If img_dims field is None, use default dataset size

    Args:
    ----
        img_dims: list of img_dims e.g. [H_new, W_new]
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

    return tf.cast(dataset, tf.float32)


def get_dataset_from_file(
    cfg: DictConfig,
    split: str,
    n_critic: int = 1,
) -> tf.data.Dataset:
    """Get dataset from a single file.

    Args:
    ----
        cfg: config
        split: one of `train`, `valid` or `test`
        n_critic: number of dicsriminator iterations (set to 1 if not Wasserstein GAN)

    Returns:
    -------
        dataset: tf.data.Dataset

    """
    # Get data directory
    if cfg.data_dir is None:
        data_path = Path(__file__).parents[3] / "datasets" / cfg.dataset_name
    else:
        data_path = Path(cfg.data_dir)

    # Load dataset from .npz
    try:
        data_path = data_path / f"{cfg.dataset_name}.npz"
        dataset_np = np.load(data_path)[f"x_{split}"]
    except FileNotFoundError:
        logger.exception("File not found at %s", data_path)
        sys.exit(1)

    # Add channel dimension if needed
    dataset_np = add_channel_dim(dataset_np)

    # Normalise and convert to tensor
    dataset_np = normalise(cfg.normalisation, dataset_np)
    dataset_tf = tf.convert_to_tensor(dataset_np)

    # Resize dataset if needed
    dataset_tf = resize_dataset(cfg.img_dims, dataset_tf)
    dataset_size, height, width, channels = dataset_tf.shape
    cfg.img_dims = [height, width, channels]
    logger.info("Dataset size: %s", dataset_tf.shape)

    dataset = tf.data.Dataset.from_tensor_slices(dataset_tf)

    return dataset.shuffle(dataset_size).batch(cfg.batch_size * n_critic)


def get_dataset_from_folder(cfg: DictConfig, n_critic: int = 1) -> tf.data.Dataset:
    """Get dataset from a folder of images.

    Args:
    ----
        cfg: config
        split: one of `train`, `valid` or `test`
        n_critic: number of dicsriminator iterations (set to 1 if not Wasserstein GAN)

    Returns:
    -------
        dataset: tf.data.Dataset

    """
    # Get data directory
    if cfg.data_dir is None:
        data_dir = Path(__file__).parents[3] / "datasets" / cfg.dataset_name
    else:
        data_dir = Path(cfg.data_dir)

    normalisation_fn = partial(normalise_tf, cfg.normalisation)

    # Load dataset from folder
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        color_mode="rgb",
        batch_size=cfg.batch_size * n_critic,
        image_size=tuple(cfg.img_dims),
        shuffle=True,
        seed=None,
        interpolation="bilinear",
        data_format="channels_last",
        verbose=True,
    )

    cfg.img_dims = tf.shape(next(iter(dataset))).numpy().tolist()[1:]

    return dataset.map(normalisation_fn, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def get_dataset(cfg: DictConfig, split: str) -> tf.data.Dataset:
    """Get dataset.

    Args:
    ----
        cfg: config
        split: one of `train`, `valid` or `test`

    Returns:
    -------
        dataset: tf.data.Dataset

    """
    if (
        cfg.data.normalisation == Normalisation.ZERO_ONE
        and cfg.model.generator.output == ActivationsEnum.TANH
    ):
        logger.warning("Generator output is tanh but data range is [0, 1]")
    elif (
        cfg.data.normalisation == Normalisation.NEG_ONE_ONE
        and cfg.model.generator.output == ActivationsEnum.SIGMOID
    ):
        logger.warning("Generator output is sigmoid but data range is [-1, 1]")

    logger.info(
        "Loading dataset with: normalisation %s, img_dims %s, batch size %s, n_critic %s",
        cfg.data.normalisation,
        cfg.data.img_dims,
        cfg.data.batch_size,
        cfg.data.n_critic,
    )

    # Allow Wasserstein GAN if necessary
    n_critic = cfg.model.get("n_critic", 1)

    if n_critic > 1 and cfg.model.loss != LossTypes.WASSERSTEIN:
        logger.warning(
            "Are you sure you want %s loss with N critic %s?",
            cfg.model.loss,
            n_critic,
        )

    if cfg.data.dataloader == FROM_FILE:
        return get_dataset_from_file(cfg.data, split, n_critic)
    elif cfg.data.dataloader == FROM_FOLDER:
        return get_dataset_from_folder(cfg.data, n_critic)
    else:
        raise ValueError(f"Dataset '{cfg.data.dataloader}' not supported.")
