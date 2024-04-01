"""Module for downloading datasets."""

import argparse
from pathlib import Path

import keras
import numpy as np

from generative.utils.logger import get_logger

logger = get_logger(__name__)


def get_cifar10(save_dir: Path) -> None:
    """Download the CIFAR-10 dataset from Keras.

    Args:
    ----
        save_dir: dataset save directory

    """
    save_dir /= "cifar10"
    save_dir.mkdir(exist_ok=True, parents=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    np.savez(
        save_dir / "cifar10.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    # Warn user that CIFAR-10 is also saved to .keras cache directory
    logger.info("CIFAR10 dataset saved to: %s", save_dir.resolve())
    logger.warning("CIFAR 10 will also be cached in your .keras/datasets folder.")


def get_mnist(save_dir: Path) -> None:
    """Download the MNIST dataset from Keras.

    Args:
    ----
        save_dir: dataset save directory

    """
    save_dir /= "mnist"
    save_dir.mkdir(exist_ok=True, parents=True)

    _ = keras.datasets.mnist.load_data(path=save_dir / "mnist.npz")
    logger.info("MNIST dataset saved to: %s", save_dir.resolve())


def main() -> None:
    """Entry point for dataset downloader."""
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        "-s",
        help="Data save directory",
        type=str,
        default=None,
    )
    parser.add_argument("--dataset", "-d", help="Dataset", type=str)
    args = parser.parse_args()

    # Determine save directory
    if args.save_dir is None:
        save_dir = Path(__file__).parents[2] / "datasets"
    else:
        save_dir = Path(args.save_dir)

    # Download dataset
    match args.dataset:
        case "cifar10":
            get_cifar10(save_dir)
        case "mnist":
            get_mnist(save_dir)


if __name__ == "__main__":
    main()
