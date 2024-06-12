"""Module containing factory for loss functions."""

import enum
from typing import Callable

import tensorflow as tf

from generative.common.registry import Categories, Registry


@enum.unique
class LossTypes(str, enum.Enum):
    """Loss function types."""

    BINARY_CROSSENTROPY = "binary_crossentropy"
    ORIGINAL_BINARY_CROSSENTROPY = "original_binary_crossentropy"
    LEAST_SQUARES = "least_squares"

    def __str__(self) -> str:
        """Get string representation."""
        return self.value


@Registry.register(Categories.LOSSES, LossTypes.ORIGINAL_BINARY_CROSSENTROPY)
class OriginalBinaryCrossentropy(tf.keras.losses.Loss):
    """Original minmax/BCE GAN loss.

    Notes:
    ------
    Poorer convergence compared to modified BCE below according to paper.
    Discriminator loss: -E{y ln[x] + (1 - y) ln[1 - x]}
    Generator loss: E{(1 - y) ln[1 - x]}

    Goodfellow et al. Generative adversarial networks. NeurIPS, 2014
    https://arxiv.org/abs/1406.2661
    """

    def __init__(self, name: str = LossTypes.ORIGINAL_BINARY_CROSSENTROPY) -> None:
        super().__init__(name=name)

    def __call__(self, real: tf.Tensor | None, fake: tf.Tensor) -> tf.Tensor:
        """Calculate loss.

        Args:
            real: real image discriminator preds (if None, calculates generator loss)
            fake: fake image discriminator preds

        Returns:
            loss: loss value
        """
        # Calculate discriminator loss
        if real is not None:
            labels = tf.concat([tf.ones_like(real), tf.zeros_like(fake)], axis=0)
            preds = tf.concat([real, fake], axis=0)
            return tf.keras.losses.binary_crossentropy(labels, preds, from_logits=True)

        # Calculate generator loss
        else:
            labels = tf.zeros_like(fake)
            return -tf.keras.losses.binary_crossentropy(labels, fake, from_logits=True)
