"""Test functions for DCGAN."""

import pytest

import numpy as np
from omegaconf import DictConfig
import tensorflow as tf

from generative.gans.dc_gan import Discriminator, Generator

MAX_CHANNELS = 4


@pytest.mark.parametrize(
    "mb_size,scale_factor,img_channels,dense",
    [
        (2, 3, 1, False), (2, 4, 1, False), (4, 3, 3, False), (4, 4, 3, False),
        (2, 3, 1, True), (2, 4, 1, True), (4, 3, 3, True), (4, 4, 3, True),
    ],
)
def test_discriminator(mb_size: int, scale_factor: int, img_channels: int, dense: bool) -> None:
    """Test discriminator initialisation."""
    img_dims = [4 * 2 ** scale_factor, 4 * 2 ** scale_factor, img_channels]
    channels = [np.min([2 ** i, MAX_CHANNELS]) for i in range(scale_factor)]

    cfg = DictConfig(
        {
            "img_dims": img_dims,
            "max_channels": 4,
            "discriminator": {
                "activation": "leaky_relu",
                "channels": 1,
                "dense": dense,
            },
        },
    )
    discriminator = Discriminator(cfg)
    data = tf.zeros([mb_size] + img_dims)
    output = discriminator(data)

    assert output.shape == [mb_size, 1]
    assert discriminator.channels == channels
    assert np.max(discriminator.channels) == MAX_CHANNELS
    assert discriminator.num_downsample == scale_factor


@pytest.mark.parametrize(
    "mb_size,scale_factor,img_channels,latent_dim,dense",
    [
        (2, 3, 1, 2, False), (2, 4, 1, 2, False), (4, 3, 3, 4, False), (4, 4, 3, 4, False),
        (2, 3, 1, 2, True), (2, 4, 1, 2, True), (4, 3, 3, 4, True), (4, 4, 3, 4, True),
    ],
)
def test_generator(mb_size: int, scale_factor: int, img_channels: int, latent_dim: int, dense: bool) -> None:
    """Test generator initialisation."""

    img_dims = [4 * 2 ** scale_factor, 4 * 2 ** scale_factor, img_channels]
    channels = [np.min([2 ** i, MAX_CHANNELS]) for i in range(scale_factor)]
    reversed(channels)

    cfg = DictConfig(
        {
            "img_dims": img_dims,
            "latent_dim": latent_dim,
            "max_channels": 4,
            "generator": {
                "activation": "relu",
                "channels": 1,
                "dense": dense,
                "output": "sigmoid",
            },
        },
    )
    generator = Generator(cfg)
    data = tf.zeros([mb_size, latent_dim])
    output = generator(data)

    assert output.shape == [mb_size] + img_dims
    assert generator.channels == channels
    assert np.max(generator.channels) == MAX_CHANNELS
    assert generator.num_downsample == scale_factor
