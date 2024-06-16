"""Test functions for DCGAN."""

import pytest

import numpy as np
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError
import tensorflow as tf

from generative.common.losses import LossTypes
from generative.gans.dc_gan import DCGAN
from generative.gans.wasserstein import WeightClipConstraint, WassersteinTypes

MAX_CHANNELS = 4


def build_layer(img_dims: list[int], layer: tf.keras.layers.Layer) -> None:
    """Build layer, e.g. discriminator to populate variable list."""
    x = tf.keras.layers.Input(img_dims)
    tf.keras.Model(inputs=[x], outputs=layer(x))


@pytest.mark.parametrize("dense", [False, True])
def test_no_wasserstein(dense: bool) -> None:
    """Test non-Wasserstein initialisation."""
    img_dims = [32, 32, 3]
    latent_dim = 4
    cfg = DictConfig(
        {
            "batch_size": 2,
            "img_dims": img_dims,
            "latent_dim": latent_dim,
            "loss": LossTypes.BINARY_CROSSENTROPY,
            "num_examples": 4,
            "model_name": "dcgan",
            "max_channels": 4,
            "discriminator": {
                "activation": "leaky_relu",
                "channels": 1,
                "dense": dense,
            },
            "generator": {
                "activation": "relu",
                "channels": 1,
                "dense": dense,
                "output": "linear",
            },
        },
    )
    # Initialise model and check no n_critic attribute
    model = DCGAN(cfg)
    assert not hasattr(model, "n_critic")

    # Build and check no weight clipping constraints in variables
    build_layer(img_dims, model.discriminator)
    build_layer([latent_dim], model.generator)
    assert len(model.variables) > 0

    for var in model.discriminator.variables:
        assert var.constraint is None
    for var in model.generator.variables:
        assert var.constraint is None


@pytest.mark.parametrize("dense", [False, True])
def test_orig_wasserstein(dense: bool) -> None:
    """Test original Wasserstein initialisation with weight clipping."""
    img_dims = [32, 32, 3]
    latent_dim = 4
    cfg = DictConfig(
        {
            "batch_size": 2,
            "img_dims": img_dims,
            "latent_dim": latent_dim,
            "loss": LossTypes.WASSERSTEIN,
            "num_examples": 4,
            "model_name": "dcgan",
            "max_channels": 4,
            "discriminator": {
                "activation": "leaky_relu",
                "channels": 1,
                "dense": dense,
            },
            "generator": {
                "activation": "relu",
                "channels": 1,
                "dense": dense,
                "output": "linear",
            },
            "wasserstein_type": WassersteinTypes.CLIP_WEIGHTS,
            "n_critic": 5,
            "clip_value": 0.01,
        },
    )
    # Initialise model and check no n_critic attribute
    model = DCGAN(cfg)
    assert hasattr(model, "n_critic")

    # Build and check weight clipping constraints in correct variables
    build_layer(img_dims, model.discriminator)
    build_layer([latent_dim], model.generator)
    assert len(model.variables) > 0

    for variable in model.discriminator.variables:
        if variable.ndim > 1:
            assert isinstance(variable.constraint, WeightClipConstraint)

    for var in model.generator.variables:
        assert var.constraint is None

def test_orig_wasserstein_fail() -> None:
    """Test Wasserstein initialisation fails without arguments."""
    img_dims = [32, 32, 3]
    cfg = DictConfig(
        {
            "batch_size": 2,
            "img_dims": img_dims,
            "latent_dim": 4,
            "loss": LossTypes.WASSERSTEIN,
            "num_examples": 4,
            "model_name": "dcgan",
            "max_channels": 4,
            "discriminator": {
                "activation": "leaky_relu",
                "channels": 1,
                "dense": False,
            },
            "generator": {
                "activation": "relu",
                "channels": 1,
                "dense": False,
                "output": "linear",
            },
        },
    )
    # Initialise model and check no n_critic attribute
    with pytest.raises(ConfigAttributeError):
        _ = DCGAN(cfg)
