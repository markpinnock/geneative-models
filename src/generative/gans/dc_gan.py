"""Deep Convolutional GAN implementation.

Radford et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR 2016.
https://arxiv.org/abs/1511.06434
"""

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from generative.common.activations import Activation
from generative.common.registry import Categories, Registry
from generative.gans.base_gan import BaseGAN


class DCDense(tf.keras.layers.Layer):
    """Dense layer for DCGAN with activation function.

    Args:
        activation: activation function
        units: size of dense layer output
        kernel_initializer: initialiser for kernel weights
        name: layer name
    """

    def __init__(self, activation: str, **kwargs: int | str) -> None:
        super().__init__(name=kwargs["name"])
        self.dense = tf.keras.layers.Dense(**kwargs)
        if activation not in Activation:
            raise ValueError(f"Activation {activation} not supported")
        self.activation = Activation[activation]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through dense layer."""
        x = self.dense(x)
        return self.activation(x)


class DownBlock(tf.keras.layers.Layer):
    """Convolutional block for DCGAN with (optional) batch normalisation and activation.

    Args:
        channels: number of output channels
        initialiser: initialiser for layer weights
        activation: activation function
        batchnorm: whether to use batch normalisation
        final: whether this is the final layer
        name: layer name
    """

    def __init__(
        self,
        channels: int,
        initialiser: tf.keras.initializers.Initializer,
        activation: str,
        batchnorm: bool = True,
        final: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if activation not in Activation:
            raise ValueError(f"Activation {activation} not supported")
        self.activation = Activation[activation]

        # If final layer, use valid padding and no stride
        if final:
            self.conv = tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(4, 4),
                strides=(1, 1),
                padding="VALID",
                kernel_initializer=initialiser,
                name="conv",
            )

        # Otherwise, use same padding and stride
        else:
            self.conv = tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
                kernel_initializer=initialiser,
                name="conv",
            )

        # Optionally add batch normalisation
        if batchnorm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")
        else:
            self.bn = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through convolutional block."""
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        return self.activation(x)


class UpBlock(tf.keras.layers.Layer):
    """Tranpose convolutional block for DCGAN with (optional) batch normalisation and activation.

    Args:
        channels: number of output channels
        initialiser: initialiser for layer weights
        activation: activation function
        batchnorm: whether to use batch normalisation
        first: whether this is the first layer
        name: layer name
    """

    def __init__(
        self,
        channels: int,
        initialiser: tf.keras.initializers.Initializer,
        activation: str,
        batchnorm: bool = True,
        first: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if activation not in Activation:
            raise ValueError(f"Activation {activation} not supported")
        self.activation = Activation[activation]

        # If first layer, use valid padding and no stride
        if first:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters=channels,
                kernel_size=(4, 4),
                strides=(1, 1),
                padding="VALID",
                kernel_initializer=initialiser,
                name="conv",
            )

        # Otherwise, use same padding and stride
        else:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters=channels,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
                kernel_initializer=initialiser,
                name="conv",
            )

        # Optionally add batch normalisation
        if batchnorm:
            self.bn = tf.keras.layers.BatchNormalization(name="batchnorm")
        else:
            self.bn = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through convolutional block."""
        x = self.conv(x)

        if self.bn:
            x = self.bn(x)

        return self.activation(x)


class Discriminator(tf.keras.Model):
    """Discriminator for DCGAN.

    Args:
        cfg: config
        name: model name
    """

    def __init__(self, cfg: DictConfig, name: str = "discriminator"):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 0.02)
        self.blocks = []

        # Get start and end resolutions
        assert cfg.img_dims[0] == cfg.img_dims[1], "Only square images supported"
        self._start_resolution = cfg.img_dims[0]
        self._end_resolution = 4
        self._img_dims = cfg.img_dims  # Needed for summary method

        # Get number of layers and channels
        self._num_layers = int(np.log2(self._start_resolution)) - int(
            np.log2(self._end_resolution),
        )
        self._channels = [
            np.min(
                [(cfg.discriminator.channels * 2**i), cfg.max_channels],
            )
            for i in range(self._num_layers)
        ]

        # Create layers
        self.blocks.append(
            DownBlock(
                channels=self._channels[0],
                initialiser=initialiser,
                activation=cfg.discriminator.activation,
                batchnorm=False,
                final=False,
                name=f"dn0",
            ),
        )

        for i in range(1, self._num_layers):
            self.blocks.append(
                DownBlock(
                    channels=self._channels[i],
                    initialiser=initialiser,
                    activation=cfg.discriminator.activation,
                    batchnorm=True,
                    final=False,
                    name=f"dn{i}",
                ),
            )

        # Add final conv layer or dense block as needed
        if cfg.discriminator.dense:
            self.blocks.append(tf.keras.layers.Flatten())
            self.blocks.append(
                DCDense(
                    units=1,
                    activation=cfg.discriminator.activation,
                    kernel_initializer=initialiser,
                    name="dense",
                ),
            )

        else:
            self.blocks.append(
                DownBlock(
                    channels=1,
                    initialiser=initialiser,
                    activation="linear",
                    final=True,
                    name="final",
                ),
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through discriminator."""
        for block in self.blocks:
            x = block(x)

        if x.ndim == 4:
            x = x[:, :, 0, 0]  # Ensure output dim [N, 1]

        return x

    @property
    def num_downsample(self) -> int:
        return self._num_layers

    @property
    def channels(self) -> list[int]:
        return self._channels

    def summary(self) -> None:
        """Print model summary."""
        x = tf.keras.layers.Input(self._img_dims)
        tf.keras.Model(inputs=[x], outputs=self.call(x), name="Discriminator").summary()


class Generator(tf.keras.layers.Layer):
    """Generator for DCGAN.

    Args:
        cfg: config
        name: model name
    """

    def __init__(self, cfg: DictConfig, name: str = "generator"):
        super().__init__(name=name)
        initialiser = tf.keras.initializers.RandomNormal(0, 0.02)
        self.blocks = []

        # Get start and end resolutions
        self._start_resolution = 4
        assert cfg.img_dims[0] == cfg.img_dims[1], "Only square images supported"
        self._end_resolution = cfg.img_dims[0]
        self._latent_dim = cfg.latent_dim  # Needed for summary method

        # Get number of layers and channels
        self._num_layers = int(np.log2(self._end_resolution)) - int(
            np.log2(self._start_resolution),
        )
        self._channels = [
            np.min(
                [(cfg.generator.channels * 2**i), cfg.max_channels],
            )
            for i in range(self._num_layers - 1, -1, -1)
        ]

        # Add initial conv layer or dense block as needed
        if cfg.generator.dense:
            dense_units = (
                self._start_resolution * self._start_resolution * self._channels[0]
            )
            self.blocks.append(
                DCDense(
                    units=dense_units,
                    activation=cfg.generator.activation,
                    kernel_initializer=initialiser,
                    name="dense",
                ),
            )
            self.blocks.append(
                tf.keras.layers.Reshape(
                    [self._start_resolution, self._start_resolution, self._channels[0]],
                ),
            )

        else:
            self.blocks.append(tf.keras.layers.Reshape([1, 1, cfg.latent_dim]))
            self.blocks.append(
                UpBlock(
                    channels=self._channels[0],
                    initialiser=initialiser,
                    activation=cfg.generator.activation,
                    batchnorm=True,
                    first=True,
                    name="up0",
                ),
            )

        for i in range(1, self._num_layers):
            self.blocks.append(
                UpBlock(
                    channels=self._channels[i],
                    initialiser=initialiser,
                    activation=cfg.generator.activation,
                    batchnorm=True,
                    first=False,
                    name=f"up{i}",
                ),
            )

        self.blocks.append(
            UpBlock(
                channels=cfg.img_dims[2],
                initialiser=initialiser,
                activation=cfg.generator.output,
                batchnorm=False,
                first=False,
                name="final",
            ),
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through generator."""
        for block in self.blocks:
            x = block(x)

        return x

    @property
    def num_downsample(self) -> int:
        return self._num_layers

    @property
    def channels(self) -> list[int]:
        return self._channels[::-1]

    def summary(self) -> None:
        """Print model summary."""
        x = tf.keras.layers.Input([self._latent_dim])
        tf.keras.Model(inputs=[x], outputs=self.call(x), name="Generator").summary()


@Registry.register(Categories.MODELS, "dcgan")
class DCGAN(BaseGAN):
    """Implementation of Deep Convolutional GAN.

    Args:
        cfg: configuration object
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

    @tf.function
    def train_step(self, real_images: tf.Tensor) -> dict[str, tf.Tensor]:
        """Perform one train step for DCGAN.

        Args:
            real_images: batch of real images

        Returns:
            dictionary of losses
        """
        self.discriminator_step(real_images)
        self.generator_step()

        return {"d_loss": self._d_metric.result(), "g_loss": self._g_metric.result()}

    def call(self, num_examples: int = 0) -> tf.Tensor:
        """Generate fake images from random noise.

        Args:
            num_examples: number of examples to generate (uses fixed noise if None)

        Returns:
            generated fake images
        """
        if num_examples == 0:
            imgs = self.generator(self.fixed_noise)

        else:
            latent_noise = tf.random.normal(
                (num_examples, self._latent_dim),
                dtype="float32",
            )
            imgs = self.generator(latent_noise)

        return imgs
