import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from generative.common.activations import Activation


class DCDense(tf.keras.layers.Dense):
    """Dense layer for DCGAN with activation function.

    Args:
        activation: activation function
        units: size of dense layer output
        kernel_initializer: initialiser for kernel weights
        name: layer name
    """

    def __init__(self, activation: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if activation not in Activation:
            raise ValueError(f"Activation {activation} not supported")
        self.activation = Activation[activation]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass through dense layer."""
        x = super(x)
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
        batchnorm=True,
        first=False,
        name=None,
    ):
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

    def call(self, x):
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
        assert cfg["img_dims"][0] == cfg["img_dims"][1], "Only square images supported"
        self._start_resolution = cfg["img_dims"][0]
        self._end_resolution = 4
        self._img_dims = cfg["img_dims"]  # Needed for summary method

        # Get number of layers and channels
        self._num_layers = int(np.log2(self._start_resolution)) - int(
            np.log2(self._end_resolution),
        )
        self._channels = [
            np.min(
                [(cfg.discriminator_channels * 2**i), cfg.max_channels],
            )
            for i in range(self.num_layers)
        ]

        # Create layers
        for i in range(self.num_layers):
            self.blocks.append(
                DownBlock(
                    channels=self._channels[i],
                    initialiser=initialiser,
                    activation=cfg.discriminator_activation,
                    batchnorm=True,
                    final=False,
                    name=f"dn{i}",
                ),
            )

        # Add final conv layer or dense block as needed
        if cfg.discriminator_dense:
            self.blocks.append(tf.keras.layers.Flatten())
            self.blocks.append(
                DCDense(
                    units=1,
                    activation=cfg.discriminator_activation,
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

        return x

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def channels(self) -> list[int]:
        return self._channels

    def summary(self) -> None:
        """Print model summary."""
        x = tf.keras.layers.Input(self._img_dims)
        tf.keras.Model(inputs=[x], outputs=self.call(x), name="Discriminator").summary()
