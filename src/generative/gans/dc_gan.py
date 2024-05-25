import tensorflow as tf

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

        # If final layer, use valid padding and no downsampling
        if final:
            self.conv = tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=(4, 4),
                strides=(1, 1),
                padding="VALID",
                kernel_initializer=initialiser,
                name="conv",
            )

        # Otherwise, use same padding and downsampling
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
