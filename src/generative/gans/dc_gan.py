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
