"""Abstract base class for GANs."""

import abc

import tensorflow as tf
from omegaconf import DictConfig

from generative.common.optimisers import Optimiser
from generative.common.registry import Categories, Registry
from generative.gans.wasserstein import GANMetaclass


class BaseGAN(tf.keras.Model, metaclass=GANMetaclass):
    """Abstract base class for GANs.

    Args:
    ----
    cfg: config
    """

    discriminator: tf.keras.layers.Layer
    generator: tf.keras.layers.Layer

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(name=cfg.model_name)
        self._config = cfg
        self._latent_dim = cfg.latent_dim
        self._mb_size = cfg.batch_size
        self.fixed_noise = tf.random.normal(
            (cfg.num_examples, cfg.latent_dim),
            dtype="float32",
        )

    @abc.abstractmethod
    def build_generator(self, cfg: DictConfig, **kwargs: str) -> None:
        """Build generator using config and additional arguments."""
        raise ValueError

    @abc.abstractmethod
    def build_discriminator(self, cfg: DictConfig, **kwargs: str) -> None:
        """Build discriminator using config and additional arguments."""
        raise ValueError

    def compile(self, cfg: DictConfig) -> None:
        """Compile Keras model.

        Args:
        ----
        cfg: configuration object
        """
        super().compile(run_eagerly=False)

        if cfg.generator.opt not in Optimiser or cfg.discriminator.opt not in Optimiser:
            raise ValueError(
                f"Optimiser {(cfg.generator.opt, cfg.discriminator.opt)} not supported",
            )

        self._g_optimiser = Optimiser[cfg.generator.opt](**cfg.generator.opt_h_params)
        self._d_optimiser = Optimiser[cfg.discriminator.opt](
            **cfg.discriminator.opt_h_params,
        )

        self._loss = Registry.build(Categories.LOSSES, cfg.loss, None)
        self._d_metric = tf.keras.metrics.Mean(name="d_metric")
        self._g_metric = tf.keras.metrics.Mean(name="g_metric")

    def generator_step(self) -> None:
        """Generator training step."""

        latent_noise = tf.random.normal(
            (self._mb_size, self._latent_dim),
            dtype="float32",
        )

        # Get gradients from discriminator predictions and update weights
        with tf.GradientTape() as tape:
            fake_images = self.generator(latent_noise, training=True)
            pred = self.discriminator(fake_images, training=True)
            loss = self._loss(real=None, fake=pred)

        grads = tape.gradient(loss, self.generator.trainable_variables)
        self._g_optimiser.apply_gradients(
            zip(grads, self.generator.trainable_variables),
        )

        # Update metrics
        self._g_metric.update_state(loss)

    def discriminator_step(self, real_images: tf.Tensor) -> None:
        """Discriminator training step.

        Args:
        ----
        real_images: tensor of real images
        """

        # Select minibatch of real images and generate fake images
        real_mb_size = tf.shape(real_images)[0]
        latent_noise = tf.random.normal(
            (real_mb_size, self._latent_dim),
            dtype="float32",
        )

        fake_images = self.generator(latent_noise, training=True)
        input_batch = tf.concat([real_images, fake_images], axis=0)

        # Get gradients from discriminator predictions and update weights
        with tf.GradientTape() as tape:
            pred = self.discriminator(input_batch, training=True)
            loss = self._loss(
                real=pred[0:real_mb_size, ...],
                fake=pred[real_mb_size:, ...],
            )

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self._d_optimiser.apply_gradients(
            zip(grads, self.discriminator.trainable_variables),
        )

        # Update metrics
        self._d_metric.update_state(loss)

    @abc.abstractmethod
    def train_step(self, data: tf.Tensor) -> tf.Tensor:
        """Perform one train step for DCGAN.

        Args:
            real_images: batch of real images

        Returns:
            dictionary of losses
        """
        raise NotImplementedError

    def summary(self) -> None:
        """Print model summary."""
        self.generator.summary()
        self.discriminator.summary()

    @abc.abstractmethod
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Generate fake images from random noise.

        Args:
            num_examples: number of examples to generate (uses fixed noise if None)

        Returns:
            generated fake images
        """
        raise NotImplementedError

    @property
    def metrics(self) -> list[tf.Tensor]:
        return [self._d_metric, self._g_metric]
