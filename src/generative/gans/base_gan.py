import abc

import tensorflow as tf
from omegaconf import DictConfig

from generative.common.optimisers import Optimiser
from generative.common.registry import Categories, Registry


class BaseGAN(tf.keras.Model, abc.ABC):
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

    def compile(self, cfg: DictConfig) -> None:
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
        """Generator training"""

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
        """Discriminator training"""

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
        raise NotImplementedError

    def summary(self) -> None:
        """Print model summary."""
        self.generator.summary()
        self.discriminator.summary()

    @abc.abstractmethod
    def call(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @property
    def metrics(self) -> list[tf.Tensor]:
        return [self._d_metric, self._g_metric]
