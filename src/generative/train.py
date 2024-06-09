"""Module for training models."""

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from generative.common.callbacks import build_callbacks
from generative.common.constants import DataSplits
from generative.common.dataloaders import get_dataset
from generative.common.registry import Categories, Registry


def get_model(cfg: DictConfig) -> tf.keras.Model:
    """Initialise and compile model.

    Args:
    ----
        cfg: config

    Returns:
    -------
        model: tf.keras.Model
    """
    model = Registry.build(Categories.MODELS, cfg.model_name, cfg)
    model.compile(cfg)
    model.summary()

    return model


@hydra.main(config_path="config/", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Entry point for training models."""
    train_dataset = get_dataset(cfg, DataSplits.TRAIN)
    num_batches = int(train_dataset.cardinality())
    callbacks = build_callbacks(cfg, num_batches)

    model = get_model(cfg.model)
    model.fit(x=train_dataset, epochs=cfg.epochs, callbacks=callbacks)


if __name__ == "__main__":
    train()
