"""Modules for callbacks during training."""

import datetime
from pathlib import Path

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from generative.common.img_utils import save_images


def build_callbacks(
    cfg: DictConfig,
    num_batches: int,
) -> list[tf.keras.callbacks.Callback]:
    """Build callbacks for use in training.

    Args:
        cfg: config
        num_batches: number of batches in dataset

    Returns:
        list of callbacks
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    expt_dir = Path(cfg.run_folder) / cfg.run_name / timestamp
    expt_dir.mkdir(parents=True, exist_ok=True)

    with open(expt_dir / "config.yaml", "w") as fp:
        OmegaConf.save(config=cfg, f=fp)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=expt_dir / "logs",
        histogram_freq=cfg.save_freq,
        write_graph=True,
        write_images=True,
        write_steps_per_second=True,
        update_freq="epoch",
    )

    checkpoint_path = str(expt_dir / "checkpoints") + "/{epoch:03d}.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=cfg.save_freq * num_batches,
    )

    example_checkpoint = SaveExamples(expt_dir / "images", cfg)

    return [checkpoint_callback, tensorboard, example_checkpoint]


class SaveExamples(tf.keras.callbacks.Callback):
    """Save example images every N epochs.

    Args:
        filepath: path to save images
        save_freq: frequency to save images
    """

    def __init__(self, save_dir: Path, cfg: DictConfig) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._save_freq = cfg.save_freq
        self._normalisation = cfg.data.normalisation

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """Save example predictions.

        Args:
            epoch: current epoch
            logs: training logs
        """
        if (epoch + 1) % self._save_freq == 0:
            preds = self.model(num_examples=0)
            save_images(
                preds.numpy(),
                self._normalisation,
                self._save_dir / f"{epoch + 1}.png",
            )
