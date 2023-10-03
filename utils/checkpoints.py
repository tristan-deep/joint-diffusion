"""Load and save checkpoints
Author(s): Tristan Stevens
"""
import os
from pathlib import Path

import tensorflow as tf
import torch
from keras.callbacks import Callback

from utils.utils import check_model_library, get_latest_checkpoint


class ModelCheckpoint(Callback):
    def __init__(self, model, config, **kwargs):
        self.model = model

        self.model_library = check_model_library(model)

        # os independent path
        log_dir = "/".join(str(config.log_dir).split("\\"))
        # make relative path
        if "wandb" in str(log_dir):
            self.checkpoint_dir = Path("./wandb" + (log_dir).split("wandb")[-1])
        else:
            self.checkpoint_dir = Path(log_dir)

        self.checkpoint_dir = self.checkpoint_dir / "training_checkpoints"
        self.save_freq = config.get("save_freq")
        self.epochs = config.get("epochs")
        self.checkpoint_prefix = Path(self.checkpoint_dir, "ckpt")

        if config.get("pretrained"):
            self.pretrained = (
                self.checkpoint_dir.parents[2]
                / config.pretrained
                / "files"
                / "training_checkpoints"
            )
        else:
            self.pretrained = None

        self.model_name = config.model_name.lower()
        if self.model_name in ["gan", "wgan"]:
            if self.model.g_optimizer and self.model.d_optimizer:
                self.checkpoint = tf.train.Checkpoint(
                    generator_optimizer=self.model.g_optimizer,
                    discriminator_optimizer=self.model.d_optimizer,
                    generator=self.model.generator,
                    discriminator=self.model.discriminator,
                )
            else:
                self.checkpoint = tf.train.Checkpoint(
                    generator=self.model.generator,
                    discriminator=self.model.discriminator,
                )

        elif self.model_name in ["score"]:
            if self.model.optimizer:
                self.checkpoint = tf.train.Checkpoint(
                    optimizer=self.model.optimizer,
                    model=self.model.model,
                )
            else:
                self.checkpoint = tf.train.Checkpoint(
                    model=self.model.model,
                )
        elif self.model_name in ["ncsnv2", "unet"]:
            if self.model.optimizer:
                self.checkpoint = tf.train.Checkpoint(
                    optimizer=self.model.optimizer,
                    model=self.model,
                )
            else:
                self.checkpoint = tf.train.Checkpoint(
                    model=self.model,
                )
        elif self.model_name in ["glow"]:
            pass
        else:
            raise ValueError(f"ModelCheckpoint not supported for {self.model_name}")

    def on_train_begin(self, logs=None):
        if self.pretrained:
            file = self.get_checkpoint(file)
            self.restore(file)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0 or ((epoch + 1) == self.epochs):
            if self.model_library == "tensorflow":
                self.checkpoint.save(file_prefix=str(self.checkpoint_prefix))
            elif self.model_library == "torch":
                path = str(self.checkpoint_prefix) + f"-{epoch}.pt"
                Path(path).parent.mkdir(exist_ok=True, parents=True)
                torch.save(self.model.state_dict(), path)

            print(f"--> Succesfully saved weights (epoch: {epoch + 1})\n")

    def restore(self, file=None):
        """Load weights from checkpoint file."""
        file = self.get_checkpoint(file)
        if self.model_library == "tensorflow":
            status = self.checkpoint.restore(str(file.with_suffix(""))).expect_partial()
            status.assert_existing_objects_matched()
        elif self.model_library == "torch":
            status = self.model.load_state_dict(torch.load(file))

        print(f"--> Succesfully loaded weights from {file}")
        return status

    def get_checkpoint(self, file):
        """Get checkpoint file path from relative string or load latest checkpoint if None."""
        if file is None:
            if self.model_library == "tensorflow":
                file = tf.train.latest_checkpoint(str(self.checkpoint_dir))
            elif self.model_library == "torch":
                file = get_latest_checkpoint(self.checkpoint_dir, "pt", split="-")

            if file is None:
                raise ValueError(
                    f"No latest checkpoint file found in {self.checkpoint_dir} !"
                )

        elif not Path(file).is_absolute():
            if self.model_library == "tensorflow":
                file = self.checkpoint_dir / file
            elif self.model_library == "torch":
                file = self.checkpoint_dir / file

        if self.model_library == "tensorflow":
            file = Path(file).with_suffix(".index")
        elif self.model_library == "torch":
            file = Path(file).with_suffix(".pt")

        if not file.is_file():
            raise ValueError(
                f"Checkpoint file: {file.name} not found in {self.checkpoint_dir}!"
            )

        return file
