"""Corruptors
Author(s): Tristan Stevens
"""
import abc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datasets import get_dataset
from generators.models import get_model
from utils.checkpoints import ModelCheckpoint
from utils.runs import assert_run_exists, init_config
from utils.signals import add_gaussian_noise

_CORRUPTORS = {}


def register_corruptor(cls=None, *, name=None):
    """A decorator for registering corruptor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRUPTORS:
            raise ValueError(f"Already registered corruptor with name: {local_name}")
        _CORRUPTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_corruptor(name):
    """Get corruptor class for a given name."""
    return _CORRUPTORS[name]


class Corruptor(abc.ABC):
    """Corruptor abstract class"""

    def __init__(
        self, config, dataset_name=None, task=None, model=None, verbose=True, **kwargs
    ):
        """Init corruptor class.

        Args:
            config (dict): inference config object / dictionary. This config is the main dataset
                config, but will only be use partially and will be overwritten by the
                corruptor dataset config.
            dataset_name (str, optional): name of dataset see Dataset class.
                Defaults to None.
            task (str, optional): name of the task of corruption. Used for plotting.
                Defaults to None.
            model (bool, optional): whether trained model is used to model the corruption process.
                Defaults to None.
            verbose (bool, optional): enable print statements. Defaults to True.
        """
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.task = "denoising" if task is None else task

        self.name = config.corruptor
        self.model = model
        self.verbose = verbose
        self.batch_size = config.batch_size

        data_root = config.data_root
        self.image_shape = config.image_shape
        self.image_size = config.image_size
        self.A = None  # measurement matrix
        self.translation = config.get("translation")

        self.paired_data = config.paired_data
        if not self.paired_data:
            print("Corruptor assumes data is already corrupted!")
        self.data_type = config.get("data_type", "image")
        self.compression_type = config.get("compression_type", None)

        self.noise = None
        self.dataset = None
        self.gen = None

        if self.model:
            if (
                (config.get("disable_corruptor_model"))
                or (config.get("denoiser") is None)
                or (config.denoiser.lower() not in ["sgm", "glow"])
            ):
                self.model = False
                self.corruptor_run_id = config["sgm"].corruptor_run_id
                assert (
                    self.corruptor_run_id is not None
                ), "corruptor_run_id is not specified in config.sgm!"
            else:
                self.corruptor_run_id = config[config.denoiser.lower()].corruptor_run_id
                self.corruptor_checkpoint_file = config[config.denoiser.lower()].get(
                    "corruptor_checkpoint_file"
                )

            assert_run_exists(self.corruptor_run_id)
            # config is data config, and self.config is noise dataset config
            self.config = init_config(self.corruptor_run_id, verbose=self.verbose)

            # overwrite corruptor config with current config params
            self.config.data_root = data_root
            self.config.batch_size = self.batch_size
            self.config.data_type = self.data_type
            self.config.compression_type = self.compression_type
            self.config.image_size = self.image_size
            self.config.translation = self.translation

            if config.get("subsample_factor"):
                n_samples = np.prod(self.image_shape) // config.subsample_factor
                self.image_shape = [n_samples, 1, 1]
                self.config.image_shape = self.image_shape
            else:
                assert list(self.config.image_shape) == list(
                    self.image_shape
                ), "Except for CS experiments, image shapes of noise and data should match"

            if self.config.dataset_name != self.dataset_name:
                print(
                    f"Warning! Current run_id does not match with {self.dataset_name} dataset."
                )

            # load corruptor model
            if self.model:
                if self.verbose:
                    print("\nLoading corruptor model...")
                self.model = get_model(self.config, training=False)
                ckpt = ModelCheckpoint(self.model, config=self.config)

                if self.corruptor_checkpoint_file:
                    file = ckpt.checkpoint_dir / self.corruptor_checkpoint_file
                else:
                    file = None

                ckpt.restore(file=file)

                if ckpt.model_library == "torch":
                    if self.config.model_name.lower() in ["glow"]:
                        self.model.set_actnorm_init()
                    self.model.eval()

    def corrupt(self, images):
        """Corrupt input images with noise."""
        batch_size = tf.gather(tf.shape(images), 0)
        self.noise = tf.gather(next(self.gen), tf.range(batch_size))
        noisy_images = self.blend_factor * self.noise + (1 - self.blend_factor) * images
        return noisy_images

    def load_corruptor_dataset(self, train: bool):
        """Load dataset for corrupting data."""
        if self.verbose:
            print("\nLoading corruptor dataset...")
        train_dataset, test_dataset = get_dataset(self.config)
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset

        # create infinately repeating dataset
        self.dataset = self.dataset.repeat()

        # create generator from TF dataset
        self.gen = iter(self.dataset)

    def get_sensing_matrix(self):
        """Returns sensing matrix for inverse problem."""
        raise NotImplementedError


@register_corruptor(name="mnist")
class MNISTCorruptor(Corruptor):
    """MNIST corruptor, adds MNIST digits to data."""

    def __init__(self, config, train=True):
        super().__init__(config, dataset_name="mnist", model=True)

        self.load_corruptor_dataset(train)

        assert config.image_shape == self.config.image_shape, (
            f"Noise data shape {self.config.image_shape} should be similar "
            f"as image data shape {config.image_shape}."
        )

        self.blend_factor = config.noise_stddev

        # needed for callback
        self.noise_stddev = self.blend_factor


@register_corruptor(name="gaussian")
class GaussianCorruptor(Corruptor):
    """Gaussian corruptor, adds gaussian noise."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.noise_stddev = config.noise_stddev

    def corrupt(self, images):
        noisy_images = add_gaussian_noise(images, self.noise_stddev)
        self.noise = noisy_images - images
        return noisy_images


@register_corruptor(name="sinenoise")
class SinusoidalNoiseCorruptor(Corruptor):
    """Sinusoidal noise corruptor, adds sinusoidal noise to columns."""

    def __init__(self, config, **kwargs):
        super().__init__(config, dataset_name="sinenoise", model=True, **kwargs)
        self.noise_stddev = config.noise_stddev
        self.blend_factor = config.noise_stddev

    def corrupt(self, images):
        stddev = np.exp(np.sin(2 * np.pi * np.arange(images.shape[1]) / 16))
        stddev_matrix = np.transpose(np.zeros_like(images), (1, 2, 3, 0))
        stddev_matrix += stddev[:, None, None, None]
        stddev_matrix = np.transpose(stddev_matrix, (3, 0, 1, 2))
        self.noise = tf.random.normal(stddev_matrix.shape, stddev=stddev_matrix)
        self.noise *= self.noise_stddev / np.std(self.noise)
        noisy_images = images + self.noise
        return noisy_images


@register_corruptor(name="cs")
class CSCorruptor(Corruptor):
    """Compressed sensing corruptor, takes random weighted samples from data."""

    def __init__(self, config, **kwargs):
        super().__init__(config, task="compressive-sensing", **kwargs)
        self.noise_stddev = config.noise_stddev
        self.subsample_factor = config.subsample_factor
        self.image_shape = config.image_shape

        self.n = np.prod(self.image_shape)
        self.m = int(self.n * (1 / self.subsample_factor))
        self.A = self.get_sensing_matrix()

    def corrupt(self, images):
        noisy_images = add_gaussian_noise(images, self.noise_stddev)
        noisy_images = tf.reshape(noisy_images, (-1, self.n)) @ self.A.T
        return noisy_images

    def get_sensing_matrix(self):
        A = np.random.normal(0, 1 / np.sqrt(self.m), size=(self.m, self.n))
        return A.astype(np.float32)


@register_corruptor(name="cs_sine")
class CSSineCorruptor(Corruptor):
    """Compressed sensing corruptor with sinusoidal noise."""

    def __init__(self, config, **kwargs):
        super().__init__(config, dataset_name="sinenoise1d", model=True, **kwargs)
        self.noise_stddev = config.noise_stddev
        self.blend_factor = config.noise_stddev

        self.subsample_factor = config.subsample_factor
        self.x_shape = config.image_shape

        self.n = np.prod(self.x_shape)
        self.m = int(self.n * (1 / self.subsample_factor))

        # truncate matrix
        self.B = (np.eye(self.n)[: self.m]).astype(np.float32)
        self.A = self.get_sensing_matrix()

    def corrupt(self, images):
        images = tf.reshape(images, (-1, self.n)) @ self.A.T

        stddev = np.exp(np.sin(2 * np.pi * np.arange(self.m) / 16))
        stddev_matrix = np.zeros_like(images)
        stddev_matrix += stddev[None, :]
        self.noise = tf.random.normal(stddev.shape, stddev=stddev_matrix)
        self.noise *= self.noise_stddev / np.std(self.noise)

        noisy_images = images + self.noise
        return noisy_images

    def get_sensing_matrix(self):
        A = np.random.normal(0, 1 / np.sqrt(self.m), size=(self.m, self.n))
        return A.astype(np.float32)
