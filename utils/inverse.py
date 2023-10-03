"""Inverse tasks
Author(s): Tristan Stevens
"""
import abc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import tqdm
from bm3d import BM3DStages, bm3d
from easydict import EasyDict as edict
from skimage.restoration import denoise_nl_means

from generators.models import get_model
from generators.SGM.sampling import ScoreSampler
from utils import opt
from utils.checkpoints import ModelCheckpoint
from utils.corruptors import Corruptor, get_corruptor
from utils.metrics import Metrics
from utils.utils import (
    convert_torch_tensor,
    get_date_filename,
    save_animation,
    tf_tensor_to_torch,
    timefunc,
)

_DENOISERS = {}
_MODEL_NAMES = {
    "gt": "Ground\nTruth",
    "noisy": "Input",
    "noise": "Noise",
    "gan": "GAN",
    "glow": "FLOW",
    "sgm": "DIFFUSION",
    "sgm_dps": "DPS",
    "sgm_proj": "Proj.",
    "sgm_pigdm": "$\Pi$GDM",
    "bm3d": "BM3D",
    "nlm": "NLM",
    "wvtcs": "LASSO",
}


def register_denoiser(cls=None, *, name=None):
    """A decorator for registering corruptor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _DENOISERS:
            raise ValueError(f"Already registered denoiser with name: {local_name}")
        _DENOISERS[local_name] = cls
        cls.name = local_name
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_denoiser(name):
    """Retrieve a denoiser object for a given name."""
    return _DENOISERS[name]


def get_list_of_denoisers():
    """Get all allowed denoiser names."""
    return list(_DENOISERS.keys()) + ["sgm_proj", "sgm_dps", "sgm_pigdm"]


def set_attributes_from_config(self, config, keys, values):
    """Set values from config as attribute of class.

    Only when the value for a key is None, the value from the config
    corresponding to the key is used. Else the value itself is used.
    This way, we are not overwriting values. When argument is not provided
    to the class __init__, the value from the config is used.

    Args:
        config (dict): config with keys and values.
        keys (list): list of keys from config to copy to attribute.
        values (list): list of values corresponding to the keys.
    """
    for key, value in zip(keys, values):
        if (key in config.keys()) and (value is None):
            setattr(self, key, getattr(config, key))
        else:
            setattr(self, key, value)


class Denoiser(abc.ABC):
    """Denoiser abstract class"""

    @abc.abstractmethod
    def __init__(
        self,
        config,
        dataset: tf.data.Dataset = None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        sweep_id: str = None,
        corruptor: Corruptor = None,
        verbose: bool = True,
    ):
        """Denoiser class solves inverse problems.

        Args:
            config (dict): config dict / object with hyperparams.
            dataset (tf.data.Dataset, optional): Necessary if no data is
                provided to the __call__ method. Automatically loads some
                data from the provided dataset in that case. Defaults to None.

            // the following parameters can also be set using config file.
            // however, when you set them using the arguments here, they
            // overwrite the config file values.
            num_img (int, optional): Number of images in a batch to denoise.
                Only gets used when dataset is provided. Gets overwritten with
                number of images in a manually provided batch to __call__.
                Defaults to None.
            metrics (list, optional): List of strings of which metrics to compute.
                Examples are `ssim`, `mse`, etc. see Metrics class. Defaults to None.
            keep_track (bool, optional): Keep track of intermediate denoise steps.
                Useful for making animation afterwards. Defaults to None, only keeps
                the final result.
            sweep_id (str, optional): string which denotes what hyperparameters sweep
                is being done, if any. Defaults to None.
            // end of config file parameters
            corruptor (Corruptor, optional): Corruptor class. Defaults to None, in that
                case a new Corruptor class is instantiated here. If provided, that corruptor
                class is used. Useful to reuse the same corruption with multiple denoisers.
            verbose (bool, optional): whether to output print statements. Defaults to True.
        """
        config.denoiser = self.name

        self.config = config
        self.dataset = dataset
        self.vmin, self.vmax = config.image_range
        self.verbose = verbose

        self.model = None

        self.target_samples = None
        self.noisy_samples = None
        self.denoised_samples = None

        self.zoom = None
        self.image_shape = None

        # self.num_img = None
        # self.metrics = None
        # self.keep_track = None
        # self.sweep_id = None
        keys = [
            "num_img",
            "metrics",
            "keep_track",
            "sweep_id",
        ]
        values = []
        for key in keys:
            values.append(eval(key))
        set_attributes_from_config(self, config, keys, values)

        # batch size can differ from model to model
        # max inference batch size for generative models.
        self.batch_size = None
        if self.name in config:
            if "batch_size" in config[self.name]:
                self.batch_size = self.config[self.name].batch_size
                self.config.batch_size = self.batch_size

        if corruptor is None:
            self.corruptor = get_corruptor(self.config.corruptor)(config, train=False)
        else:
            self.corruptor = corruptor

        if self.metrics:
            self.eval_noisy = None
            self.eval_denoised = None
            if self.config.paired_data:
                self.metrics = Metrics(self.metrics, self.config.image_range)
            else:
                self.metrics = None

        self.model_names = _MODEL_NAMES

    def __call__(
        self,
        noisy_samples=None,
        target_samples=None,
        plot=True,
        save=True,
    ):
        # if self.model is not None:
        #     self.model.summary()

        # get / set data
        self.set_data(noisy_samples, target_samples)

        # body of call, can switch out for each different denoiser
        self._call()

        # evaluate metrics
        if self.metrics:
            self.get_metrics()

        # plot results
        if plot:
            self.plot(save=save, figsize=self.config.figsize)

        # return denoised samples
        return self.denoised_samples

    def _call(self):
        self.denoised_samples = self._denoise(self.noisy_samples)

    @abc.abstractmethod
    def _denoise(self, images):
        return images

    def set_data(self, noisy_samples=None, target_samples=None):
        """Set data method

        Sets data for the inverse problem. Noisy samples can be either
        supplied or generated if not. Same holds for the target data.

        Args:
            noisy_samples (ndarray, optional): Batch of noisy samples.
                Defaults to None. In that case noisy samples are generated
                by corrupting the target samples with corruptor class.
            target_samples (ndarray, optional): Batch of target samples.
                Defaults to None. In that case a random batch of target samples
                are loaded using the data loader.

        Raises:
            ValueError: Dataset should be assigned during init
                if target_samples is not provided.

        Returns:
            tuple: (noisy_samples, target_samples)
        """
        if target_samples is None:
            if self.dataset:
                self.dataset = self.dataset.unbatch().batch(self.num_img)
                self.target_samples = next(iter(self.dataset))
            else:
                raise ValueError(
                    "Either provide a dataset during init of "
                    "denoiser or provide data to run()"
                )
        else:
            self.target_samples = target_samples
            self.num_img = len(self.target_samples)

        if self.config.paired_data:
            if noisy_samples is None:
                self.noisy_samples = self.corruptor.corrupt(self.target_samples)
            else:
                self.noisy_samples = noisy_samples
        else:
            self.noisy_samples = self.target_samples

        return self.noisy_samples, self.target_samples

    def get_metrics(self):
        """Get metrics after inference."""
        if self.metrics:
            if self.config.corruptor not in ["cs", "sr", "cs_sine"]:
                self.eval_noisy = self.metrics.eval_metrics(
                    self.target_samples, self.noisy_samples
                )
            else:
                self.eval_noisy = None

            if self.corruptor.model:
                denoised_samples, noise_samples = self.denoised_samples
                if self.keep_track:
                    denoised_samples, noise_samples = (
                        denoised_samples[-1],
                        noise_samples[-1],
                    )
            else:
                denoised_samples = self.denoised_samples
                if self.keep_track:
                    denoised_samples = denoised_samples[-1]

            self.eval_denoised = self.metrics.eval_metrics(
                self.target_samples, denoised_samples
            )

            if self.verbose:
                self.metrics.print_results(self.eval_denoised)

    def plot(
        self,
        save=True,
        zoom=None,
        fig=None,
        axs=None,
        show_metrics=True,
        figsize=None,
        dpi=600,
    ):
        """Plot results after inference."""
        denoised_samples = self.denoised_samples
        self.zoom = zoom

        if self.corruptor.model:
            denoised_samples, noise_samples = denoised_samples
            if self.keep_track:
                denoised_samples, noise_samples = (
                    denoised_samples[-1],
                    noise_samples[-1],
                )
            titles = [
                "Ground Truth",
                "Noisy",
                self.model_names[self.name],
                "Noise Posterior",
            ]
            samples = [
                self.target_samples,
                self.noisy_samples,
                denoised_samples,
                noise_samples,
            ]

            if self.config.paired_data:
                noise = self.corruptor.noise
                samples.append(noise)
                titles.append("actual noise")

            width = 8

            if self.config.corruptor in ["cs", "sr", "cs_sine"]:
                samples.pop(1)
                titles.pop(1)
                samples.pop(-1)
                titles.pop(-1)
                samples.pop(-1)
                titles.pop(-1)
            else:
                if self.config.show_noise_priors:
                    noise_priors = self.corruptor.model.sample(
                        shape=self.target_samples.shape
                    )
                    samples.append(noise_priors)
                    titles.append("noise prior")

        else:
            titles = ["ground truth", "noisy", self.model_names[self.name]]
            if self.keep_track:
                denoised_samples = denoised_samples[-1]
            samples = [self.target_samples, self.noisy_samples, denoised_samples]
            width = 5

            if self.config.corruptor in ["cs", "sr", "cs_sine"]:
                samples.pop(1)
                titles.pop(1)

        # remove target_data (equals None if not paired data)
        if not self.config.paired_data:
            samples = samples[1:]
            titles = titles[1:]
            width -= 1

        samples = [
            (tf.clip_by_value(sample, self.vmin, self.vmax)).numpy()
            for sample in samples
        ]

        num_img = len(self.noisy_samples)
        fig_contents = []
        if fig is None:
            if figsize is None:
                figsize = (width, int(num_img * 1.5))
            fig, axs = plt.subplots(
                num_img,
                len(samples),
                figsize=figsize,
            )
            if num_img == 1:
                axs = np.expand_dims(axs, 0)

        for n in range(num_img):
            for i, (sample, title) in enumerate(zip(samples, titles)):
                image = np.squeeze(sample[n])

                vmin, vmax = self.vmin, self.vmax

                if self.config.color_mode != "rgb" and len(image.shape) == 3:
                    image = image[..., 0]

                im = axs[n, i].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
                fig_contents.append(im)
                if zoom:
                    x_zoom, y_zoom, zoom_size = zoom
                    axs[n, i].set_xlim(x_zoom, x_zoom + zoom_size)
                    axs[n, i].set_ylim(y_zoom + zoom_size, y_zoom)
                axs[n, i].axis("off")
                if n == 0:
                    axs[n, i].set_title(title)

        if bool(self.metrics) & show_metrics:
            offset = image.shape[0] * 0.1
            fontsize = 7
            color = "yellow"

            for n in range(num_img):
                if self.eval_noisy is None:
                    iterator = zip([1], [self.eval_denoised])
                else:
                    iterator = zip([1, 2], [self.eval_noisy, self.eval_denoised])
                for row, evals in iterator:
                    for m, metric in enumerate(self.metrics.metrics):
                        evaluation = evals[metric][n]
                        axs[n, row].text(
                            0,
                            offset + offset * m,
                            f"{metric}: {evaluation:.3f}",
                            color=color,
                            fontsize=fontsize,
                        )

        title = f"{self.model_names[self.name]} {self.corruptor.task} "

        if self.config.paired_data:
            if hasattr(self.corruptor, "noise_stddev"):
                title += f"with $\sigma$={self.corruptor.noise_stddev:.2f}"
        if (
            self.config.subsample_factor is not None
            and self.config.subsample_factor > 1
        ):
            title += f", $\downarrow_s$={self.config.subsample_factor}"
        fig.suptitle(title)

        fig.tight_layout()

        if save:
            self.savefig(fig, dpi=dpi, path=save)

        return fig, fig_contents

    def savefig(self, fig, dpi=600, path=None):
        """Save figure to png."""
        if isinstance(path, (Path, str)):
            path = path
        else:
            filename = (
                f"{self.name.lower()}_{self.config.dataset_name}_{self.corruptor.task}"
            )
            if self.zoom:
                filename += "_zoom"

            folder = "figures"
            if self.sweep_id:
                folder += f"/{self.sweep_id}"

            path = get_date_filename(f"{folder}/{filename}.png")

        fig.savefig(path, dpi=dpi)
        print(f"Succesfully saved plot to {path}")

    def animate(self, downsample: int = None, ending=10, duration=None, fps=30):
        """Animate the iterations of the generative process.

        Args:
            denoiser (Denoiser): denoiser object.
            downsample (int, optional): list with denoised images.
            duration (optional): duration of the animation in seconds. If provided,
                fps argument will be ignored.
            fps (int, optional): frames per second of animation. Defaults to 30 fps.

        """
        if not self.keep_track:
            print("Cannot animate since `keep_track` is set to False!")
            return False

        if self.corruptor.model:
            denoised_samples, noise_samples = self.denoised_samples
            denoised_samples = [
                (dn, ns) for dn, ns in zip(denoised_samples, noise_samples)
            ]
        else:
            denoised_samples = self.denoised_samples

        # set parameters temporarily for plotting
        temp = list(denoised_samples)
        self.keep_track = False

        num_img = len(self.target_samples)

        if self.corruptor.model:
            n_columns = 5
        else:
            n_columns = 3
        if self.config.corruptor in ["cs", "sr", "cs_sine"]:
            n_columns = 2

        fig_contents = []
        fig, axs = plt.subplots(num_img, n_columns, figsize=(num_img, 8.5))

        n_frames = len(denoised_samples)

        if downsample is None:
            if n_frames > 100:
                downsample = 10
            else:
                downsample = 1

        denoised_samples = list(denoised_samples[::downsample]) + [
            denoised_samples[-1]
        ] * int(ending / 100 * n_frames)
        for ds in denoised_samples:
            self.denoised_samples = ds
            fig, contents = self.plot(fig=fig, axs=axs, save=False, show_metrics=False)
            fig_contents.append(contents)

        # restore parameters
        self.keep_track = True
        self.denoised_samples = temp

        filename = get_date_filename(
            f"figures/{self.name.lower()}_{self.config.dataset_name}_"
            f"{self.corruptor.task}_animation.gif"
        )

        if duration:
            fps = len(fig_contents) / duration
        save_animation(fig, fig_contents, filename=filename, fps=fps)
        plt.close()

        return fig


@register_denoiser(name="none")
class NoneDenoiser(Denoiser):
    """None denoiser, does nothing."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        **kwargs,
    ):
        super().__init__(
            config,
            dataset,
            num_img,
            metrics,
            keep_track=False,
            **kwargs,
        )

    def _denoise(self, images):
        if self.verbose:
            print("\nNone Denoiser...")
        denoised_images = np.array(images)
        return denoised_images


@register_denoiser(name="bm3d")
class BM3DDenoiser(Denoiser):
    """Block matching 3D denoiser."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        stage="all_stages",
        **kwargs,
    ):
        super().__init__(
            config,
            dataset,
            num_img,
            metrics,
            keep_track=False,
            **kwargs,
        )

        self.stddev = self.corruptor.noise_stddev

        str_to_stage = {
            "hard_thresholding": BM3DStages.HARD_THRESHOLDING,
            "all_stages": BM3DStages.ALL_STAGES,
        }

        self.stage = str_to_stage[stage]

    @timefunc
    def _denoise(self, images):
        if self.verbose:
            print("\nBM3D Denoiser...")
        denoised_images = []
        for image in images:
            denoised_image = bm3d(
                image,
                self.stddev,
                stage_arg=self.stage,
            )

            denoised_images.append(denoised_image)
        denoised_images = np.stack(denoised_images)

        if self.config.color_mode == "grayscale":
            denoised_images = np.expand_dims(denoised_images, axis=-1)

        return denoised_images


@register_denoiser(name="nlm")
class NLMDenoiser(Denoiser):
    """Non-local means denoiser."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        patch_size=6,
        patch_distance=5,
        **kwargs,
    ):
        super().__init__(
            config,
            dataset,
            num_img,
            metrics,
            keep_track=False,
            **kwargs,
        )

        self.stddev = self.corruptor.noise_stddev

        if self.config.color_mode == "rgb":
            channel_axis = -1
        else:
            channel_axis = None

        self.patch_kw = dict(
            patch_size=patch_size,  # patches size
            patch_distance=patch_distance,  # search area
            channel_axis=channel_axis,
        )

    def _denoise(self, images):
        if self.verbose:
            print("\nNLM Denoiser...")
        denoised_images = []
        images = np.squeeze(images)
        for image in images:
            denoised_image = denoise_nl_means(
                image,
                h=0.6 * self.stddev,
                sigma=self.stddev,
                fast_mode=True,
                **self.patch_kw,
            )

            denoised_images.append(denoised_image)

        denoised_images = np.stack(denoised_images)
        if self.config.color_mode == "grayscale":
            denoised_images = np.expand_dims(denoised_images, axis=-1)

        return denoised_images


@register_denoiser(name="wvtcs")
class WVTCS(Denoiser):
    """Wavelet cosine transform LASSO."""

    def __init__(
        self,
        config,
        dataset=None,
        num_img=None,
        metrics=None,
        **kwargs,
    ):
        super().__init__(
            config,
            dataset,
            num_img,
            metrics,
            keep_track=False,
            **kwargs,
        )
        if "wvtcs" in self.config:
            self.hyperparams = self.config.wvtcs
            self.hyperparams.size = self.config.image_size
            self.hyperparams.n_channels = self.config.image_shape[-1]
        else:
            raise ValueError("Did not find WVTCS in config")

    def _denoise(self, images):
        if self.verbose:
            print("\nWVTCS Denoiser...")
        self.hyperparams.batch_size = images.shape[0]

        A = self.corruptor.A

        estimator = opt.lasso_wavelet_estimator(self.hyperparams)
        x_hat = estimator(
            np.sqrt(2 * self.corruptor.m) * A.T,
            np.sqrt(2 * self.corruptor.m) * images.numpy(),
            self.hyperparams,
        )
        x_hat = np.array(x_hat)
        x_hat = x_hat.reshape(-1, *self.config.image_shape)

        x_hat = np.clip(x_hat, 0, 1)

        return x_hat


@register_denoiser(name="sgm")
class SGMDenoiser(Denoiser):
    """Score-based generative model (diffusion model) denoiser."""

    def __init__(
        self,
        config,
        dataset: tf.data.Dataset = None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            num_img=num_img,
            metrics=metrics,
            keep_track=keep_track,
            **kwargs,
        )
        self.config = edict({**self.config, **self.config.sgm})

        self.model = get_model(self.config, training=False)
        ckpt = ModelCheckpoint(self.model, config=self.config)
        ckpt.restore(self.config.get("checkpoint_file"))
        self.set_sampler()

    def set_sampler(self):
        """Reset diffusion sampler with updated config parameters."""
        self.model.set_sde(self.config)
        self.sampler = ScoreSampler(
            self.model,
            self.config.image_shape,
            self.model.sde,
            self.config.sampling_method,
            self.config.predictor,
            self.config.corrector,
            self.config.guidance,
            corruptor=self.corruptor,
            keep_track=self.config.keep_track,
            corrector_snr=self.config.snr,
            lambda_coeff=self.config.lambda_coeff,
            kappa_coeff=self.config.get('kappa_coeff'),
            noise_model=self.corruptor.model,
            noise_shape=self.corruptor.image_shape,
            start_diffusion=self.config.get('ccdf'),
            sampling_eps=self.config.get('sampling_eps'),
            early_stop=self.config.get('early_stop'),
        )

    @timefunc
    def _call(self):
        self.denoised_samples = self._denoise(self.noisy_samples)

    def _denoise(self, images):
        denoised_samples = self.sampler(y=images, progress_bar=self.verbose)
        return denoised_samples

    def plot(
        self,
        save=True,
        zoom=None,
        fig=None,
        axs=None,
        show_metrics=True,
        figsize=None,
        dpi=600,
    ):
        fig, fig_contents = super().plot(
            False, zoom, fig, axs, show_metrics, figsize, dpi
        )

        title = f"{self.model_names[self.name]} {self.corruptor.task} "

        if self.config.paired_data:
            if hasattr(self.corruptor, "noise_stddev"):
                title += f"with $\sigma$={self.corruptor.noise_stddev:.2f}"

        if self.config.lambda_coeff:
            title += f", $\lambda$={np.round(self.config.lambda_coeff, 5)}"
        if self.corruptor.model:
            title += f", $\kappa$={np.round(self.config.kappa_coeff, 5)}"

        fig.suptitle(title)
        fig.tight_layout()

        if save:
            self.savefig(fig, path=save)

        return fig, fig_contents


@register_denoiser(name="gan")
class GANDenoiser(Denoiser):
    """Generative adversarial network denoiser."""

    def __init__(
        self,
        config,
        dataset: tf.data.Dataset = None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        **kwargs,
    ):
        super().__init__(
            config,
            dataset,
            num_img,
            metrics,
            keep_track,
            **kwargs,
        )

        self.config = edict({**self.config, **self.config.gan})

        self.model = get_model(config, training=False)
        ckpt = ModelCheckpoint(self.model, config=config)
        ckpt.restore(self.config.get("checkpoint_file"))

    def _denoise(self, images):
        denoised_images = self.optimize(images)
        return denoised_images

    @timefunc
    def optimize(self, y, initial_vector=None):
        X = []
        metrics = []
        metrics_str = []
        losses = {
            "log_p_y_z": [],
            "log_p_z": [],
            "loss": [],
        }

        latent_dim = self.model.generator.input_shape[-1]

        if initial_vector is None:
            initial_latent_vector = tf.random.normal([len(y), latent_dim])
        z_estimate = tf.Variable(initial_latent_vector)

        if self.config.get("scheduler") is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.step_size,
                decay_steps=self.config.scheduler.decay_steps,
                decay_rate=self.config.scheduler.decay_rate,
            )
            optimizer = tf.optimizers.Adam(lr_schedule)
        else:
            optimizer = tf.optimizers.Adam()

        # self.loss_fn = tf.losses.MeanAbsoluteError(reduction='sum')
        self.loss_fn = tf.losses.MeanSquaredError(reduction="sum")

        with tqdm.tqdm(
            range(self.config.num_steps),
            desc="GAN denoiser",
            disable=(not self.verbose),
        ) as pbar:
            for step in pbar:
                with tf.GradientTape() as tape:
                    # TODO: currently hardcoded image range from [-1, 1] to [0, 1] conversion
                    x = (self.model.generator(z_estimate.read_value()) + 1) / 2
                    if self.keep_track:
                        X.append(x)

                    log_p_y_z = self._log_p_y_z(x, y)

                    # The latent vectors were sampled from a normal distribution. We can get
                    # more realistic images if we regularize the length of the latent vector to
                    # the average length of vector from this distribution.
                    if self.config.prior == "z_squared":
                        log_p_z = tf.abs(tf.norm(z_estimate) ** 2)
                    elif self.config.prior == "z":
                        log_p_z = tf.abs(tf.norm(z_estimate))
                    # regularizer = tf.abs(tf.norm(vector) - np.sqrt(latent_dim))

                    loss = self.config.lambda_coeff * log_p_y_z + log_p_z

                grads = tape.gradient(loss, [z_estimate])
                optimizer.apply_gradients(zip(grads, [z_estimate]))

                if self.verbose:
                    losses["log_p_y_z"].append(log_p_y_z.numpy())
                    losses["log_p_z"].append(log_p_z.numpy())
                    losses["loss"].append(loss.numpy())

                    if self.metrics is not None:
                        result = self.metrics.eval_metrics(
                            self.target_samples,
                            x,
                        )
                        metrics.append(result)
                        metrics_str.append(
                            self.metrics.print_results(result, to_screen=False)
                        )
                    else:
                        metrics_str = ["NA"]

                    loss_str = ", ".join(
                        [f"{key}: {value[-1]:.3f}" for key, value in losses.items()]
                    )
                    pbar.set_description(
                        f"GAN denoiser: step {step + 1}, {loss_str}, {metrics_str[-1]}"
                    )

        if self.keep_track:
            return X
        else:
            return x

    def _log_p_y_z(self, x, y):
        if self.corruptor.name == "mnist":
            x = (1 - self.corruptor.blend_factor) * x

        if self.config.corruptor == "sr":
            x = self.corruptor.decimate(x)
            y = self.corruptor.decimate(y)

        if self.config.corruptor in ["cs", "cs_sine"]:
            x = tf.reshape(x, (-1, self.corruptor.n)) @ self.corruptor.A.T

        log_p_y_z = self.loss_fn(x, y)

        return log_p_y_z


@register_denoiser(name="glow")
class GlowDenoiser(Denoiser):
    """Glow (Normalizing Flow) denoiser."""

    def __init__(
        self,
        config,
        dataset: tf.data.Dataset = None,
        num_img: int = None,
        metrics: list = None,
        keep_track: bool = None,
        **kwargs,
    ):
        super().__init__(config, dataset, num_img, metrics, keep_track, **kwargs)

        self.config = edict({**self.config, **self.config.glow})

        self.device = self.config.device

        self.model = get_model(config, training=False)
        ckpt = ModelCheckpoint(self.model, config=config)
        ckpt.restore(self.config.get("checkpoint_file"))

        self.model.set_actnorm_init()
        self.model.eval()

        self.show_optimization_progress = self.config.get("show_optimization_progress")

    @timefunc
    def _denoise(self, y):
        y = (
            tf_tensor_to_torch(y, device=self.config.device)
            .detach()
            .requires_grad_(False)
        )

        X = []
        losses = {
            "log_p_y_z": [],
            "log_p_z": [],
            "loss": [],
        }
        metrics = []
        metrics_str = []
        lr = []

        batch_size = len(y)
        x_example = tf_tensor_to_torch(self.target_samples)
        n = np.prod(x_example.shape[1:])

        self.lambda_coeff = torch.tensor(
            self.config.lambda_coeff,
            requires_grad=True,
            dtype=torch.float,
            device=self.device,
        )

        # making a forward to record shapes of z's for reverse pass
        _ = self.model(self.model.preprocess(torch.zeros_like(x_example)))

        if self.config.init_strat == "random":
            z_estimate = np.random.normal(0, self.config.init_std, [batch_size, n])
            z_estimate = torch.tensor(
                z_estimate, requires_grad=True, dtype=torch.float, device=self.device
            )
        elif self.config.init_strat == "observation":
            y_processed = self.model.preprocess(y * 255, clone=True)
            z, _, _ = self.model(y_processed)
            z = self.model.flatten_z(z)
            z_estimate = z.clone().detach().requires_grad_(True)
        else:
            raise NotImplementedError(
                f"Unknown initialization strategy {self.config.init_strat}"
            )

        # selecting optimizer
        if self.config.optim == "adam":
            optimizer = torch.optim.Adam(
                [z_estimate],
                lr=self.config.step_size,
            )
        elif self.config.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [z_estimate],
                lr=self.config.step_size,
            )

        # lr_sch_metric = self.config.scheduler.pop('metric', self.metrics.metrics[0])

        if self.config.get("scheduler") is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **self.config.scheduler,
            )
        else:
            scheduler = None

        with tqdm.tqdm(
            range(self.config.num_steps),
            desc="Glow denoiser",
            disable=(not self.verbose),
        ) as pbar:
            for step in pbar:
                optimizer.zero_grad()
                z_unflat = self.model.unflatten_z(z_estimate, clone=False)
                x = self.model(z_unflat, reverse=True, reverse_clone=False)
                x = self.model.postprocess(x, floor_clamp=False)

                if self.corruptor.model:
                    log_p_y_z = self._log_p_y_z(x, y, noise_model="trained")
                else:
                    log_p_y_z = self._log_p_y_z(x, y, noise_model="additive_gaussian")

                if self.config.prior == "z_squared":
                    log_p_z = -(z_estimate.norm(dim=1) ** 2).mean()
                elif self.config.prior == "z":
                    log_p_z = -z_estimate.norm(dim=1).mean()
                elif self.config.prior == "log_p_x":
                    nll = self.model.nll_loss(self.model.preprocess(x * 255))[0]
                    log_p_z = -nll
                elif self.config.prior is None:
                    log_p_z = torch.tensor(0.0)
                else:
                    raise ValueError(f"Unknown prior {self.config.prior}")

                log_p_z = torch.nan_to_num(log_p_z, nan=0.0, posinf=0.0, neginf=0.0)
                log_p_y_z = torch.nan_to_num(log_p_y_z, nan=0.0, posinf=0.0, neginf=0.0)

                losses["log_p_y_z"].append(log_p_y_z.item())
                losses["log_p_z"].append(log_p_z.item())

                loss = -log_p_y_z - log_p_z
                loss.backward(retain_graph=True)

                losses["loss"].append(loss.item())

                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x_out = convert_torch_tensor(x)

                if self.verbose:
                    if self.metrics is not None:
                        result = self.metrics.eval_metrics(
                            self.target_samples,
                            x_out,
                        )
                        metrics.append(result)
                        metrics_str.append(
                            self.metrics.print_results(result, to_screen=False)
                        )
                    else:
                        metrics_str = ["NA"]

                if self.keep_track:
                    X.append(x_out)

                lr.append(optimizer.param_groups[0]["lr"])

                optimizer.step()

                # z_estimate = torch.nan_to_num(z_estimate, nan=0.0, posinf=0.0, neginf=0.0)

                # metric = np.mean(metrics[-1][lr_sch_metric]) * self.metrics.loss_multiplier[lr_sch_metric]
                if scheduler:
                    scheduler.step(loss)

                if self.verbose:
                    loss_str = ", ".join(
                        [f"{key}: {value[-1]:.3f}" for key, value in losses.items()]
                    )
                    pbar.set_description(
                        f"Glow denoiser: step {step + 1}, {loss_str}, {metrics_str[-1]}"
                    )

        if self.show_optimization_progress and self.verbose:
            self.plot_optimization_progress(losses, metrics, lr)

        z_unflat = self.model.unflatten_z(z_estimate, clone=False)
        x_gen = self.model(z_unflat, reverse=True, reverse_clone=False)
        x_gen = self.model.postprocess(x_gen, floor_clamp=False)
        x = convert_torch_tensor(x_gen)

        self.model.zero_grad()
        optimizer.zero_grad()
        del x_gen, optimizer, z_estimate
        torch.cuda.empty_cache()

        if self.corruptor.model:
            beta = self.corruptor.blend_factor
            if self.corruptor.name == "mnist":
                h = (convert_torch_tensor(y) - (1 - beta) * x) / beta
                H = [h] * len(X)
            else:
                h = None
                H = []
            return X if self.keep_track else x, H if self.keep_track else h
        else:
            return X if self.keep_track else x

    def _log_p_y_z(self, x, y, noise_model="additive_gaussian"):
        if noise_model == "additive_gaussian":
            if self.config.corruptor == "cs":
                # first permute torch tensor as the A matrix is applied to [b, y, x, c]
                x = torch.permute(x, (0, 2, 3, 1))
                x = torch.matmul(
                    torch.reshape(x, (-1, self.corruptor.n)),
                    tf_tensor_to_torch(self.corruptor.A.T),
                )
                delta = torch.square(y - x)
                return -self.lambda_coeff * torch.sum(delta, dim=-1).mean()
            elif self.config.corruptor == "sr":
                x = self.corruptor.decimate(x, torch_tensor=True)
                y = self.corruptor.decimate(y, torch_tensor=True)
                delta = torch.square(y - x)
                return -self.lambda_coeff * torch.sum(delta, dim=[1, 2, 3]).mean()
            else:
                delta = torch.square(y - x)
                return -self.lambda_coeff * torch.sum(delta, dim=[1, 2, 3]).mean()
        elif noise_model == "multiplicative_gaussian":
            delta = torch.square(y / (x - 1))
            return -self.lambda_coeff * torch.sum(delta, dim=[1, 2, 3]).mean()

        elif noise_model == "trained":
            if self.corruptor.name == "mnist":
                beta = self.corruptor.blend_factor
                delta = (y - (1 - beta) * x) / beta

                delta = self.corruptor.model.preprocess(delta * 255)
            elif self.corruptor.name == "sinenoise":
                delta = y - x
                delta = self.corruptor.model.preprocess(delta * 255)

            elif self.corruptor.name == "cs_sine":
                # first permute torch tensor as the A matrix is applied to [b, y, x, c]
                x = torch.permute(x, (0, 2, 3, 1))
                x = torch.matmul(
                    torch.reshape(x, (-1, self.corruptor.n)),
                    tf_tensor_to_torch(self.corruptor.A.T),
                )

                delta = y - x
                delta = delta[:, None, :, None]
            else:
                raise ValueError(f"Unknown corruptor {self.corruptor.name}")

            nll, logdet, logpz, z_mu, z_std = self.corruptor.model.nll_loss(delta)
            log_p_y_z = -self.lambda_coeff * nll
            return log_p_y_z

    def plot(
        self,
        save=True,
        zoom=None,
        fig=None,
        axs=None,
        show_metrics=True,
        figsize=None,
        dpi=600,
    ):
        fig, fig_contents = super().plot(
            False, zoom, fig, axs, show_metrics, figsize, dpi
        )

        title = f"{self.model_names[self.name]} {self.corruptor.task} "

        if self.config.paired_data:
            if hasattr(self.corruptor, "noise_stddev"):
                title += f"with $\sigma$={self.corruptor.noise_stddev:.2f}"
        if self.config.lambda_coeff:
            title += f", $\lambda$={np.round(self.config.lambda_coeff, 5)}"
        if self.config.prior:
            prior_str = {
                "z": "$||z||$",
                "z_squared": "$||z||^2$",
                "log_p_x": "$\log{p(x)}$",
            }
            title += f", prior={prior_str[self.config.prior]}"
        if self.config.step_size:
            title += f", $\Delta t$={np.round(self.config.step_size, 3)}"

        fig.suptitle(title)
        fig.tight_layout()

        if save:
            self.savefig(fig, path=save)

        return fig, fig_contents

    def plot_optimization_progress(self, losses, metrics, lr=None):
        losses.pop("loss")
        metrics = self.metrics.parse_metrics(metrics, reduce_mean=True)

        def create_plot_from_dict(dic, lr=None, title=None):
            assert len(dic.keys()) == 2
            if lr:
                fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]})
                lr_ax, dic_ax = axes
            else:
                fig, dic_ax = plt.subplots()

            dic_ax2 = dic_ax.twinx()
            dic_axes = [dic_ax, dic_ax2]
            colors = ["red", "blue"]
            for ax, (key, value), c in zip(dic_axes, dic.items(), colors):
                ax.plot(value, color=c)
                ax.set_xlabel("step")
                ax.set_ylabel(key, color=c)

            if lr:
                lr_ax.plot(lr, color="green")
                lr_ax.set_ylabel("lr")

            fig.suptitle(title)
            fig.tight_layout()

        try:
            create_plot_from_dict(metrics, lr, title="Glow optimization")
            create_plot_from_dict(losses, lr, title="Glow optimization")
        except Exception as e:
            print(f"Could not plot optimization progress because of {e}")
        return


def plot_multiple_denoisers(
    denoisers: list,
    dpi: int = 600,
    show_metrics: bool = True,
    fig=None,
    axs=None,
    save=True,
    figsize=None,
) -> plt.Figure:
    """Plot multiple denoised results from different denoisers.

    Args:
        denoisers (list): list with denoiser objects
        dpi (int, optional): dpi of plotted image. Defaults to 600.
        show_metrics (bool, optional): show metrics on plot. Defaults to True.

    Returns:
        plt.Figure: matplotlib figure object
    """
    for denoiser in denoisers:
        assert isinstance(denoiser, Denoiser)

    denoiser = denoisers[0]

    denoised_samples = []
    for d in denoisers:
        ds = d.denoised_samples
        if d.corruptor.model:
            ds, noise_samples = ds
        if d.keep_track:
            ds = ds[-1]
        denoised_samples.append(ds)

    titles = ["ground truth", "noisy", *[d.model_names[d.name] for d in denoisers]]
    samples = [denoiser.target_samples, denoiser.noisy_samples, *denoised_samples]

    if denoiser.config.corruptor in ["cs", "sr", "cs_sine"]:
        samples.pop(1)
        titles.pop(1)

    # remove target_data (equals None if not paired data)
    if not denoiser.config.paired_data:
        samples = samples[1:]
        titles = titles[1:]

    width = 5 + len(samples) / 2

    samples = [
        (tf.clip_by_value(sample, denoiser.vmin, denoiser.vmax)).numpy()
        for sample in samples
    ]

    num_img = len(denoiser.noisy_samples)
    fig_contents = []
    if fig is None:
        if figsize is None:
            figsize = (width, int(num_img * 1.5))
        fig, axs = plt.subplots(
            num_img,
            len(samples),
            figsize=figsize,
        )
    for n in range(num_img):
        for i, (sample, title) in enumerate(zip(samples, titles)):
            image = np.squeeze(sample[n])

            vmin, vmax = denoiser.vmin, denoiser.vmax

            if denoiser.config.color_mode != "rgb" and len(image.shape) == 3:
                image = image[..., 0]

            im = axs[n, i].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            fig_contents.append(im)
            axs[n, i].axis("off")
            if n == 0:
                axs[n, i].set_title(title)

    if bool(denoiser.metrics) & show_metrics:
        offset = image.shape[0] * 0.1
        fontsize = 7
        color = "yellow"

        eval_denoised = []
        for denoiser in denoisers:
            eval_denoised.append(denoiser.eval_denoised)

        if denoiser.eval_noisy is not None:
            eval_denoised = [denoiser.eval_noisy, *eval_denoised]

        for n in range(num_img):
            for row, evals in zip(range(1, len(samples)), eval_denoised):
                for m, metric in enumerate(denoiser.metrics.metrics):
                    evaluation = evals[metric][n]
                    axs[n, row].text(
                        0,
                        offset + offset * m,
                        f"{metric}: {evaluation:.3f}",
                        color=color,
                        fontsize=fontsize,
                    )

    title = f"{denoiser.corruptor.task} "
    if denoiser.config.paired_data:
        if hasattr(denoiser.corruptor, "noise_stddev"):
            title += f"with $\sigma$={denoiser.corruptor.noise_stddev:.2f}"
    if (
        denoiser.config.get("subsample_factor") is not None
        and denoiser.config.get("subsample_factor") > 1
    ):
        title += f", $\downarrow_s$={denoiser.config.subsample_factor}"
    fig.suptitle(title)

    fig.tight_layout()

    folder = "figures"
    names = "-".join([d.name for d in denoisers])
    filename = f"{names}_{denoiser.config.dataset_name}_{denoiser.corruptor.task}"
    path = get_date_filename(f"{folder}/{filename}.png")
    if save:
        denoiser.savefig(fig, dpi, path)

    return fig, fig_contents


def animate_multiple_denoisers(
    denoisers,
    downsample: int = None,
    ending: int = 10,
    duration=None,
    fps=30,
) -> plt.Figure:
    """Animate the iterations of the generative process for multiple denoiers.

    Args:
        denoiser (Denoiser): denoiser object.
        downsample (int, optional): list with denoised images.
        duration (optional): duration of the animation in seconds. If provided,
            fps argument will be ignored.
        fps (int, optional): frames per second of animation. Defaults to 30 fps.

    Returns:
        plt.Figure: matplotlib figure object
    """
    n_frames = []
    for denoiser in denoisers:
        assert isinstance(denoiser, Denoiser)
        if denoiser.corruptor.model:
            denoised_samples, noise_samples = denoiser.denoised_samples
            denoised_samples = [
                (dn, ns) for dn, ns in zip(denoised_samples, noise_samples)
            ]
            denoiser.denoised_samples = denoised_samples

        if not denoiser.keep_track:
            continue
        n = len(denoiser.denoised_samples)
        n_frames.append(n)

    min_n_frames = np.min(n_frames)
    n_frames = []
    denoised_samples = []
    for denoiser in denoisers:
        if not denoiser.keep_track:
            denoiser.denoised_samples = [denoiser.denoised_samples] * min_n_frames

        n = len(denoiser.denoised_samples)
        n_frames.append(n)
        denoised_samples.append(denoiser.denoised_samples)

    start = []
    for n in n_frames:
        start.append(n - np.min(n_frames))

    num_img = len(denoiser.target_samples)

    # set parameters temporarily for plotting
    denoised_samples_list = []
    for ds, s in zip(denoised_samples, start):
        ds = ds[s:]
        n_frames = len(ds)

        if downsample is None:
            if n_frames > 100:
                downsample = 10
            else:
                downsample = 1

        ds = list(ds[::downsample]) + [ds[-1]] * int(ending / 100 * n_frames)
        denoised_samples_list.append(ds)
        n_frames = len(ds)

    n_cols = 2 + len(denoisers)
    fig, axs = plt.subplots(
        num_img,
        n_cols,
        figsize=(5 + n_cols / 2, int(num_img * 1.5)),
    )

    fig_contents = []
    for frame in range(n_frames):
        for denoiser, ds in zip(denoisers, denoised_samples_list):
            denoiser.keep_track = False
            denoiser.denoised_samples = ds[frame]

        fig, contents = plot_multiple_denoisers(
            denoisers=denoisers,
            fig=fig,
            axs=axs,
            save=False,
            show_metrics=False,
        )
        fig_contents.append(contents)

    # restore parameters
    for denoiser in denoisers:
        denoiser.keep_track = True

    folder = "figures"
    names = "-".join([d.name for d in denoisers])
    filename = f"{names}_{denoiser.config.dataset_name}_{denoiser.corruptor.task}"
    path = get_date_filename(f"{folder}/{filename}_animation.gif")

    if duration:
        fps = len(fig_contents) / duration
    save_animation(fig, fig_contents, filename=path, fps=fps)
    plt.close()

    return fig
