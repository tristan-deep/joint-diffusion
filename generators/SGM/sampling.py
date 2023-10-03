"""Sampling functionality for score-based diffusion models.
Author(s): Tristan Stevens
"""
import abc
import warnings

import tensorflow as tf
from scipy import integrate

from generators.SGM import sde_lib
from generators.SGM.guidance import get_guidance

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered predictor with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered corrector with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    """Get predictor class for given name."""
    return _PREDICTORS[name]


def get_corrector(name):
    """Get corrector class for given name."""
    return _CORRECTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False, compute_grad=True):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn
        self.compute_grad = compute_grad
        self.grad_x0_xt = None

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
            x: A TensorFlow tensor representing the current state
            t: A TensorFlow tensor representing the current time step.

        Returns:
            x: A TensorFlow tensor of the next state.
            x_mean: A TensorFlow tensor. The next state without random noise. Useful for denoising.
        """
        return


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
            x: A TensorFlow tensor representing the current state
            t: A TensorFlow tensor representing the current time step.

        Returns:
            x: A TensorFlow tensor of the next state.
            x_mean: A TensorFlow tensor. The next state without random noise. Useful for denoising.
        """
        return


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    """Euler Maruyama diffusion sampler."""

    def update_fn(self, x, t):
        dt = -1.0 / self.rsde.N
        z = tf.random.normal(x.shape)

        if self.compute_grad:
            with tf.GradientTape() as tape:
                tape.watch(x)
                drift, diffusion = self.rsde.sde(x, t)
                x_mean = x + drift * dt

            self.grad_x0_xt = tape.gradient(x_mean, x)
        else:
            drift, diffusion = self.rsde.sde(x, t)
            x_mean = x + drift * dt

        x = x_mean + diffusion[:, None, None, None] * tf.math.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion sampler."""

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = tf.random.normal(tf.shape(x))
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    """Langevin diffusion sampler."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
            and not isinstance(sde, sde_lib.simple)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE, sde_lib.simple)):
            timestep = tf.cast(t * (sde.N - 1) / sde.T, dtype=tf.int64)
            alpha = tf.gather(sde.alphas, timestep)
        else:
            alpha = tf.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = tf.random.normal(tf.shape(x))
            grad_norm = tf.reduce_mean(
                tf.norm(tf.reshape(grad, (grad.shape[0], -1)), axis=-1)
            )
            noise_norm = tf.reduce_mean(
                tf.norm(tf.reshape(noise, (noise.shape[0], -1)), axis=-1)
            )
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + tf.math.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = tf.cast(t * (sde.N - 1) / sde.T, dtype=tf.int64)
            alpha = tf.gather(sde.alphas, timestep)
        else:
            alpha = tf.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = tf.random.normal(tf.shape(x))
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * tf.math.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    # pylint: disable=super-init-not-called
    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    # pylint: disable=super-init-not-called
    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


class ScoreSampler:
    """Sampler class for score-based generative models."""

    def __init__(
        self,
        model,
        image_shape,
        sde: sde_lib.SDE,
        sampling_method: str,
        predictor: str = None,
        corrector: str = None,
        guidance: str = None,
        corruptor=None,
        keep_track: bool = False,
        n_corrector_steps: int = 1,
        corrector_snr: float = 0.15,
        lambda_coeff: float = 0.1,
        kappa_coeff: float = 0.1,
        noise_model=None,
        noise_shape=None,
        start_diffusion: float = None,
        sampling_eps: float = None,
        early_stop: int = None,
    ):
        """Sampler class for score-based models.

        Can sample both uncoditionally and conditionally using predictor-corrector (PC)
        and ODE methods. Designed for multiple invers tasks, such as denoising,
        and compressive sensing.

        Args:
            model (ScoreNet): Score-based model.
            image_shape (tuple): image shape (batch_size, height, width, channels).
            sde (sde_lib.SDE):  An `sde_lib.SDE` object that represents the forward SDE.
            sampling_method (str):  pc or ode sampling methods.
            predictor (str, optional): A subclass of `sampling.Predictor` that represents
                a predictor algorithm. Defaults to None.
            corrector (str, optional): A subclass of `sampling.Corrector` that represents
                a corrector algorithm. Defaults to None.
            guidance (str, optional): A subclass of `sampling.Guidance` that represents
                a guidance algorithm. Defaults to None.
            keep_track (bool, optional): keep track of intermdiate samples during
                optimization. Defaults to False.
            n_corrector_steps (int, optional): The number of corrector steps per update
                of the corrector. Defaults to 1.
            corrector_snr (float, optional): The signal-to-noise ratio for the corrector.
                Defaults to 0.15.
            lambda_coeff (float, optional): Data consistency likelihood weighting.
                Defaults to 0.1.
            kappa_coeff (float, optional): Data consistency likelihood weighting for
                noise model. Only used with joint denoiser. Defaults to 0.1.
            noise_model (ScoreNet, optional): score-based model that models the noise.
                Defaults to None.
            start_diffusion (float): number between 0 and 1 specifying where to start
                diffusion (if not at zero).
            sampling_eps (float, optional): The reverse-time SDE is only integrated to
                `sampling_eps` for numerical stability.
        """
        assert sampling_method in ["pc", "ode"], f"{sampling_method} is not supported"

        self.image_shape = image_shape
        self.model = model
        self.sde = sde
        self.sampling_method = sampling_method
        self.corruptor = corruptor
        self.batch_size = None

        self.score_fn = lambda x, t: self.model.get_score(x, t, training=False)
        if noise_model:
            self.noise_score_fn = lambda x, t: noise_model.get_score(
                x, t, training=False
            )
            self.noise_model = noise_model
        else:
            self.noise_model = None

        self.keep_track = keep_track
        self.n_corrector_steps = n_corrector_steps
        self.corrector_snr = corrector_snr
        self.lambda_coeff = lambda_coeff
        self.kappa_coeff = kappa_coeff
        self.noise_shape = self.image_shape if noise_shape is None else noise_shape
        self.start_diffusion = start_diffusion
        self.eps = sampling_eps if sampling_eps is not None else 1e-3
        self.early_stop = early_stop
        self.guidance = guidance
        self.compute_grad = False

        if self.sampling_method == "pc":
            if self.guidance:
                if guidance.lower() in ["pigdm", "dps"]:
                    self.compute_grad = True
                    assert (
                        predictor == "euler_maruyama"
                    ), "PIGDM only supported by Euler-Maruyama predictor."
                self.guidance = get_guidance(guidance)(
                    self.sde, self.corruptor, self.lambda_coeff, self.kappa_coeff
                )
                print("Using guidance model: ", guidance)

            self.predictor, self.corrector = self.get_predictor_corrector_fn(
                predictor,
                corrector,
                self.score_fn,
            )
            if self.noise_model:
                (
                    self.noise_predictor,
                    self.noise_corrector,
                ) = self.get_predictor_corrector_fn(
                    predictor,
                    corrector,
                    self.noise_score_fn,
                )

        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")

    def get_predictor_corrector_fn(self, predictor: str, corrector: str, score_fn):
        """Return functions for predictor and corrector."""
        if predictor is None:
            # Corrector-only sampler
            predictor = NonePredictor(self.sde, score_fn, probability_flow=False)
        else:
            predictor = get_predictor(predictor.lower())
            predictor = predictor(
                self.sde,
                score_fn,
                probability_flow=False,
                compute_grad=self.compute_grad,
            )

        if corrector is None:
            # Predictor-only sampler
            corrector = NoneCorrector(
                self.sde,
                score_fn,
                self.corrector_snr,
                self.n_corrector_steps,
            )
        else:
            corrector = get_corrector(corrector.lower())
            corrector = corrector(
                self.sde,
                score_fn,
                self.corrector_snr,
                self.n_corrector_steps,
            )
        return predictor, corrector

    def __call__(self, y=None, **kwargs):
        if y is None:
            x = self._sample(**kwargs)
        else:
            x = self._conditional_sample(y, **kwargs)
        return x

    def _sample(self, z=None, shape=None, progress_bar=True):
        if self.sampling_method == "pc":
            x = self.pc_sampler(z=z, shape=shape, progress_bar=progress_bar)
        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")
        return x

    def _conditional_sample(self, y, progress_bar=True):
        if self.sampling_method == "pc":
            x = self.pc_sampler(y=y, progress_bar=progress_bar)
        elif self.sampling_method == "ode":
            raise NotImplementedError("ODE not supported")
        return x

    # @tf.function(jit_compile=True)
    def pc_sampler(self, y=None, z=None, shape=None, progress_bar=True):
        """The PC sampler function.

        Args:
            y (array / tensor): If given, do conditional (posterior) sampling.
                Defaults to None, in that case prior sampling.
            z (array / tensor): If given, generate samples from latent code `z`.
                Only used in uncoditional setting.
            shape (tuple): Shape of images to generate, only used for prior sampling.
                Shape is inferred with posterior sampling and Defaults to None.
            progress_bar (bool): Whether to have progress bar during inference.

        Returns:
            List or Tensor or (List, List) or (Tensor, Tensor)

                samples from either prior or posterior if a measurement `y` is provided.

                if joint inference (i.e. with noise_model), returns a tuple with x and n.
                if keep_track, returns List with all intermediate steps.s

        """
        # Initialize with z or x, also check if y is provided
        if y is None:
            if z is None:
                # If not represent, sample the latent code from
                # the prior distibution of the SDE.
                x = self.sde.prior_sampling(shape)
            else:
                x = tf.cast(z, tf.float32)
            self.batch_size = tf.shape(x)[0]
        else:
            y = tf.cast(y, tf.float32)
            self.batch_size = tf.shape(y)[0]

            shape = (self.batch_size, *self.image_shape)
            self.guidance.batch_size = self.batch_size
            self.guidance.image_shape = self.image_shape

            # if structured noise intialize noise sample
            if self.noise_model is not None:
                noise_shape = (self.batch_size, *self.noise_shape)
                n = self.sde.prior_sampling(noise_shape)
                self.guidance.noise_shape = self.noise_shape

            # forward diffuse measurement is start diffusion is not zero
            if (self.start_diffusion is not None) and self.start_diffusion > 0:
                x = self.sde.forward_diffuse(y, (self.sde.T - self.start_diffusion))
            else:
                x = self.sde.prior_sampling(shape)

        # for keeping track of denoising steps (animation)
        if self.keep_track:
            x_list = [x]
            if self.noise_model:
                n_list = [n]

        # diffusion timeline
        if self.start_diffusion:
            timesteps = tf.linspace(
                (self.sde.T - self.start_diffusion), self.eps, self.sde.N
            )
        else:
            timesteps = tf.linspace(self.sde.T, self.eps, self.sde.N)

        if self.early_stop:
            timesteps = tf.gather(timesteps, tf.range(self.early_stop))

        # progress bar
        if progress_bar:
            pbar = tf.keras.utils.Progbar(self.sde.N)

        # main reverse diffusion loop
        for t in timesteps:
            vec_t = tf.ones(self.batch_size, dtype=tf.float32) * t

            x, x_mean = self.corrector.update_fn(x, vec_t)
            x, x_mean = self.predictor.update_fn(x, vec_t)

            # data consistency steps
            if y is not None:
                assert (
                    self.guidance is not None
                ), "Please select a guidance model for conditional sampling."

                # if structured noise and second noise model is available
                if self.noise_model is not None:
                    n, n_mean = self.noise_corrector.update_fn(n, vec_t)
                    n, n_mean = self.noise_predictor.update_fn(n, vec_t)

                    x, n = self.guidance.joint_update_fn(
                        y,
                        x,
                        n,
                        vec_t,
                        x_mean,
                        n_mean,
                        self.predictor.grad_x0_xt,
                        self.noise_predictor.grad_x0_xt,
                    )
                else:
                    x = self.guidance.update_fn(
                        y, x, vec_t, x_mean, self.predictor.grad_x0_xt
                    )

            # store intermediate results (animation)
            if self.keep_track:
                x_list.append(x_mean)
                if self.noise_model:
                    n_list.append(n_mean)

            # kill diffusion is NaN values are found (should not happen)
            if tf.math.is_nan(tf.reduce_sum(x)):
                warnings.warn("NaN in intermediate solution, breaking out...")
                break

            if progress_bar:
                pbar.add(1)

        if self.noise_model:
            return (
                x_list if self.keep_track else x_mean,
                n_list if self.keep_track else n_mean,
            )
        else:
            return x_list if self.keep_track else x_mean
