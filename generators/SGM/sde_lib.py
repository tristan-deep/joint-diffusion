"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc

import numpy as np
import tensorflow as tf

from utils.utils import tf_expand_multiple_dims


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """Stochastic differential equation step."""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    def forward_diffuse(self, x, t):
        """Compute x_hat_t ~ p(x_t|x_0)
        Comput forward diffusion using marginal_prob mean / std
        """
        x_hat, std = self.marginal_prob(x, t)
        shape = tf.shape(x_hat)
        std = tf_expand_multiple_dims(std, len(shape) - 1)
        x_hat = x_hat + tf.random.normal(shape) * std
        return x_hat

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a TensorFlow tensor
            t: a TensorFlow float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * tf.math.sqrt(dt)
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_fn: A time-dependent score-based model that takes x and t
                and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used
                for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE:
            """Reverse SDE class."""

            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                """End time of the SDE."""
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = tf.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    """Variance Preserving SDE."""

    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)
        self.N = N
        self.discrete_betas = tf.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = tf.math.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = tf.math.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = (
            tf.math.exp(tf.reshape(log_mean_coeff, (-1, *[1] * (len(x.shape) - 1)))) * x
        )
        std = tf.math.sqrt(1.0 - tf.math.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return tf.random.normal(shape)

    def prior_logp(self, z):
        shape = tf.shape(z)
        N = tf.reduce_prod(shape[1:])
        logps = (
            -N / 2.0 * tf.math.log(2 * np.pi)
            - tf.reduce_sum(z**2, axis=(1, 2, 3)) / 2.0
        )
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = tf.cast(t * (self.N - 1) / self.T, dtype=tf.int64)
        beta = tf.gather(self.discrete_betas, timestep)
        alpha = tf.gather(self.alphas, timestep)
        sqrt_beta = tf.math.sqrt(beta)
        f = tf.math.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    """sub Variance Preserving SDE."""

    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = float(beta_min)
        self.beta_1 = float(beta_max)
        self.N = N
        self.discrete_betas = tf.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - tf.math.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = tf.math.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = tf.math.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - tf.math.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return tf.random.normal(shape)

    def prior_logp(self, z):
        shape = tf.shape(z)
        N = tf.reduce_prod(shape[1:])
        return (
            -N / 2.0 * tf.math.log(2 * np.pi)
            - tf.reduce_sum(z**2, axis=(1, 2, 3)) / 2.0
        )


class VESDE(SDE):
    """Variance Exploding SDE."""

    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.discrete_sigmas = tf.math.exp(
            tf.linspace(tf.math.log(self.sigma_min), tf.math.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = tf.zeros_like(x)
        diffusion = sigma * tf.math.sqrt(
            (2 * (tf.math.log(self.sigma_max) - tf.math.log(self.sigma_min)))
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return tf.random.normal(shape) * self.sigma_max

    def prior_logp(self, z):
        shape = tf.shape(z)
        N = tf.reduce_prod(shape[1:])
        return -N / 2.0 * tf.math.log(2 * np.pi * self.sigma_max**2) - tf.reduce_sum(
            z**2, axis=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = tf.cast(t * (self.N - 1) / self.T, dtype=tf.int64)
        sigma = tf.gather(self.discrete_sigmas, timestep)
        adjacent_sigma = tf.where(
            timestep == 0,
            tf.zeros_like(t),
            tf.gather(self.discrete_sigmas, timestep - 1),
        )
        f = tf.zeros_like(x)
        G = tf.math.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G


class simple(SDE):
    """simple SDE."""

    def __init__(self, sigma=25.0, N=1000):
        """Construct simple SDE.

        Args:
            sigma: sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma = float(sigma)
        self.N = N

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        drift = tf.zeros_like(x)
        diffusion = self.sigma**t
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = tf.math.sqrt(
            (self.sigma ** (2 * t) - 1.0) / 2.0 / tf.math.log(self.sigma)
        )
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return tf.random.normal(shape)

    def prior_logp(self, z):
        shape = tf.shape(z)
        N = tf.reduce_prod(shape[1:])
        logps = -N / 2.0 * tf.math.log(2 * np.pi * self.sigma**2) - tf.reduce_sum(
            z**2, axis=(1, 2, 3)
        ) / (2 * self.sigma**2)
        return logps
