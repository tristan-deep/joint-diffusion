"""Guidance class for joint posterior sampling
Author(s): Tristan Stevens
"""
import abc

import tensorflow as tf

_GUIDANCE = {}


def register_guidance(cls=None, *, name=None):
    """A decorator for registering guidance classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _GUIDANCE:
            raise ValueError(f"Already registered guidance with name: {local_name}")
        _GUIDANCE[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_guidance(name):
    """Get guidance class for given name."""
    return _GUIDANCE[name]


class Guidance(abc.ABC):
    """The abstract class for guidance / conditional sampling."""

    def __init__(self, sde, corruptor, lambda_coeff=None, kappa_coeff=None):
        self.sde = sde
        self.corruptor = corruptor
        self.lambda_coeff = lambda_coeff
        self.kappa_coeff = kappa_coeff
        self.A = self.corruptor.A
        if self.A is not None:
            self.A_T = tf.transpose(self.A)

    @tf.function
    def update_fn(self, y, x, t, x_mean, grad_x0_xt=None):
        """One update for guidance"""
        if self.corruptor.name in ["gaussian", "mnist", "cs", "cs_sine"]:
            x = self.denoise_update(y, x, t, x_mean, grad_x0_xt)
        else:
            raise ValueError(f"Unknown corruptor: {self.corruptor.name}")
        return x

    @tf.function
    def joint_update_fn(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """One update for guidance"""
        if self.corruptor.name in ["gaussian", "mnist", "cs", "cs_sine"]:
            x, n = self.joint_denoise_update(
                y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt
            )
        else:
            raise ValueError(f"Unknown corruptor: {self.corruptor.name}")
        return x, n


@register_guidance(name="pigdm")
class PIGDM(Guidance):
    """Pseudo inverse guidance
    https://openreview.net/forum?id=9_gsMA8MRKQ
    """

    def rt_squared(self, x, t):
        """rt^2 = sigma / (sigma + 1)"""
        sigma_t_squared = self.sde.marginal_prob(x, t)[1] ** 2
        r_t_squared = sigma_t_squared / (sigma_t_squared + 1)
        return r_t_squared

    def denoise_update(self, y, x, t, x_mean, grad_x0_xt):
        """Compute denoising data consistency step for Gaussian noise."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"

        r_t_squared = self.rt_squared(x, t)[:, None, None, None]

        # compressed sensing y = Ax + n
        if self.A is not None:
            m, d = tf.shape(self.A)
            I = tf.eye(m, m)

            # flatten x_0t and dx_0t_xt
            x_mean = tf.reshape(x_mean, (self.batch_size, -1))
            grad_x0_xt = tf.reshape(grad_x0_xt, (self.batch_size, -1))

            # posterior mean p(x0|xt) = A x_0
            mu_t = tf.linalg.matmul(x_mean, self.A_T)
            # posterior variance p(x0|xt) = sigma_xt^2 A A^T + sigma_y^2 I
            sigma_t = (
                r_t_squared * tf.linalg.matmul(self.A, self.A_T)
                + self.corruptor.noise_stddev**2 * I
            )
            sigma_t_inv = tf.linalg.inv(sigma_t)
            sigma_inv_A = tf.linalg.matmul(sigma_t_inv, self.A)

            # ∇_xt p(y | x_t) = (y - mu_t) inv(sigma_t) A * dx_0t_xt
            grad_p_y_xt = tf.linalg.matmul(y - mu_t, sigma_inv_A) * grad_x0_xt
            grad_p_y_xt = tf.reshape(grad_p_y_xt, (self.batch_size, *self.image_shape))

        # denoising y = x + n
        else:
            # x_t-1 = x_t + λ * (y - x_0) * dx_0_dx_t / sigma_t
            sigma_t = self.corruptor.noise_stddev**2 + r_t_squared
            grad_p_y_xt = grad_x0_xt * (y - x_mean) / sigma_t

        # data consistency step for x
        x = x + self.lambda_coeff * r_t_squared * grad_p_y_xt

        return x

    def joint_denoise_update(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """Compute denoising data consistency step for structured noise."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"
        assert grad_n0_nt is not None, "Gradients for p(n_0|t | n_t) were not provided"

        r_t_squared = self.rt_squared(x, t)[:, None, None, None]
        q_t_squared = r_t_squared

        if self.A is not None:
            # compressed sensing y = Ax + n
            m, d = tf.shape(self.A)
            I = tf.eye(m, m)

            # flatten x_0t and dx_0t_xt
            x_mean = tf.reshape(x_mean, (self.batch_size, -1))
            n_mean = tf.reshape(n_mean, (self.batch_size, -1))
            grad_x0_xt = tf.reshape(grad_x0_xt, (self.batch_size, -1))
            grad_n0_nt = tf.reshape(grad_n0_nt, (self.batch_size, -1))

            # posterior mean p(x0|xt) = A x_0
            mu_t = tf.linalg.matmul(x_mean, self.A_T) + n_mean
            # posterior variance p(x0|xt) = r_t^2^2 A A^T + q_t^2 I
            sigma_t = r_t_squared * tf.linalg.matmul(self.A, self.A_T) + q_t_squared * I
            sigma_t_inv = tf.transpose(tf.linalg.inv(sigma_t))

            # diff = tf.linalg.matmul(sigma_t_inv, tf.transpose(y - mu_t))
            sigma_inv_A = tf.linalg.matmul(sigma_t_inv, self.A)

            # ∇_xt p(y | x_t, n_t) = A^T inv(sigma_t) * (y - mu_t) * dx_0t_xt
            grad_p_y_xt = grad_x0_xt * tf.linalg.matmul(y - mu_t, sigma_inv_A)
            grad_p_y_xt = tf.reshape(grad_p_y_xt, (self.batch_size, *self.image_shape))

            # ∇_nt p(y | x_t, n_t) = inv(sigma_t) * (y - mu_t) * dn_0t_nt
            grad_p_y_nt = grad_n0_nt * tf.linalg.matmul(y - mu_t, sigma_t_inv)
            grad_p_y_nt = tf.reshape(grad_p_y_nt, (self.batch_size, *self.noise_shape))

        else:
            # y = beta * x + alpha * n
            alpha = self.corruptor.blend_factor
            beta = 1 - alpha

            sigma_t = r_t_squared + q_t_squared

            grad_p_y_xt = (
                -1
                * grad_x0_xt
                * (beta**2 * x_mean - beta * y + alpha * beta * n_mean)
                / sigma_t
            )
            grad_p_y_nt = (
                -1
                * grad_n0_nt
                * (alpha**2 * n_mean - alpha * y + alpha * beta * x_mean)
                / sigma_t
            )

        # data consistency step for x
        x = x + self.lambda_coeff * grad_p_y_xt * r_t_squared
        # data consistency step for n
        n = n + self.kappa_coeff * grad_p_y_nt * q_t_squared

        return x, n


@register_guidance(name="dps")
class DPS(Guidance):
    """Diffusion Posterior Sampling
    https://arxiv.org/pdf/2209.14687.pdf
    """

    def update_fn(self, y, x, t, x_mean, grad_x0_xt, *args):
        """Compute denoising data consistency step."""
        # x_t+1 = x_t - grad_x_t||y - x_0t||^2_2
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"

        with tf.GradientTape() as tape:
            tape.watch(x_mean)
            if self.A is not None:
                # compressed sensing y = Ax + n
                Ax = tf.linalg.matmul(x_mean, self.A_T)
                norm = tf.linalg.norm((y - Ax))
            else:
                norm = tf.linalg.norm((y - x_mean))

        # chain rule dy_dxt = dy_dx0 * dx0_dxt
        grad_p_y_xt = -1 * tape.gradient(norm, x_mean) * grad_x0_xt

        # data consistency step for x
        x = x + self.lambda_coeff * grad_p_y_xt
        return x

    def joint_denoise_update(self, y, x, n, t, x_mean, n_mean, grad_x0_xt, grad_n0_nt):
        """Compute denoising data consistency step for structured noise."""
        assert grad_x0_xt is not None, "Gradients for p(x_0|t | x_t) were not provided"
        assert grad_n0_nt is not None, "Gradients for p(n_0|t | n_t) were not provided"

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_mean)
            tape.watch(n_mean)
            if self.A is not None:
                # compressed sensing y = Ax + n
                _x_mean = tf.reshape(x_mean, (self.batch_size, -1))
                Ax = tf.linalg.matmul(_x_mean, self.A_T)
                norm = tf.linalg.norm((y - Ax - n_mean))
            else:
                # y = beta * x + alpha * n
                alpha = self.corruptor.blend_factor
                beta = 1 - alpha
                norm = tf.linalg.norm((y - beta * x_mean - alpha * n_mean))

        # chain rule dy_dxt = dy_dx0 * dx0_dxt
        grad_p_y_xt = -1 * tape.gradient(norm, x_mean) * grad_x0_xt
        # chain rule dy_dnt = dy_dn0 * dn0_dnt
        grad_p_y_nt = -1 * tape.gradient(norm, n_mean) * grad_n0_nt
        del tape

        # data consistency step for x
        x = x + self.lambda_coeff * grad_p_y_xt
        # data consistency step for n
        n = n + self.kappa_coeff * grad_p_y_nt

        return x, n


@register_guidance(name="projection")
class Projection(Guidance):
    """Projection sampling
    https://arxiv.org/pdf/2111.08005.pdf
    """

    def denoise_update(self, y, x, t, *args):
        """Compute data consistency for compressed sensing task."""
        y_hat = self.sde.forward_diffuse(y, t)

        if self.A is not None:
            # x_t+1 = x_t - λ * A^T(Ax_t - y_hat)
            # flatten x
            _x = tf.reshape(x, (self.batch_size, -1))

            # A^T (Ax - y)
            grad_y_hat_xt = tf.transpose(
                tf.linalg.matmul(
                    self.A_T,
                    tf.transpose(tf.linalg.matmul(_x, self.A_T)) - tf.transpose(y),
                )
            )
            # reshape again into image shape size
            grad_y_hat_xt = -tf.reshape(
                grad_y_hat_xt, (self.batch_size, *self.image_shape)
            )
        else:
            # x_t+1 = x_t - λ * (x_t - y_hat)
            grad_y_hat_xt = y_hat - x

        # data consistency step for x
        x = x + self.lambda_coeff * grad_y_hat_xt
        return x

    def joint_denoise_update(self, y, x, n, t, *args):
        """Compute data consistency for denoising for using score models."""

        y_hat = self.sde.forward_diffuse(y, t)

        if self.A is not None:
            # flatten x
            _x = tf.reshape(x, (self.batch_size, -1))

            # flatten n
            _n = tf.reshape(n, (self.batch_size, -1))

            # (Ax - y_hat + n)
            grad_y_hat_nt = tf.linalg.matmul(_x, self.A_T) - y_hat + _n
            # A^T (Ax - y_hat + n)
            grad_y_hat_xt = tf.transpose(
                tf.linalg.matmul(
                    self.A_T,
                    tf.transpose(grad_y_hat_nt),
                )
            )

            # reshape again into image shape size
            grad_y_hat_xt = -tf.reshape(
                grad_y_hat_xt, (self.batch_size, *self.image_shape)
            )
            grad_y_hat_nt = -tf.reshape(
                grad_y_hat_nt, (self.batch_size, *self.noise_shape)
            )
        else:
            # y = beta * x + alpha * n
            alpha = self.corruptor.blend_factor
            beta = 1 - alpha

            grad_y_hat_xt = -(beta**2 * x - beta * y_hat + alpha * beta * n)
            grad_y_hat_nt = -(alpha**2 * n - alpha * y_hat + alpha * beta * x)

        # data consistency step for x
        x = x + self.lambda_coeff * grad_y_hat_xt
        # data consistency step for n
        n = n + self.kappa_coeff * grad_y_hat_nt

        return x, n
