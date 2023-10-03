"""Metric functions
Author(s): Tristan Stevens
"""
import numpy as np
import tensorflow as tf

from utils.utils import translate


def mean_squared_error(y_true, y_pred, **kwargs):
    """Gives the MSE for two input tensors.
    Args:
        y_true: tensor
        y_pred: tensor
    Returns:
        mse: mean squared error between y_true and y_pred. L2 loss.

    """
    return reduce_mean(tf.math.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred, **kwargs):
    """Gives the MAE for two input tensors.
    Args:
        y_true: tensor
        y_pred: tensor
    Returns:
        mse: mean absolute error between y_true and y_pred. L1 loss.

    """
    return reduce_mean(tf.abs(y_true - y_pred))


def peak_signal_to_noise_ratio(y_true, y_pred, max_val=255, **kwargs):
    """Gives the Peak Signal to Noise Ratio (PSNR) for two input tensors.
    Args:
        y_true: tensor [None, height, width]
        y_pred: tensor [None, height, width]
        max_val: The dynamic range of the images

    Returns:
        psnr: peak signal to noise ratio of y_true and y_pred.

        psnr = 20 * log10(max_val / sqrt(MSE(y_true, y_pred)))
    """
    return tf.image.psnr(y_true, y_pred, max_val=max_val)


def structural_similarity_index(
    y_true,
    y_pred,
    max_val=255,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    **kwargs,
):
    """Gives the structural similiary index (SSIM) for two input tensors.
    Args:
        y_true: tensor [None, height, width]
        y_pred: tensor [None, height, width]
        max_val: The dynamic range of the images
        filter_size: size of gaussian filter
        filter_sigma: width of gaussian filter
        k1, k2: ssim constants
    Returns:
        ssim: structural similiary index between y_true and y_pred.

    """
    return tf.image.ssim(
        y_true,
        y_pred,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
    )


METRIC_FUNCS = dict(
    mse=mean_squared_error,
    mae=mean_absolute_error,
    psnr=peak_signal_to_noise_ratio,
    ssim=structural_similarity_index,
)

MINIMIZE = dict(
    mse=True,
    mae=True,
    psnr=False,
    ssim=False,
)


def reduce_mean(array, keep_batch_dim=True):
    """Reduce array by taking the mean.
    Preserves batch dimension if keep_batch_dim=True.
    """
    if keep_batch_dim:
        axis = len(array.shape)
        axis = range(axis)[1:]
    else:
        axis = None
    return tf.reduce_mean(array, axis=axis)


class Metrics:
    """Class for evaluating metrics."""

    def __init__(self, metrics: list, image_range: list):
        self.metrics = metrics
        self.image_range = image_range

        if isinstance(self.metrics, str):
            if self.metrics == "all":
                self.metrics = list(METRIC_FUNCS.keys())
            else:
                self.metrics = [self.metrics]

        for metric in self.metrics:
            assert metric in METRIC_FUNCS, (
                f"cannot find metric: {metric}, should be in \n"
                f"{list(METRIC_FUNCS.keys())}"
            )

        # link each metric to a bool which specifiec whether it is a
        # metric that should be minimized or not
        self.minimize = {metric: MINIMIZE[metric] for metric in self.metrics}

        # multiply metric with this value such that it can become a loss
        # so minimize objectives stay the same and maximize objective are multiplied with -1
        self.loss_multiplier = {
            metric: np.sign(int(self.minimize[metric]) - 0.5) for metric in self.metrics
        }

    def eval_metrics(
        self,
        y_true,
        y_pred,
        single=False,
        dtype=tf.float32,
        add_channel_axis=False,
        average_batch=False,
        to_numpy=True,
    ):
        """Evaluate metric on y_true and y_pred.

        Args:
            y_true (ndarray): first input array.
            y_pred (ndarray): second input array.
            single (bool, optional): _description_. Defaults to False.
            dtype (str, optional): cast to dtype. Defaults to tf.float32.
            add_channel_axis (bool, optional): add channel axis to data.
                Defaults to False.
            average_batch (bool, optional): return metric averaged over
                batch dimension. Defaults to False.
            to_numpy (bool, optional): return numpy array instead of
                a tensor. Defaults to True.

        Returns:
            dict: dict with metrics. keys are metric names.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if single:
            y_true = tf.expand_dims(y_true, axis=0)
            y_pred = tf.expand_dims(y_true, axis=0)

        if add_channel_axis or len(y_true.shape) == 3:
            y_true = tf.expand_dims(y_true, axis=-1)
            y_pred = tf.expand_dims(y_pred, axis=-1)

        y_true = tf.cast(y_true, dtype)
        y_pred = tf.cast(y_pred, dtype)

        m_dict = {}
        for metric in self.metrics:
            evaluations = METRIC_FUNCS[metric](
                y_true,
                y_pred,
                max_val=self.image_range[1],
            )

            if average_batch:
                evaluations = tf.reduce_mean(evaluations, axis=0)

            if to_numpy:
                evaluations = evaluations.numpy()

            m_dict[metric] = evaluations

        return m_dict

    @staticmethod
    def print_results(results, to_screen=True, precision=3):
        strings = []
        for metric, value in results.items():
            string = f"{metric} : {np.mean(value):.{precision}f}"
            strings.append(string)
            if to_screen:
                print(string)

        return ", ".join(str(metric) for metric in strings)

    @staticmethod
    def parse_metrics(metrics, reduce_mean=True):
        metrics = {k: [dic[k] for dic in metrics] for k in metrics[0]}
        if reduce_mean:
            metrics = {k: [np.mean(_v) for _v in v] for k, v in metrics.items()}
        return metrics
