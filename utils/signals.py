"""Utility functions for signals
Author(s): Tristan Stevens
"""
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def add_gaussian_noise(x, sigma):
    """Add i.i.d. Gaussian noise with std `sigma`."""
    return tf.add(x, tf.random.normal(tf.shape(x), stddev=sigma))


def grayscale_to_random_rgb(images, image_shape=None, single_channel=True):
    """Converts grayscale image to a random color channel (rgb).

    Args:
        images (ndarray): Batch of grayscale images (with single channel dim).
        image_shape (ndarray): Output shape of batch of images. Necessary for
            TF dataset to retain shape information.
        single_channel (bool, optional): Whether output image is only R, G or B.
            if False, output image is random RGB color. Defaults to True.
    Returns:
        rgb_images (ndarray): Batch of rgb image (with 3 channel dims).

    """

    if len(tf.shape(images)) == 3:
        images = tf.expand_dims(images, axis=0)

    n_channels = 3
    batch_size = tf.shape(images)[0]

    if single_channel:
        rgb_images = tf.zeros_like(images)
        rgb_images = tf.repeat(rgb_images, n_channels, axis=-1)
        # move color channels to front since this is easier indexing
        rgb_images = tf.transpose(rgb_images, (0, 3, 1, 2))

        ch_indexes = tf.random.uniform(
            [batch_size], minval=0, maxval=n_channels, dtype=tf.int32
        )

        # rgb_images[tf.range(batch_size), ch_indexes, :, :] = np.squeeze(images)
        indices = tf.transpose(tf.stack([tf.range(batch_size), ch_indexes]))
        rgb_images = tf.tensor_scatter_nd_update(
            rgb_images, indices, tf.squeeze(images, axis=-1)
        )
        # move color channels to back as original
        rgb_images = tf.squeeze(tf.transpose(rgb_images, (0, 2, 3, 1)))
    else:
        rgb_images = tf.repeat(images, n_channels, axis=-1)
        random_colors = tf.nn.softmax(tf.random.normal((batch_size, 3)))
        random_colors = tf.expand_dims(random_colors, axis=1)
        random_colors = tf.expand_dims(random_colors, axis=2)
        rgb_images = tf.multiply(rgb_images, random_colors)

    if image_shape is not None:
        rgb_images.set_shape(image_shape)
    return rgb_images


class RandomTranslation(tf.keras.layers.Layer):
    """Random translation of images."""

    def __init__(self, width_factor, height_factor):
        """
        Randomly translates the input images by a random amount in the
        horizontal and vertical directions.
        The amount of translation is specified by the `width_factor`
        and `height_factor` parameters, which determine the maximum fraction
        of the image size by which the image can be translated.

        Args:
            width_factor (float): Maximum fraction of image width by
                which to translate horizontally.
            height_factor (float): Maximum fraction of image height
                by which to translate vertically.
        """
        super().__init__()
        self.width_factor = width_factor
        self.height_factor = height_factor

    def call(self, inputs, image_shape=None):
        """
        Applies random translations to the input images.

        Args:
            inputs (tf.Tensor): A tensor of shape
                (batch_size, height, width, channels) representing
                a batch of images to be translated.

        Returns:
            A tensor of the same shape as the input, but with each
            image translated by a random amount.
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Generate a random translation for each image in the batch
        translate_x = tf.random.uniform(
            shape=[batch_size], minval=-self.width_factor, maxval=self.width_factor
        )
        translate_x *= tf.cast(width, tf.float32)
        translate_y = tf.random.uniform(
            shape=[batch_size], minval=-self.height_factor, maxval=self.height_factor
        )
        translate_y *= tf.cast(height, tf.float32)

        # Construct the transformation matrices
        ones = tf.ones(shape=[batch_size], dtype=tf.float32)
        zeros = tf.zeros(shape=[batch_size], dtype=tf.float32)
        transform = tf.stack(
            [ones, zeros, translate_x, zeros, ones, translate_y, zeros, zeros], axis=1
        )

        # Apply the transformation to the input images
        output = tfa.image.transform(inputs, transform, interpolation="BILINEAR")
        if image_shape is not None:
            output.set_shape(image_shape)

        return output
