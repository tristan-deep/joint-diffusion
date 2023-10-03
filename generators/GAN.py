"""Generative Adversarial Networks
Author(s): Tristan Stevens
    inspired from:
        - https://www.tensorflow.org/tutorials/generative/dcgan
        - https://keras.io/examples/generative/
"""
import tensorflow as tf
from keras import Model


class GAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        label_sigma=0.05,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.label_sigma = label_sigma

        self.real_label = 1
        self.fake_label = 0

    def compile(self, d_optimizer=None, g_optimizer=None, loss_fn=None, **kwargs):
        super().compile(**kwargs)

        assert (
            bool(d_optimizer) ^ bool(g_optimizer) == 0
        ), "Must specify both or neither of d_optimizer and g_optimizer"

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        if loss_fn:
            self.loss_fn = loss_fn

            self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
            self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

            self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
            self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

            self.monitor_loss = None

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    @tf.function
    def train_step(self, real_images):
        # Sample random points in the latent space
        self.batch_size = tf.shape(real_images)[0]

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        for _ in range(self.d_steps):
            ## Train discriminator with all-real batch
            labels = tf.ones((self.batch_size, 1)) * self.real_label

            with tf.GradientTape() as d_tape:
                predictions = self.discriminator(real_images, training=True)
                d_loss_real = self.discriminator_loss(labels, predictions)
            grads = d_tape.gradient(d_loss_real, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            ## Train with all-fake batch
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(self.batch_size, self.latent_dim)
            )

            # Decode them to generated images
            generated_images = self.generator(random_latent_vectors, training=True)
            labels = tf.ones((self.batch_size, 1)) * self.fake_label

            with tf.GradientTape() as d_tape:
                predictions = self.discriminator(generated_images, training=True)
                d_loss_fake = self.discriminator_loss(labels, predictions)
            grads = d_tape.gradient(d_loss_fake, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            d_loss = d_loss_real + d_loss_fake

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # fake labels are real for generator cost
        labels = tf.ones((self.batch_size, 1)) * self.real_label

        with tf.GradientTape() as g_tape:
            # generated images again within scope of g_tape
            # (probably more efficient way to do this)
            generated_images = self.generator(random_latent_vectors, training=True)
            predictions = self.discriminator(generated_images, training=True)
            g_loss = self.generator_loss(labels, predictions)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

    def discriminator_loss(self, labels, predictions):
        labels += self.label_sigma * tf.random.uniform(tf.shape(labels))
        d_loss = self.loss_fn(labels, predictions)
        return d_loss

    def generator_loss(self, labels, predictions):
        return self.loss_fn(labels, predictions)

    def sample(self, random_latent_vector):
        samples = self.generator(random_latent_vector, training=False)
        # samples = (samples + 1) / 2
        return samples

    def get_latent_vector(self, batch_size):
        return tf.random.normal(shape=(batch_size, self.latent_dim))

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
