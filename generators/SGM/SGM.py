"""SGM: Score-Based Generative Modeling
Inspired from: https://github.com/yang-song/score_sde
Author(s): Tristan Stevens
"""
import tensorflow as tf
import tqdm
from keras import Model
from keras.layers import Input

from generators.layers import (
    ConvBlock,
    RefineBlock,
    ResidualBlock,
    get_activation,
    get_normalization,
)
from generators.SGM import sde_lib
from generators.SGM.sampling import ScoreSampler


def NCSNv2(config, name=None):
    img_size_y, img_size_x, in_channels = config.image_shape
    act = config.activation
    norm = config.normalization
    ks = config.kernel_size
    nf = config.channels

    inputs_x = Input(shape=[img_size_y, img_size_x, in_channels], name="input_image")

    # ResNet backbone
    h = ConvBlock(filters=nf, strides=1, activation=act, normalization=None, bias=True)(
        inputs_x
    )
    h = ResidualBlock(
        h,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=False,
    )
    layer1 = ResidualBlock(
        h,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=False,
    )

    nf = nf * 2

    h = ResidualBlock(
        layer1,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=True,
    )
    layer2 = ResidualBlock(
        h,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=False,
    )
    h = ResidualBlock(
        layer2,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=True,
        dilation=2,
    )
    layer3 = ResidualBlock(
        h,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=False,
        dilation=2,
    )
    h = ResidualBlock(
        layer3,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=True,
        dilation=4,
    )
    layer4 = ResidualBlock(
        h,
        filters=nf,
        kernel_size=ks,
        activation=act,
        normalization=norm,
        downsample=False,
        dilation=4,
    )

    # U-Net with RefineBlocks
    ref1 = RefineBlock([layer4], nf, activation=act, start=True)
    ref2 = RefineBlock([layer3, ref1], nf, activation=act)
    ref3 = RefineBlock([layer2, ref2], nf, activation=act)

    nf = nf // 2

    ref4 = RefineBlock([layer1, ref3], nf, activation=act, end=True)

    h = get_normalization(norm)(ref4)
    h = get_activation(act)(h)
    h = ConvBlock(
        filters=in_channels, strides=1, activation=None, normalization=None, bias=True
    )(h)

    return Model(inputs=inputs_x, outputs=h, name=name)


class ScoreNet(Model):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_shape = self.config.image_shape

        self.set_sde(config)

        self.sampling_shape = (self.config.get("num_img"), *self.config.image_shape)
        self.sampler = ScoreSampler(
            model=self,
            sde=self.sde,
            image_shape=self.image_shape,
            sampling_method=self.config.sampling_method,
            predictor=self.config.predictor,
            corrector=self.config.corrector,
            corrector_snr=self.config.snr,
        )

        score_backbone = self.config.get("score_backbone", "NCUNet")

        self.model = eval(score_backbone)(config=config, name="SGM")

    def compile(self, optimizer=None, loss_fn=None, **kwargs):
        super().compile(**kwargs)

        self.optimizer = optimizer

        if loss_fn:
            if self.config.get("reduce_mean"):
                self.reduce_op = tf.reduce_mean
            else:
                self.reduce_op = lambda *args, **kwargs: 0.5 * tf.reduce_sum(
                    *args, **kwargs
                )

            self.likelihood_weighting = self.config.get("likelihood_weighting")

            self.score_loss_tracker = tf.keras.metrics.Mean(name="score_loss")
            if loss_fn == "score_loss":
                self.monitor_loss = loss_fn
                self.loss_fn = self.score_loss
            else:
                raise NotImplementedError

    def set_sde(self, config):
        """Setup SDEs"""
        if config.sde.lower() == "vpsde":
            self.sde = sde_lib.VPSDE(
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                N=config.num_scales,
            )
        elif config.sde.lower() == "subvpsde":
            self.sde = sde_lib.subVPSDE(
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                N=config.num_scales,
            )
        elif config.sde.lower() == "vesde":
            self.sde = sde_lib.VESDE(
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                N=config.num_scales,
            )
        elif config.sde.lower() == "simple":
            self.sde = sde_lib.simple(
                sigma=config.sigma,
                N=config.num_scales,
            )
        else:
            raise NotImplementedError(f"SDE {config.sde} unknown.")

    @property
    def metrics(self):
        return [self.score_loss_tracker]

    def call(self, data, training=True):
        x, t = data
        score = self.get_score(x, t, training=training)
        return score

    def train_step(self, x):
        with tf.GradientTape() as tape:
            score_loss = self.loss_fn(x)
        grads = tape.gradient(score_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        # Update metrics
        self.score_loss_tracker.update_state(score_loss)
        return {m.name: m.result() for m in self.metrics}

    def score_loss(self, batch, training=True):
        """Compute the loss function.

        Args:
            batch: A mini-batch of training data.
            training: `True` for training loss and `False` for evaluation loss.

        Already set:
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
            likelihood_weighting: If `True`, weight the mixture of score matching losses
                according to https://arxiv.org/abs/2101.09258; otherwise use the weighting
                recommended in our paper.
            eps: A `float` number. The smallest time step to sample from.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        eps = 1e-5

        shape = tf.shape(batch)
        t = tf.random.uniform([shape[0]]) * (self.sde.T - eps) + eps
        z = tf.random.normal(shape=shape)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = self.get_score(perturbed_data, t, training=training)

        if not self.likelihood_weighting:
            losses = tf.math.square(score * std[:, None, None, None] + z)
            loss_shape = tf.shape(losses)[0]
            losses = self.reduce_op(tf.reshape(losses, shape=(loss_shape, -1)), axis=-1)

        else:
            g2 = self.sde.sde(tf.zeros_like(batch), t)[1] ** 2
            losses = tf.math.square(score + z / std[:, None, None, None])
            loss_shape = tf.shape(losses)[0]
            losses = (
                self.reduce_op(tf.reshape(losses, shape=(loss_shape, -1)), axis=-1) * g2
            )

        loss = tf.reduce_mean(losses)
        return loss

    def sample(self, z=None, **kwargs):
        samples = self.sampler(z=z, **kwargs)
        if self.config.image_range is not None:
            samples = (tf.clip_by_value(samples, *self.config.image_range)).numpy()
        return samples

    def get_score(self, x, t, training=True):
        """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

        Args:
            x: A TensorFlow tensor representing the current state
            t: A TensorFlow tensor representing the current time step.
            training: `True` for training score and `False` for evaluation score.
        Returns:
            A score function.
        """

        score = self.model(x, training=training)
        _, std = self.sde.marginal_prob(x, t)
        score = score / std[:, None, None, None]
        return score

    def get_latent_vector(self, batch_size):
        z = self.sde.prior_sampling([batch_size, *self.image_shape])
        return z

    def summary(self):
        self.model.summary()

    def get_eval_loss(self, dataloader, n_batches=None):
        losses = []

        if n_batches is None:
            n_batches = len(dataloader)
        else:
            n_batches = min(n_batches, len(dataloader))

        gen = iter(dataloader)
        for _ in tqdm.tqdm(range(n_batches), desc="Score eval loss"):
            batch = next(gen)
            loss = self.loss_fn(batch, training=False)
            losses.append(loss.numpy())

        return tf.reduce_mean(losses)
