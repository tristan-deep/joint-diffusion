"""Load and save deep learning models
Author(s): Tristan Stevens
"""
import numpy as np
import tensorflow_addons as tfa
import torch
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

from generators.GAN import GAN
from generators.glow.glow import Glow
from generators.layers import (UNet, get_decoder_model,
                               get_discriminator_model, get_encoder_model,
                               get_generator_model)
from generators.SGM.SGM import NCSNv2, ScoreNet


def get_model(config, run_eagerly=False, plot_summary=False, training=True):
    """Return model based on config parameters.

    Args:
        config (dict): dict object with model init parameters.
            requires different keys for different models.
        run_eagerly (bool, optional): whether to compile model in eager mode or graph mode.
            Defaults to False (i.e. graph mode).
        plot_summary (bool, optional): plot summary of model. Defaults to False.
        training (bool, optional): trainig mode, used to compile model with optimizer
            and set loss function. Defaults to True. If False, inference mode without optimizer.

    Returns:
        model: ML model (TF / torch depending on which model)
    """
    if run_eagerly:
        print("Warning, run_eagerly is turned to on!!")

    model_name = config.model_name

    assert model_name.lower() in [
        "gan",
        "score",
        "glow",
        "ncsnv2",
        "unet",
    ], """Invalid model name found in config file. Should
        be either 'gan', 'score', 'glow', 'ncsnv2' or 'unet'."""

    print(f"\nLoading {model_name} model...")

    if model_name.lower() == "gan":
        discriminator = get_discriminator_model(config)
        generator = get_generator_model(config)

        model = GAN(
            discriminator=discriminator,
            generator=generator,
            discriminator_extra_steps=config.d_steps,
            latent_dim=config.latent_dim,
            label_sigma=config.label_sigma,
        )

        if plot_summary:
            model.summary()

        if training:
            beta_1, beta_2 = config.get("adam_betas", [0.9, 0.999])
            compile_args = {
                "run_eagerly": run_eagerly,
                "d_optimizer": Adam(
                    learning_rate=config.d_lr, beta_1=beta_1, beta_2=beta_2
                ),
                "g_optimizer": Adam(
                    learning_rate=config.g_lr, beta_1=beta_1, beta_2=beta_2
                ),
                "loss_fn": BinaryCrossentropy(from_logits=True),
            }

    if model_name.lower() == "score":
        model = ScoreNet(config)

        if training:
            optimizer = Adam(learning_rate=config.lr)
            if config.get("ema") is not None:
                print(f"Using EMA: {config.ema}")
                optimizer = tfa.optimizers.MovingAverage(
                    optimizer=optimizer,
                    average_decay=config.ema,
                )
            compile_args = {
                "run_eagerly": run_eagerly,
                "optimizer": optimizer,
                "loss_fn": "score_loss",
            }

    if model_name.lower() == "glow":
        model = Glow(
            np.array(config.image_shape)[[2, 0, 1]],
            K=config.K,
            L=config.L,
            coupling=config.coupling,
            n_bits_x=config.n_bits_x,
            nn_init_last_zeros=config.last_zeros,
            device=config.device,
        )

        if training:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            compile_args = {
                "run_eagerly": run_eagerly,
                "optimizer": optimizer,
                "config": config,
                "loss_fn": "nll",
            }

    if model_name.lower() == "ncsnv2":
        model = NCSNv2(config, name="ncsnv2")

        if training:
            compile_args = {
                "run_eagerly": run_eagerly,
                "loss": config.loss,
                "optimizer": Adam(learning_rate=config.lr),
            }

    if model_name.lower() == "unet":
        model = UNet(config, name="unet")

        if training:
            compile_args = {
                "run_eagerly": run_eagerly,
                "loss": config.loss,
                "optimizer": Adam(learning_rate=config.lr),
            }

    if not training:
        model.compile(run_eagerly=run_eagerly)
    else:
        model.compile(**compile_args)

    return model
