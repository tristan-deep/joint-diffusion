import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import Model, Sequential
from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Cropping2D, Dense, Dropout, Flatten,
                          Input, Lambda, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Resizing, UpSampling2D)


def autocrop(encoder_layer, decoder_layer):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[1:-1] != decoder_layer.shape[1:-1]:
        ds = encoder_layer.shape[1:-1]
        es = decoder_layer.shape[1:-1]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if len(encoder_layer.shape) == 4:  # 2D
            encoder_layer = encoder_layer[
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
                :,
            ]
        elif len(encoder_layer.shape) == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
                ((ds[2] - es[2]) // 2) : ((ds[2] + es[2]) // 2),
                :,
            ]
    return encoder_layer, decoder_layer


def DenseBlock(units, normalization=None, activation=None, drop_prob=None):
    """Dense block.

    Args:
        units (int): Dimensionality of the output space.
        drop_prob (float): Dropout probability.

    Returns:
        output (sequential model):
    """
    output = Sequential()
    output.add(Dense(units))
    if normalization:
        output.add(get_normalization(normalization))
    if activation:
        output.add(get_activation(activation))
    if drop_prob:
        output = output.add(Dropout(drop_prob))

    return output


def SpectralNorm(layer: tf.keras.layers, n_iters: int = 1):
    return tfa.layers.SpectralNormalization(layer=layer, power_iterations=n_iters)


def ConvBlock(
    filters,
    normalization=None,
    activation=None,
    drop_prob=0.0,
    kernel_size=3,
    strides=1,
    dilation=None,
    spec_norm=None,
    bias=False,
):
    """
    Args:
        filters (int): Number of channels in the output.
        drop_prob (float): Dropout probability.
    """

    if dilation is None:
        dilation = 1

    output = Sequential()
    conv2d = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation,
        padding="same",
        use_bias=bias,
    )

    if spec_norm:
        output.add(SpectralNorm(conv2d))
    else:
        output.add(conv2d)
    if normalization:
        output.add(get_normalization(normalization))
    if activation:
        output.add(get_activation(activation))
    if drop_prob:
        output = output.add(Dropout(drop_prob))

    return output


def DownSample(
    filters,
    kernel_size,
    normalization=None,
    drop_prob=None,
    padding="same",
    activation="leakyrelu",
):
    """Down Sample block.

    Args:
        filters (int): number of filters.
        kernel_size (int / Tuple): size of kernel of conv2d filters.
        normalization (str, optional): whether to apply normalization. Defaults to None.
        drop_prob (float, optional): DESCRIPTION. Defaults to None.
        padding (str, optional): Type of padding of conv2d. Defaults to 'same'.

    Returns:
        output (sequential model): sequential downsample model.

    """
    output = Sequential()
    output.add(Conv2D(filters, kernel_size, strides=2, padding=padding, use_bias=False))

    if normalization:
        output.add(get_normalization(normalization))

    if drop_prob:
        output.add(Dropout(drop_prob))

    output.add(get_activation(activation))

    return output


def UpSample(
    filters,
    kernel_size,
    upmode,
    normalization=None,
    drop_prob=None,
    padding="same",
    activation="leakyrelu",
):
    """Up Sample block.

    Args:
        filters (int): number of filters.
        kernel_size (int / Tuple): size of kernel of conv2d filters.
        upmode (TYPE): type of upsampling: 'upconv', or 'upsample'.
        normalization (str, optional): whether to apply normalization. Defaults to None.
        drop_prob (float, optional): DESCRIPTION. Defaults to None.
        padding (str, optional): Type of padding of conv2d. Defaults to 'same'.

    Returns:
        output (sequential model): sequential upsample model.

    """
    output = Sequential()

    if upmode == "upconv":
        output.add(
            Conv2DTranspose(
                filters,
                kernel_size,
                strides=2,
                padding=padding,
                use_bias=False,
            )
        )

    elif upmode == "upsample":
        output.add(UpSampling2D(size=2, interpolation="bilinear"))

        output.add(Conv2D(filters, kernel_size=1, use_bias=False))

    if normalization:
        output.add(get_normalization(normalization))

    if drop_prob:
        output.add(Dropout(drop_prob))

    output.add(get_activation(activation))

    return output


def ResidualBlock(
    inputs,
    filters,
    kernel_size=3,
    dilation=None,
    normalization=None,
    activation=None,
    spec_norm=False,
    downsample=None,
):
    x = ConvBlock(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        normalization=normalization,
        strides=1,
        dilation=dilation,
        spec_norm=spec_norm,
    )(inputs)

    x = ConvBlock(
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        normalization=normalization,
        strides=1,
        dilation=dilation,
        spec_norm=spec_norm,
    )(x)

    if inputs.shape[-1] != x.shape[-1]:
        "1x1 convolution"
        inputs = ConvBlock(
            filters=filters,
            kernel_size=1,
            activation=None,
            normalization=None,
            strides=1,
            dilation=dilation,
            spec_norm=spec_norm,
        )(inputs)

    else:
        if dilation:
            inputs = ConvBlock(
                filters=filters,
                kernel_size=3,
                activation=None,
                normalization=None,
                strides=1,
                dilation=dilation,
                spec_norm=spec_norm,
            )(inputs)

    out = Add()([x, inputs])

    return out


def RCUBlock(inputs, filters, n_blocks, n_stages, activation=None, spec_norm=False):
    """Residual Conv Block

    Args:
        inputs (TYPE): DESCRIPTION.
        filters (TYPE): DESCRIPTION.
        n_blocks (TYPE): DESCRIPTION.
        n_stages (TYPE): DESCRIPTION.
        activation (TYPE, optional): DESCRIPTION. Defaults to None.
        spec_norm (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        x (TYPE): DESCRIPTION.

    """
    strides = 1
    blocks = {}

    for i in range(n_blocks):
        for j in range(n_stages):
            blocks[f"{i + 1}_{j + 1}_conv"] = ConvBlock(
                filters=filters,
                kernel_size=3,
                activation=activation,
                normalization=None,
                strides=strides,
                spec_norm=spec_norm,
            )

    x = inputs
    for i in range(n_blocks):
        residual = x
        for j in range(n_stages):
            x = blocks[f"{i + 1}_{j + 1}_conv"](x)

        x = Add()([x, residual])

    return x


def MSFBlock(inputs, shape, filters, spec_norm=False):
    """Multi Scale Fusion
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.


    Args:
        inputs (TYPE): DESCRIPTION.
        shape (TYPE): DESCRIPTION.
        filters (TYPE): DESCRIPTION.
        spec_norm (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        sums (TYPE): DESCRIPTION.

    """
    assert isinstance(inputs, list) or isinstance(inputs, tuple)

    layers = []
    for _ in range(len(inputs)):
        layers.append(
            ConvBlock(
                filters=filters,
                kernel_size=3,
                activation=None,
                normalization=None,
                strides=1,
                spec_norm=spec_norm,
                bias=True,
            )
        )

    for i in range(len(inputs)):
        h = layers[i](inputs[i])
        h = Resizing(shape[0], shape[1], interpolation="bilinear")(h)
        if i == 0:
            sums = tf.zeros_like(h)
        sums = Add()([h, sums])
    return sums


def CRPBlock(inputs, filters, n_stages, activation, spec_norm=False):
    """Chained Residual Pooling Block

    Chained residual pooling aims to capture background
    context from a large image region. This component is
    built as a chain of 2 pooling blocks, each consisting
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are
    fused together with the input feature map through summation
    of residual connections.

    Args:
        inputs (TYPE): DESCRIPTION.
        filters (TYPE): DESCRIPTION.
        n_stages (TYPE): DESCRIPTION.
        activation (TYPE): DESCRIPTION.
        spec_norm (TYPE, optional): DESCRIPTION. Defaults to False.

    Returns:
        x (TYPE): DESCRIPTION.

    """
    x = inputs
    h = inputs
    for i in range(n_stages):
        h = ConvBlock(
            filters=filters,
            kernel_size=3,
            activation=activation,
            normalization=None,
            strides=1,
            spec_norm=spec_norm,
        )(h)
        h = MaxPooling2D(pool_size=5, strides=1, padding="same")(h)
        x = Add()([h, x])
    return x


def RefineBlock(
    inputs, filters, start=False, end=False, activation=None, spec_norm=False
):
    """A RefineNet Block

    Combines together the ResidualConvUnits, fuses the feature maps using
    MultiResolutionFusion, and then gets large-scale context with the
    ResidualConvUnit.
    """
    assert isinstance(inputs, list) or isinstance(inputs, tuple)
    n_blocks = len(inputs)

    output_shape = inputs[0].get_shape().as_list()[1:3]

    hs = []
    for input_i in inputs:
        h = RCUBlock(
            input_i,
            filters=input_i.shape[-1],
            n_blocks=2,
            n_stages=2,
            activation=activation,
            spec_norm=spec_norm,
        )
        hs.append(h)

    if n_blocks > 1:
        h = MSFBlock(hs, shape=output_shape, filters=filters, spec_norm=spec_norm)
    else:
        h = hs[0]

    h = CRPBlock(
        h, filters=filters, n_stages=2, activation=activation, spec_norm=spec_norm
    )
    h = RCUBlock(
        h,
        filters=filters,
        n_blocks=3 if end else 1,
        n_stages=2,
        activation=activation,
        spec_norm=spec_norm,
    )

    return h


def get_normalization(normalization: str):
    if normalization == "batch":
        return tf.keras.layers.BatchNormalization()
    elif normalization == "instance":
        return tfa.layers.InstanceNormalization()
    elif normalization == "layer":
        return tf.keras.layers.LayerNormalization()
    elif "group" in normalization:
        num_groups = int(
            normalization.partition("group")[-1]
        )  # get the group size from string
        return tfa.layers.GroupNormalization(groups=num_groups)
    else:
        raise ValueError("Unknown normalization layer.")


def get_activation(activation: str = None):
    if activation.lower() == "relu":
        return ReLU()
    elif activation.lower() == "leakyrelu":
        return LeakyReLU()
    elif activation.lower() == "swish":
        return Lambda(lambda x: tf.keras.activations.swish(x))
    elif activation.lower() == "sigmoid":
        return Lambda(lambda x: tf.keras.activations.sigmoid(x))
    elif activation is None:
        return Lambda(lambda x: x)
    else:
        raise ValueError("Unknown activation function.")


def UNet(config, name=None):
    img_size_y, img_size_x, in_channels = config.image_shape

    inputs_x = Input(shape=[img_size_y, img_size_x, in_channels], name="input_image")

    last = Conv2DTranspose(
        in_channels,
        config.kernel_size,
        strides=1,
        padding="same",
    )

    x = inputs_x

    # Downsampling through the model
    skips = []
    for ch in config.channels:
        x = DownSample(
            ch,
            config.kernel_size,
            config.normalization,
            config.drop_prob,
        )(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for ch, skip in zip(reversed(config.channels), skips):
        x = UpSample(
            ch,
            config.kernel_size,
            config.upmode,
            config.normalization,
            config.drop_prob,
        )(x)
        # concatenate x with skip connection
        x = Concatenate()(autocrop(x, skip))

    x = UpSample(
        config.channels[0],
        config.kernel_size,
        config.upmode,
        config.normalization,
        config.drop_prob,
    )(x)

    x = last(x)

    return Model(inputs=inputs_x, outputs=x, name=name)


def get_discriminator_model(config):
    ndf = config.d_n_channels

    model = tf.keras.Sequential(name="discriminator")
    model.add(Input(shape=config.image_shape))
    model.add(Conv2D(ndf, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(LeakyReLU())

    model.add(Conv2D(ndf * 2, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(ndf * 4, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(ndf * 8, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(1, (4, 4), strides=(1, 1), use_bias=False))
    model.add(Reshape((1,)))

    return model


def get_generator_model(config):
    nz = config.latent_dim
    ngf = config.g_n_channels

    img_size_y, img_size_x, nc = config.image_shape

    model = tf.keras.Sequential(name="generator")
    model.add(
        Input(
            shape=[
                nz,
            ]
        )
    )
    model.add(Reshape((1, 1, nz)))
    model.add(
        Conv2DTranspose(
            ngf * 8, (4, 4), strides=(2, 2), padding="valid", use_bias=False
        )
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(ngf * 4, (4, 4), strides=(2, 2), padding="same", use_bias=False)
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(ngf * 3, (4, 4), strides=(2, 2), padding="same", use_bias=False)
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(ngf, (4, 4), strides=(2, 2), padding="same", use_bias=False)
    )
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(
        Conv2DTranspose(
            nc,
            (4, 4),
            strides=(2, 2),
            padding="same",
            activation="tanh",
            use_bias=False,
        )
    )
    return model


def get_flexible_discriminator_model(config):
    inputs = Input(shape=config.image_shape)

    x = inputs
    for ch in config.d_channels:
        layer = DownSample(
            filters=ch,
            kernel_size=config.d_kernel_size,
            normalization=config.d_normalization,
            drop_prob=config.d_drop_prob,
        )
        x = layer(x)

    # x = Flatten()(x)
    # outputs = Dense(1, activation=config.d_output_activation)(x)
    outputs = Conv2D(1, 1, config.d_kernel_size, activation=config.d_output_activation)(
        x
    )
    outputs = Flatten()(outputs)

    return Model(inputs=inputs, outputs=outputs, name="discriminator")


def get_flexible_generator_model(config):
    img_size_y, img_size_x, out_channels = config.image_shape
    image_shape = np.array([img_size_y, img_size_x])

    n_layers = len(config.g_channels)

    n_possible_layers = np.ceil(np.log2(image_shape))
    assert any(n_layers < n_possible_layers), (
        f"Generator network to deep {n_layers} > {int(min(n_possible_layers))}"
        f" layers for given image size!"
    )

    size = image_shape / (2**n_layers)

    # if size is not integer, the image size is not divisible by #layers
    # in this case, round to nearest image size. at the end we will crop.
    if any(size.astype(int) != size):
        new_shape = (np.ceil(size) * 2**n_layers).astype(int)
        size = new_shape // 2**n_layers
    else:
        new_shape = image_shape
        size = size.astype(int)

    inputs = Input(
        shape=[
            config.latent_dim,
        ]
    )

    # x = DenseBlock(
    #     units=size[0] * size[1] * config.g_first_dense_size,
    #     drop_prob=config.g_drop_prob,
    # )(inputs)

    # x = Reshape((*size, config.g_first_dense_size))(x)
    x = Lambda(lambda x: tf.expand_dims(x, 1))(inputs)
    x = Lambda(lambda x: tf.expand_dims(x, 1))(x)

    UpLayer = UpSample(
        filters=config.g_channels[0],
        kernel_size=1,
        upmode=config.g_upmode,
        normalization=config.d_normalization,
        drop_prob=config.g_drop_prob,
    )
    x = UpLayer(x)

    for ch in config.g_channels:
        UpLayer = UpSample(
            filters=ch,
            kernel_size=config.g_kernel_size,
            upmode=config.g_upmode,
            normalization=config.d_normalization,
            drop_prob=config.g_drop_prob,
        )
        x = UpLayer(x)

    outputs = Conv2DTranspose(
        out_channels,
        kernel_size=config.g_kernel_size,
        strides=2,
        padding="same",
        activation=config.g_output_activation,
    )(x)

    # crop image in case we scaled up the input due to image size not being
    # divisible by #layers
    if any(new_shape != image_shape):
        diff = (new_shape - image_shape) // 2
        outputs = Cropping2D((int(diff[0]), int(diff[1])))(outputs)

    generator = Model(inputs=inputs, outputs=outputs, name="generator")

    output_shape = generator.output_shape[1:]
    assert output_shape == tuple(
        config.image_shape
    ), f"Output shape {output_shape} not equal to image shape {config.image_shape}"

    return generator


def get_decoder_model(latent_inputs):
    x = Dense(64)(latent_inputs)

    x = Reshape((2, 2, 16))(x)

    x = Conv2DTranspose(
        filters=16, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    x = UpSampling2D()(x)

    x = Conv2DTranspose(
        filters=16, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    x = UpSampling2D()(x)

    x = Conv2DTranspose(
        filters=16, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(
        filters=16, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    x = UpSampling2D()(x)

    x = Conv2DTranspose(
        filters=1, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)

    decoder = Model(latent_inputs, x, name="decoder")

    return decoder


def get_encoder_model(inputs):
    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(
        inputs
    )

    x = MaxPooling2D()(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(x)

    x = MaxPooling2D()(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(x)

    x = MaxPooling2D()(x)

    x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(x)

    x = MaxPooling2D()(x)

    x = Flatten()(x)

    z_mean = Dense(2, name="z_mean")(x)
    z_log_sigma = Dense(2, name="z_log_sigma")(x)

    z = Sampling()([z_mean, z_log_sigma])

    encoder = Model(inputs, [z, z_mean, z_log_sigma], name="encoder")

    return encoder
