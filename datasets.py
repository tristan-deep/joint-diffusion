"""Load / generate datasets.
Author(s): Tristan Stevens
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from utils.signals import RandomTranslation, grayscale_to_random_rgb
from utils.utils import download_and_unpack, get_normalization_layer

AUTOTUNE = tf.data.AUTOTUNE

_DATASETS = [
    "mnist",
    "celeba",
    "sinenoise",
    "sinenoise1d",
    "tmnist",
]


def get_dataset(config):
    """
    Generate a dataset based on parameters specified in config object.

    Args:
        config (dict): config dict.

    Returns:
        tuple(Dataset, Dataset)
            Tuple of train-, and test-dataloader respectively
    """
    dataset_name = config["dataset_name"]

    assert (
        dataset_name.lower() in _DATASETS
    ), f"""Invalid dataset name {dataset_name.lower()} found in config file.
        Should be in {_DATASETS}."""

    print(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == "mnist":
        train, test = _get_mnist(config)
    if dataset_name.lower() == "celeba":
        train, test = _get_celeba(config)
    if dataset_name.lower() == "sinenoise":
        train, test = _get_sine_noise_dataset(config)
    if dataset_name.lower() == "sinenoise1d":
        train, test = _get_sine_noise1D_dataset(config)
    if dataset_name.lower() == "tmnist":
        train, test = _get_tmnist(config)

    datasets = train, test
    dataset = datasets[datasets != None]
    ## check image shape after all transforms
    try:
        # fast way, but somehow not always possible
        image_shape = dataset.element_spec.shape[1:].as_list()
    except:
        # slow way, literally reading a sample and checking the size
        image_shape = list(tf.shape(next(iter(dataset))))[1:]

    config.image_shape = image_shape

    if train:
        train = train.prefetch(buffer_size=AUTOTUNE)
    if test:
        test = test.prefetch(buffer_size=AUTOTUNE)

    return train, test


def _get_mnist(config):
    """Loads MNIST dataset.

    Loads and preprocesses MNIST dataset into a train and test dataloader.

    Args:
        config (dict): config dict.

    Returns:
        tuple(Dataset, Dataset)
            Tuple of train-, and test-dataloader respectively
    """
    default_image_size = 28

    image_size = config.get("image_size") or default_image_size
    image_range = config.get("image_range", [-1, 1])
    color_mode = config.get("color_mode", "grayscale")
    seed = config.get("seed", None)
    shuffle = config.get("shuffle", True)
    batch_size = config.get("batch_size")

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()

    # create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

    # shuffle
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_images), seed=seed)
        test_dataset = test_dataset.shuffle(len(test_dataset), seed=seed)

    # limit number of samples
    if config.get("limit_n_samples"):
        train_dataset = train_dataset.take(config.limit_n_samples)
        test_dataset = test_dataset.take(config.limit_n_samples)

    print(f"Using {len(train_dataset)} files for training.")
    print(f"Using {len(test_dataset)} files for validation.")

    # batch
    if batch_size:
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

    # transforms
    transforms = []
    transforms.append(lambda x: tf.expand_dims(x, axis=-1))
    transforms.append(lambda x: tf.image.resize(x, image_size))
    transforms.append(lambda x: get_normalization_layer(*image_range)(x))
    if color_mode == "rgb":
        transforms.append(lambda x: grayscale_to_random_rgb(x, (None, *image_size, 3)))
    if config.get("translation"):
        translate_random = RandomTranslation(config.translation, config.translation)
        transforms.append(lambda x: translate_random(x, (None, *image_size, 3)))

    for transform in transforms:
        train_dataset = train_dataset.map(transform, num_parallel_calls=AUTOTUNE)
        test_dataset = test_dataset.map(transform, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def _get_tmnist(config):
    """Loads TMNIST-alpha dataset.

    https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters

    Loads and preprocesses TMNIST-alpha dataset with 94 charachters
    into a train and test dataloader.

    Args:
        config (dict): config dict.

    Returns:
        tuple(Dataset, Dataset)
            Tuple of train-, and test-dataloader respectively
    """

    def _tmnist_load_data(data_root, validation_split=0.2):
        dataset_file = Path(data_root) / "images/TMNIST/tmnist_dataset.npy"
        if dataset_file.is_file():
            dataset = np.load(dataset_file, allow_pickle=True).item()
            X_train = dataset["X_train"]
            y_train = dataset["y_train"]
            X_test = dataset["X_test"]
            y_test = dataset["y_test"]
        else:
            csv_file = Path(data_root) / "images/TMNIST/94_character_TMNIST.csv"

            df = pd.read_csv(csv_file)
            X = df.iloc[:, 2:].astype("float32")
            y = df[["labels"]]

            labels = y["labels"].unique()
            values = [num for num in range(len(df["labels"].unique()))]
            label_dict = dict(zip(labels, values))

            y["labels"].replace(label_dict, inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split
            )

            Length, Height = 28, 28  # <---- Defining LxH
            NCl = y_train.nunique()[0]  # Unique targets -- > 94

            # ------>  N of images 28x28
            X_train = np.reshape(X_train.values, (X_train.shape[0], Length, Height))
            X_test = np.reshape(X_test.values, (X_test.shape[0], Length, Height))

            # -------> Target into Categorical Values
            y_train = to_categorical(y_train, NCl, dtype="int")
            y_test = to_categorical(y_test, NCl, dtype="int")

            save_dict = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }

            np.save(dataset_file, save_dict)

        return (X_train, y_train), (X_test, y_test)

    default_image_size = 28

    data_root = Path(config.data_root)
    image_size = config.get("image_size") or default_image_size
    image_range = config.get("image_range", [-1, 1])
    color_mode = config.get("color_mode", "grayscale")
    seed = config.get("seed", None)
    shuffle = config.get("shuffle", True)
    batch_size = config.get("batch_size")
    validation_split = config.get("validation_split", 0.2)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    (train_images, train_labels), (test_images, test_labels) = _tmnist_load_data(
        data_root, validation_split
    )

    # create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

    # shuffle
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_images), seed=seed)
        test_dataset = test_dataset.shuffle(len(test_dataset), seed=seed)

    # limit number of samples
    if config.get("limit_n_samples"):
        train_dataset = train_dataset.take(config.limit_n_samples)
        test_dataset = test_dataset.take(config.limit_n_samples)

    print(f"Using {len(train_dataset)} files for training.")
    print(f"Using {len(test_dataset)} files for validation.")

    # batch
    if batch_size:
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

    # transforms
    transforms = []
    transforms.append(lambda x: tf.expand_dims(x, axis=-1))
    transforms.append(lambda x: tf.image.resize(x, image_size))
    transforms.append(lambda x: get_normalization_layer(*image_range)(x))
    if color_mode == "rgb":
        transforms.append(lambda x: grayscale_to_random_rgb(x, (None, *image_size, 3)))

    for transform in transforms:
        train_dataset = train_dataset.map(transform, num_parallel_calls=AUTOTUNE)
        test_dataset = test_dataset.map(transform, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def _get_celeba(config):
    """Loads CelebA dataset.

    Downloaded using:
        wget https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
        tar -xvf celeb-tfr.tar
        update: gets automatically downloaded

    Required directory:
        data_root / 'images' / 'celeba-tfr'

    Loads and preprocesses Celeb dataset into a train and test dataloader.

    Args:
        dataset_config (dict): config dict.

    Returns:
        tuple(Dataset, Dataset)
            Tuple of train-, and test-dataloader respectively

    """
    data_root = Path(config.data_root)

    default_image_size = 256
    image_size = config.get("image_size") or default_image_size
    config.image_size = image_size

    image_range = config.get("image_range", [-1, 1])
    color_mode = config.get("color_mode", "rgb")
    seed = config.get("seed", None)
    shuffle = config.get("shuffle", True)
    batch_size = config.get("batch_size")

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    path = data_root / "images" / "celeba-tfr"
    if not path.is_dir():
        download_and_unpack(
            url="https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar",
            save_path=path.parent,
        )
        # rename `train` to `training` folder for consistency with other datasets
        (path / "train").rename(path / "training")

    datasets = []

    features = {
        "shape": tf.io.FixedLenFeature([3], tf.int64),
        "data": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([1], tf.int64),
        "attr": tf.io.FixedLenFeature([40], tf.int64),
    }

    def _parse_tf_record(record):
        r = tf.io.parse_single_example(record, features)
        data, label, shape, attr = r["data"], r["label"], r["shape"], r["attr"]
        img = tf.io.decode_raw(data, tf.uint8)
        # label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
        # res = 256
        # img = tf.reshape(img, [res, res, 3])
        img = tf.reshape(img, shape)
        return img

    for dataset_type in ["training", "validation"]:
        # create dataset
        data_path = path / dataset_type
        files = [str(file) for file in Path(data_path).glob("*.tfrecords")]
        dataset = tf.data.TFRecordDataset(files)

        # parse
        dataset = dataset.map(_parse_tf_record, num_parallel_calls=AUTOTUNE)

        # limit number of samples
        if config.get("limit_n_samples"):
            dataset = dataset.take(config.limit_n_samples)

        n_samples = sum(1 for record in dataset)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(n_samples))
        print(f"Using {n_samples} files for {dataset_type}.")

        # shuffle
        if shuffle:
            dataset = dataset.shuffle(n_samples, seed=seed)

        # batch
        if batch_size:
            dataset = dataset.batch(batch_size)

        datasets.append(dataset)

    train_dataset, test_dataset = datasets

    # transforms
    transforms = []

    # resize
    transforms.append(lambda x: tf.image.resize(x, image_size))
    transforms.append(lambda x: tf.ensure_shape(x, (None, *image_size, 3)))

    if color_mode == "grayscale":
        transforms.append(lambda x: tf.image.rgb_to_grayscale(x))

    # normalize
    transforms.append(lambda x: get_normalization_layer(*image_range)(x))

    for transform in transforms:
        train_dataset = train_dataset.map(transform, num_parallel_calls=AUTOTUNE)
        test_dataset = test_dataset.map(transform, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset


def _get_sine_noise_dataset(config):
    image_size = config["image_size"]
    noise_stddev = config["noise_stddev"]
    batch_size = config["batch_size"]

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if config["color_mode"] == "rgb":
        image_shape = [batch_size, *image_size, 3]
    else:
        image_shape = [batch_size, *image_size, 1]

    if isinstance(noise_stddev, int):
        noise_stddev = (noise_stddev, noise_stddev)

    def sine_noise_gen():
        stddev = np.exp(np.sin(2 * np.pi * np.arange(image_shape[1]) / 16))
        stddev_matrix = np.transpose(np.zeros(image_shape), (1, 2, 3, 0))
        stddev_matrix += stddev[:, None, None, None]
        stddev_matrix = np.transpose(stddev_matrix, (3, 0, 1, 2))
        while True:
            noise_pattern = tf.random.normal(stddev_matrix.shape, stddev=stddev_matrix)
            noise_pattern *= tf.random.uniform(
                (batch_size, 1, 1, 1), *noise_stddev
            ) / tf.math.reduce_std(noise_pattern)
            yield noise_pattern

    dataset = tf.data.Dataset.from_generator(
        sine_noise_gen,
        output_signature=tf.TensorSpec(shape=image_shape, dtype=tf.float32),
    )
    return dataset, dataset


def _get_sine_noise1D_dataset(config):
    image_size = config["image_size"]
    noise_stddev = config["noise_stddev"]
    batch_size = config["batch_size"]
    subsample_factor = config["subsample_factor"]

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if config["color_mode"] == "rgb":
        image_shape = [batch_size, int(np.prod(image_size) * 3 / subsample_factor)]
    else:
        image_shape = [batch_size, int(np.prod(image_size) / subsample_factor)]

    image_shape = image_shape + [1, 1]

    if isinstance(noise_stddev, int):
        noise_stddev = (noise_stddev, noise_stddev)

    def sine_noise_gen():
        stddev = np.exp(np.sin(2 * np.pi * np.arange(image_shape[1]) / 16))
        stddev_matrix = np.zeros(image_shape)
        stddev_matrix += stddev[:, None, None]
        while True:
            noise_pattern = tf.random.normal(stddev_matrix.shape, stddev=stddev_matrix)
            # vary magnitude
            noise_pattern *= tf.random.uniform(
                (batch_size, 1, 1, 1), *noise_stddev
            ) / tf.math.reduce_std(noise_pattern)
            yield noise_pattern

    dataset = tf.data.Dataset.from_generator(
        sine_noise_gen,
        output_signature=tf.TensorSpec(shape=image_shape, dtype=tf.float32),
    )
    return dataset, dataset
