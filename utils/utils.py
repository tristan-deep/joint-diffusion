"""Utilities
Author(s): Tristan Stevens
"""
import datetime
import functools
import random
import tarfile
import time
import zipfile
from pathlib import Path

import cv2
import cvxopt
import gdown
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
import torch
import tqdm
import yaml
from easydict import EasyDict as edict
from matplotlib import animation
from PIL import Image


def timefunc(func):
    """Decorator that times performance of methods of a class. The wrapper
    utilizes the verbose attribute of the class."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            start = time.perf_counter()
            func_value = func(self, *args, **kwargs)
            run_time = time.perf_counter() - start
            print(f"Function {func.__name__!r} in {run_time:.4f} secs")
        else:
            func_value = func(self, *args, **kwargs)
        return func_value

    return wrapper


def tqdm_progress_bar(total, *args, **kwargs):
    """Adds tqdm progress bar to a method as decorator."""
    pbar = tqdm.tqdm(total=total, *args, **kwargs)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pbar.update()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def set_random_seed(seed=None):
    """Set random seed to all random generators."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    cvxopt.setseed(seed)
    torch.random.manual_seed(0)
    return seed


def get_normalization_layer(a, b, x_min=0, x_max=255):
    """Normalization layer.

    Args:
        a (float): minimum value of range to map to.
        b (float): maximum value of range to map to.
        x_min (float, optional): min value of input image. Defaults to 0.
        x_max (float, optional): max value of input image. Defaults to 255.

    Returns:
        tf layer: tensorflow Rescaling layer.

    """
    scale = (b - a) / (x_max - x_min)
    offset = a
    return tf.keras.layers.Rescaling(scale=scale, offset=offset)


def tensor_to_images(x, x_min=-1, x_max=1, a=0, b=255):
    """Convert / quantize tensors to image values (uint32)."""
    x = np.clip(x, x_min, x_max)
    scale = (x_max - x_min) / (b - a)
    offset = x_min
    image = (x - offset) / scale
    return tf.cast(image, dtype=tf.uint32)


def images_to_tensor(x, a, b, x_min=0, x_max=255):
    """Convert / dequantize images to tensors."""
    image = tf.cast(x, dtype=tf.float32)
    scale = (b - a) / (x_max - x_min)
    offset = a
    image = image * scale + offset
    return image


def save_to_gif(images, filename, fps=20):
    """Saves a sequence of images to .gif file.
    Args:
        images: list of images (numpy arrays).
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
    """
    duration = 1 / (fps) * 1000  # milliseconds per frame

    # convert grayscale images to RGB
    if len(images[0].shape) == 2:
        images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

    pillow_img, *pillow_imgs = [Image.fromarray(img) for img in images]

    pillow_img.save(
        fp=filename,
        format="GIF",
        append_images=pillow_imgs,
        save_all=True,
        loop=0,
        duration=duration,
        interlace=False,
        optimize=False,
    )
    return print(f"Succesfully saved GIF to -> {filename}")


def save_to_video(images, filename, fps=20, colorformat="rgb"):
    """Saves a sequence of images to video.
    Args:
        images: list of images (numpy arrays [0, 255] (int)).
        filename: string containing filename to which data should be written.
        fps: frames per second of rendered format.
        colorformat: when images with colorchannels are passed and this arg is
            set to rgb, the images are converted to bgr for cv2.
    """
    img_shape = images[0].shape
    # img_shape cv2 = (width, height)
    if len(img_shape) == 3:
        img_shape = img_shape[:2][::-1]
        color = 1
        if colorformat == "rgb":
            images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
    else:
        img_shape = img_shape[::-1]
        color = 0

    out = cv2.VideoWriter(
        str(filename),
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        img_shape,
        isColor=color,
    )
    for img in images:
        out.write(img)

    out.release()
    print(f"Succesfully saved MP4 to {filename}")


def save_animation(fig, fig_contents, filename, fps, verbose=1):
    """Saves a sequence of matplotlib artist objects to file.

    Args:
        fig (plt.fig): matplotlib Figure object on which the content is generated.
        fig_contents (List): list of matplotlib artist objects (AxesImage)
            example: fig_content = plt.imshow(image)
        filename (str): string containing filename to which data should be written.
            extension of filename either 'gif' or 'mp4'.
        fps (float): frames per second of rendered format.
    """
    # this step is necessary, since the ArtistAnimation requires each element
    # of the list to be a sequence of artist objects.
    if not isinstance(fig_contents[0], list):
        fig_contents = [[c] for c in fig_contents]

    filename = Path(filename)
    ani = animation.ArtistAnimation(fig, fig_contents)
    extension = filename.suffix.split(".")[-1]
    assert extension in ("gif", "mp4")

    if extension == "gif":
        writer = animation.PillowWriter(fps)
    elif extension == "mp4":
        writer = animation.FFMpegWriter(fps)
    ani.save(str(filename), writer=writer)

    if verbose:
        return print(f"Succesfully saved animation to -> {filename}")


def plot_image_grid(
    images,
    ncols=None,
    cmap="gray",
    vmin=0,
    vmax=1,
    titles=None,
    suptitle=None,
    aspect=None,
    **kwargs,
):
    """Plot a batch of images in a grid.

    Args:
        images (ndarray): batch of images.
        ncols (int, optional): Number of columns. Defaults to None.
        cmap (str, optional): Colormap. Defaults to 'gray'.
        vmin (float, optional): Minimum plot value. Defaults to 0.
        vmax (float, optional): Maximum plot value. Defaults to 1.
        titles (list, optional): List of titles for subplots. Defaults to None.
        suptitle (str, optional): Title for the plot. Defaults to None.
        aspect (optional): Aspect ratio for imshow.
        **kwargs: arguments for plt.Figure.

    Returns:
        fig (figure): Matplotlib figure object

    """
    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    axes = axes.flatten()[: len(imgs)]

    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())):
        if img is not None:
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = np.squeeze(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
            if titles:
                ax.set_title(titles[i])
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig


def get_custom_image(path, config):
    """Open a custom image from path."""
    image = Image.open(Path(path)).resize((config.image_size, config.image_size))
    image = tf.expand_dims(image, axis=0)
    image = images_to_tensor(image, *config.image_range)


def load_config_from_yaml(path, wandb_file=False):
    """Load configuration file from yaml into dictionary."""
    with open(Path(path)) as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)
    if dictionary:
        if wandb_file:
            new_dictionary = {}
            for key, value in dictionary.items():
                if "value" in value:
                    new_dictionary[key] = value["value"]
                else:
                    new_dictionary[key] = value
            dictionary = new_dictionary
        return edict(dictionary)
    else:
        return {}


def save_dict_to_yaml(dictionary, path):
    """Save dictionary to yaml file."""
    with open(Path(path), "w") as file:
        yaml.dump(dictionary, file, default_flow_style=False)


def update_dict(dictionary: dict, update: dict):
    """Update a dictionary with another dictionary."""
    # dictionary = edict(dictionary | update)
    dictionary = edict({**dictionary, **update})
    return dictionary


def make_unique_path(save_dir):
    """Create unique directory from save_dir ussing incremental suffix."""
    save_dir = Path(save_dir)
    try:
        save_dir.mkdir(exist_ok=False, parents=True)
    except:
        unique_dir_found = False
        post_fix = 0
        while not unique_dir_found:
            try:
                Path(str(save_dir) + f"_{post_fix}").mkdir(exist_ok=False, parents=True)
                unique_dir_found = True
                save_dir = Path(str(save_dir) + f"_{post_fix}")
            except:
                post_fix += 1
    return save_dir


def create_unique_filename(filename):
    """Create unique filename by appendix counting integer index to `filename`."""
    filename = Path(filename)
    if not filename.is_file():
        return filename

    i = 0
    base_name = filename.stem
    while filename.is_file():
        filename = filename.parent / (base_name + f"_{i}" + filename.suffix)
        i += 1
    return filename


def get_date_string(string: str = None):
    """Generate a date string for current time, according to format specified by `string`."""
    now = datetime.datetime.now()
    if string is None:
        string = "%Y_%m_%d"

    date_str = now.strftime(string)
    return date_str


def get_date_filename(filename):
    """Generate a unique filename based on `filename` but with date prepended."""
    filename = Path(filename)
    date_str = get_date_string()

    new_filename = date_str + "_" + filename.name
    filename = filename.parent / new_filename
    filename.parent.mkdir(exist_ok=True, parents=True)
    filename = create_unique_filename(filename)

    return filename


def translate(array, range_from, range_to):
    """Map values in array from one range to other.

    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.

    Returns:
        (ndarray): translated array
    """
    leftMin, leftMax = range_from
    rightMin, rightMax = range_to
    if leftMin == leftMax:
        assert ValueError("ranges are not compatible")
        # return np.ones_like(array) * rightMax

    # Convert the left range into a 0-1 range (float)
    valueScaled = (array - leftMin) / (leftMax - leftMin)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * (rightMax - rightMin))


def tf_tensor_to_torch(tf_tensor, device="cuda"):
    """Convert tensorflow tensor to torch tensor."""
    if len(tf_tensor.shape) == 4:
        tf_tensor = np.transpose(tf_tensor, (0, 3, 1, 2))
    torch_tensor = np.copy(tf_tensor)
    return torch.from_numpy(torch_tensor).to(device=device)


def convert_torch_tensor(torch_tensor, to="numpy"):
    """Convert torch tensor to numpy or tensorflow tensor"""
    assert to in ["numpy", "tensorflow"]
    numpy_array = torch_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    if to == "numpy":
        return numpy_array
    else:
        return tf.convert_to_tensor(numpy_array)


def check_model_library(model):
    """Check whether a model is a pytorch or tensorflow model."""
    if isinstance(model, torch.nn.Module):
        model_library = "torch"
    elif isinstance(model, tf.keras.Model):
        model_library = "tensorflow"
    else:
        raise NotImplementedError("unknown model library")
    return model_library


def get_latest_checkpoint(checkpoint_dir: str, extension: str, split=None):
    """Retrieve latest checkpoint in checkpoint directory.

    Inspired on tf.train.latest_checkpoint

    Args:
        checkpoint_dir (str): directory that points to checkpoints.
        extension (str): consider files only with extensions specified.
        split (str, optional): split filenames based on provided character
            and sort list of filenames based on remainder. Defaults to None.

    Returns:
        str: filename that points to latest checkpoint
    """
    path = Path(checkpoint_dir)
    files = list(path.glob(f"*.{extension}"))
    if len(files) == 1:
        return files[0]

    if split is not None:
        key = lambda path: int(path.stem.rsplit(split, 1)[1])
        files = sorted(files, key=key)

    if len(files) == 0:
        return None

    return files[-1]


def tf_expand_multiple_dims(tensor, n):
    """Add multiple (`n`) singleton dimensions to a tensor."""
    shape = tf.shape(tensor)
    return tf.reshape(tensor, tf.concat([shape, [1] * n], axis=0))


def random_augmentation(config: dict):
    """Random tensorflow augmentation operations.

    Function can be mapped to tensorflow dataset as follows:
    dataset = dataset.map(random_augmentation(config))

    config dictionary can contain `fliplr`, `flipud` and `brightness`.

    """
    transforms = []
    aug_cfg = config.augmentation

    # Random flip left right
    if aug_cfg["fliplr"]:
        transforms.append(tf.image.random_flip_left_right)
    # Random flip up down
    if aug_cfg["flipud"]:
        transforms.append(tf.image.random_flip_up_down)
    # Random brightness level
    if aug_cfg["brightness"]:

        def brightness(x):
            x = tf.image.random_brightness(x, max_delta=aug_cfg["brightness"])
            x = tf.clip_by_value(x, *config.image_range)
            return x

        transforms.append(brightness)

    # Stack all augmentations into a single function
    def _random_augmentation(image):
        for transform in transforms:
            image = transform(image)
        return image

    return _random_augmentation


def add_args_to_config(args, config, verbose=False):
    """Add command line arguments to the configuration file."""
    for key, value in vars(args).items():
        if value is not None:
            if verbose:
                print(f"changing {key} from {config.get(key)} to {value}")
            setattr(config, key, value)
    return config


def download_file(url, save_path):
    """Download a file from url and save to disk."""
    filename = url.split("/")[-1]
    print(f"Downloading file: {filename} to {save_path}...")
    filename = Path(save_path) / filename
    if filename.is_file():
        print("File already exists")
        return filename
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.tqdm(
            desc=url,
            total=total,
            unit="b",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    pbar.update(len(chunk))
                    f.write(chunk)
    print("Download complete")
    return filename


def download_and_unpack(url, save_path):
    """Downloads *tar.gz from url and saves to save_path."""
    if "drive.google.com" in url:
        filename = gdown.download(url, save_path, quiet=False)
    else:
        filename = download_file(url, save_path)
    filename = Path(filename)
    print(f"Unpacking {filename}")
    if filename.suffix == ".zip":
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(filename.parent)
        file = filename.name
    else:
        with tarfile.open(filename) as tar:
            file = tar.extractall(filename.parent)
        filename.unlink()
    return print(f"Succesfully saved {url} to {file}")


def convert_to_integers(lst: str) -> list:
    """
    This function takes a string of space separated integers
    and converts it into a list of integers.

    Args:
    str_numbers (str): A string of space separated integers.

    Returns:
    list: A list of integers.

    Example:
    >>> convert_to_integers(['13 11 0 12 14'])
    [13, 11, 0, 12, 14]
    """
    if len(lst) != 1:
        return lst
    try:
        return [int(x) for x in lst[0].split()]
    except:
        return lst
