"""Callbacks
Author(s): Tristan Stevens
"""
import shutil
from pathlib import Path

import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import tensorflow as tf
import tqdm
import yaml
from keras.callbacks import Callback
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io as sio

import wandb
from utils.corruptors import get_corruptor
from utils.inverse import _MODEL_NAMES, Denoiser, get_list_of_denoisers
from utils.metrics import Metrics
from utils.utils import get_date_filename, make_unique_path, tensor_to_images, translate


class EvalDataset(Callback):
    def __init__(
        self,
        dataset,
        config,
        model=None,
        inference=False,
        name=None,
        eval_folder=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.config = config
        self.model = model
        self.inference = inference
        self.name = name
        self.eval_freq = config.get("eval_freq")
        self.epochs = config.get("epochs")
        self.n_eval_batches = config.get("n_eval_batches", 100)
        self.num_img = config.get("num_img")
        self.N = int(np.sqrt(self.num_img))

        self.denoiser = None
        self.target_samples = None
        self.noisy_samples = None
        self.denoised_samples = None
        self.noise_samples = None
        self.results = {}
        self.results_summary = {}

        if self.config.image_range is not None:
            self.vmin, self.vmax = self.config.image_range

        if self.model:
            self.model_name = config.model_name.lower()

        if dataset is not None:
            if self.config.get("corruptor"):
                self.corruptor = get_corruptor(self.config.corruptor)(
                    config, train=False
                )
            else:
                self.corruptor = None

            batch = self.get_batch()
            if isinstance(batch, tuple):
                self.group = "supervised"
                self.batch_size = batch[0].shape[0]
            else:
                self.group = "generative"
                self.batch_size = batch.shape[0]

        if self.inference:
            if eval_folder is None:
                self.eval_folder_path = make_unique_path(
                    "./results/" / get_date_filename(f"evaluation_{name}")
                )
            else:
                self.eval_folder_path = "./results" / Path(eval_folder)
                assert (
                    self.eval_folder_path.exists()
                ), f"Eval folder {self.eval_folder_path} does not exist."

        self.model_names = _MODEL_NAMES

    def on_train_begin(self, logs=None):
        fig = self.plot_batch()
        wandbimage = wandb.Image(fig, caption="test images")
        wandb.log({"test images": wandbimage})

    def plot_batch(self):
        if self.group == "supervised":
            noisy_images, images = self.get_batch()
            images = tf.concat(
                (noisy_images[: self.num_img], images[: self.num_img]), axis=0
            )
            fig, axs = plt.subplots(self.N, self.N * 2, figsize=(8 * 2, 8))

        elif self.group == "generative":
            images = self.get_batch(self.num_img)
            if self.corruptor and self.config.paired_data:
                noisy_images = self.corruptor.corrupt(images)
                images = tf.concat((images, noisy_images, self.corruptor.noise), axis=0)
                fig, axs = plt.subplots(self.N, self.N * 3, figsize=(8 * 3, 8))
            else:
                fig, axs = plt.subplots(self.N, self.N, figsize=(8, 8))

        if self.config.image_range is not None:
            images = tensor_to_images(images, self.vmin, self.vmax)

        for img, ax in zip(images, axs.ravel(order="F")):
            img = np.squeeze(img)
            if len(img.shape) == 1:
                ax.plot(img)
            elif len(img.shape) == 2:
                ax.imshow(np.squeeze(img), cmap="gray", vmin=0, vmax=255, aspect="auto")
            elif len(img.shape) == 3:
                if self.config.color_mode == "rgb":
                    ax.imshow(img)
                elif self.config.color_mode == 2:
                    img = np.abs(img[..., 0] + 1j * img[..., 1])
                    ax.imshow(img, cmap="gray")
            else:
                raise ValueError(
                    f"Image shape {img.shape} not compatible with plotting."
                )
            ax.axis("off")

        if self.corruptor and self.config.paired_data:
            axs[0, 0].set_title("Clean", loc="left")
            axs[0, self.N].set_title("Corrupted", loc="left")
            axs[0, self.N * 2].set_title("Noise", loc="left")

        fig.tight_layout()
        return fig

    def at_inference(self, **kwargs):
        self.plot_batch()

    def get_batch(self, num=None):
        if num is None:
            return next(iter(self.dataset))
        else:
            if self.batch_size < num:
                dataset = self.dataset.unbatch().batch(num)
                batch = next(iter(dataset))
                self.batch_size = num
            else:
                batch = next(iter(self.dataset))[:num]
            return batch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_freq == 0 or ((epoch + 1) == self.epochs):
            if self.model:
                if self.model_name in ["score"]:
                    eval_score_loss = self.model.get_eval_loss(
                        dataloader=self.dataset,
                        n_batches=self.n_eval_batches,
                    )
                    wandb.log({"eval_score_loss": eval_score_loss})
                    print(f"Eval score loss: {eval_score_loss}")

    def run_metrics(self):
        self.metric_path = self.eval_folder_path / "metrics.npy"
        if self.metric_path.is_file():
            raise ValueError("Metrics already exist, please move first")

        folders = self.eval_folder_path.glob("*")
        folders = [f for f in folders if f.is_dir()]

        # sense which methods are present using folder names
        methods = [f.stem for f in folders]
        available_methods = get_list_of_denoisers()
        methods = [m for m in methods if m in available_methods]

        # sort metrics according to preferred order (for plotting)
        sort_order = [
            "nlm",
            "bm3d",
            "wvtcs",
            "gan",
            "glow",
            "sgm",
            "sgm_proj",
            "sgm_dps",
            "sgm_pigdm",
        ]
        methods = sorted(methods, key=sort_order.index)

        if "cs" not in self.config.corruptor:
            methods = ["noisy"] + methods

        metrics = Metrics(metrics="all", image_range=(0, 255))
        gt_images = self.read_images_in_folder(self.eval_folder_path / "gt")

        self.results = {}
        self.results_summary = {}
        for method in methods:
            images = self.read_images_in_folder(self.eval_folder_path / method)
            results = metrics.eval_metrics(
                gt_images,
                images,
            )
            self.results[method] = results

            self.results_summary[method] = {
                key: {
                    "mean": float(np.mean(value)),
                    "std": float(np.std(value)),
                }
                for key, value in self.results[method].items()
            }

    @staticmethod
    def read_images_in_folder(folder):
        """Read all png files in folder as image."""
        files = folder.glob("*.png")
        images = [sio.imread(file) for file in files]
        return images

    def gen_eval_dataset(self, denoiser: Denoiser):
        """Generate evaluation dataset.
        Creates two subfolders `gt` and `noisy` in `eval_folder_path`.
        """
        self.denoiser = denoiser
        noisy_list = []
        target_list = []
        noise_samples = []

        if (self.eval_folder_path / "gt").is_dir() and (
            self.eval_folder_path / "noisy"
        ).is_dir():
            print("Found existing gt and noisy files")
            return self.get_eval_dataset()

        if (self.eval_folder_path / "gt").is_dir():
            print("Found only existing gt files")
            target_list = []
            gt_files = sorted(
                (self.eval_folder_path / "gt").glob("*"), key=lambda x: int(x.stem)
            )
            for gt in gt_files:
                target_list.append(sio.imread(gt) / 255)
            data = tf.cast(tf.stack(target_list), dtype=tf.float32)
            self.noisy_samples, self.target_samples = denoiser.set_data(
                noisy_samples=None, target_samples=data
            )
        else:
            for data in self.denoiser.dataset:
                noisy, target = denoiser.set_data(
                    noisy_samples=None, target_samples=data
                )
                noisy_list.append(noisy)
                target_list.append(target)
                noise_samples.append(denoiser.corruptor.noise)
            self.noisy_samples = tf.concat(noisy_list, axis=0)
            self.target_samples = tf.concat(target_list, axis=0)
            if self.noise_samples:
                self.noise_samples = tf.concat(noise_samples, axis=0)
            self.save_images(self.target_samples, "gt", idx=0)

        self.save_images(self.noisy_samples, "noisy", idx=0)
        if self.noise_samples:
            self.save_images(self.noise_samples, "noise", idx=0)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.noisy_samples, self.target_samples)
        )
        return self.dataset

    def get_eval_dataset(self):
        """Get dataset for evaluation from a folder containing `gt` and `noisy` subfolders.
        Automatically matches the images in the two folders by name.
        """
        noisy_list = []
        target_list = []
        gt_files = (self.eval_folder_path / "gt").glob("*")
        noisy_files = (self.eval_folder_path / "noisy").glob("*")
        gt_files = sorted(gt_files, key=lambda x: int(x.stem))
        noisy_files = sorted(noisy_files, key=lambda x: int(x.stem))
        for gt, noisy in zip(gt_files, noisy_files):
            target_list.append(sio.imread(gt) / 255)
            noisy_list.append(sio.imread(noisy) / 255)
        self.noisy_samples = tf.cast(tf.stack(noisy_list), dtype=tf.float32)
        self.target_samples = tf.cast(tf.stack(target_list), dtype=tf.float32)
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.noisy_samples, self.target_samples)
        )
        if self.config.color_mode == "grayscale":
            self.dataset = self.dataset.map(
                lambda x, y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1))
            )
        return self.dataset

    def eval_inverse_problem(self, denoiser: Denoiser):
        """Evaluate inverse problem on generated dataset.
        Saves denoised results in `eval_folder_path`.
        """
        if (self.eval_folder_path / denoiser.name).is_dir():
            print(
                f"Skipping evaluation for {self.denoiser.name} "
                f"because results already exists in {self.eval_folder_path}"
            )
            return

        self.denoiser = denoiser

        metrics = []
        idx = 0

        if self.denoiser.config.get("batch_size"):
            batch_size = self.denoiser.batch_size
        else:
            batch_size = 32
            print(
                "Setting evaluation batch size automatically to "
                f"{batch_size}, please specify in config."
            )

        print(f"Using batch size {batch_size}")
        dataset = self.dataset.batch(batch_size)
        for noisy_samples, target_samples in tqdm.tqdm(
            dataset, desc=f"Evaluating {self.denoiser.name} on dataset..."
        ):
            try:
                denoised_samples = self.denoiser(
                    noisy_samples=noisy_samples,
                    target_samples=target_samples,
                    plot=False,
                )
                metrics.append(self.denoiser.eval_denoised)
            except AssertionError as ae:
                print(f"skipping evaluation because of assertion: {ae}")
                metrics.append(np.NaN)

            ## save images
            if self.denoiser.corruptor.model:
                denoised_samples = denoised_samples[0]

            self.save_images(denoised_samples, self.denoiser.name, idx)
            idx += len(target_samples)

        if self.denoiser.metrics is None:
            return

        metrics = self.denoiser.metrics.parse_metrics(metrics, reduce_mean=False)
        metrics_flattened = {
            key: [val for sublist in values for val in sublist]
            for key, values in metrics.items()
        }

        metrics_summary = {
            key: {
                "mean": float(np.mean(value)),
                "std": float(np.std(value)),
            }
            for key, value in metrics_flattened.items()
        }

        self.results[self.denoiser.name] = metrics_flattened
        self.results_summary[self.denoiser.name] = metrics_summary
        return metrics

    def save_images(self, images, name, idx):
        """Save images to directory"""
        if len(images.shape) != 4 and self.config.color_mode != "grayscale":
            print(f"Skipping saving step as {name} does have 4 dimensions")
            return
        folder = self.eval_folder_path / name
        folder.mkdir(exist_ok=True)

        images = np.clip(images, self.denoiser.vmin, self.denoiser.vmax)
        images = translate(images, (self.denoiser.vmin, self.denoiser.vmax), (0, 255))
        images = np.array(images)
        images = images.astype(np.uint8)
        for i, image in zip(range(idx, idx + len(images)), images):
            path = folder / (str(i) + ".png")
            sio.imsave(path, image)
            print(f"saving images to {path}")

    def save_results(self):
        """Saves computed metrics on every image to npy file as well as summary to yaml file."""
        if not hasattr(self, "results_summary"):
            print("Could not find results dictionary. Skipping saving results.")
            return
        self.metric_path = self.eval_folder_path / "metrics.yaml"
        with open(self.metric_path, "w+") as yml:
            yaml.dump(self.results_summary, yml, default_flow_style=False)

        np.save(self.metric_path.with_suffix(".npy"), self.results)

    def load_results(self):
        """Load results containing metrics on all images.

        Either loads from yaml file or npy file depending on file extension.
        """
        self.metric_path = self.eval_folder_path / "metrics.npy"
        if self.metric_path.suffix == ".yaml":
            self.results_summary = yaml.load(
                open(self.metric_path),
                Loader=yaml.FullLoader,
            )
            return self.results_summary
        elif self.metric_path.suffix == ".npy":
            self.results = np.load(self.metric_path, allow_pickle=True).item()
            return self.results
        else:
            raise ValueError("Unsupported metric file")

    def plot_comparison(
        self,
        models=None,
        sample_idx=8,
        dpi: int = 600,
        show_metrics: bool = False,
        fig=None,
        axs=None,
        transpose=True,
        save=True,
        metric_color="white",
        double_column=True,
    ) -> plt.Figure:
        """Plot multiple denoised results from different denoisers.

        Args:
            dpi (int, optional): dpi of plotted image. Defaults to 600.
            show_metrics (bool, optional): show metrics on plot. Defaults to True.

        Returns:
            plt.Figure: matplotlib figure object
        """
        if models is None:
            models = ["gt", "noisy", "bm3d", "gan", "glow", "sgm"]
        else:
            if transpose:
                models = ["gt", "noisy", *models[::-1]]
            else:
                models = ["gt", "noisy", *models]

        if not (self.eval_folder_path / "noisy").is_dir():
            if "noisy" in models:
                models.remove("noisy")

        if not self.config.paired_data:
            if "gt" in models:
                models.remove("gt")

        if self.config.color_mode == "grayscale":
            cmap = "gray"
        else:
            cmap = None

        gt_files = list((self.eval_folder_path / "gt").glob("*"))

        if sample_idx is None:
            sample_idx = 8

        if isinstance(sample_idx, int):
            sample_idx = np.random.choice(
                range(0, len(gt_files)), size=sample_idx, replace=False
            )
        elif isinstance(sample_idx, list):
            assert max(sample_idx) <= len(gt_files), (
                "please specify sample idx within number of"
                "generated images in evaluation folder"
            )
        else:
            raise ValueError(
                f"sample_idx should be int or list, not {type(sample_idx)}"
            )

        num_img = len(sample_idx)

        if shutil.which("latex"):
            plt.style.use(["science", "ieee"])
        else:
            plt.style.use(["science", "ieee", "no-latex"])
            print("Cannot find LaTex")

        if double_column:
            width = 2 * 3.3
        else:
            # compact (single column)
            width = 1 * 3.3

        plt.rcParams.update(
            {
                "figure.dpi": "200",
                "figure.figsize": (width, len(models) * 1.15),
            }
        )

        if transpose:
            ax_dims = (len(models), num_img)
        else:
            ax_dims = (num_img, len(models))

        fig = plt.figure()
        axs = ImageGrid(
            fig,
            111,
            nrows_ncols=ax_dims,  # creates 2x2 grid of axes
            axes_pad=0.05,  # pad between axes in inch.
        )
        title_size = 7
        axs = np.reshape(axs, ax_dims)

        if transpose:
            columns = models
            rows = sample_idx
        else:
            columns = sample_idx
            rows = models

        images = []
        for c, column in enumerate(columns):
            for r, row in enumerate(rows):
                if transpose:
                    model = column
                    idx = row
                else:
                    model = row
                    idx = column

                file = self.eval_folder_path / model / str(idx)
                file = file.with_suffix(".png")
                image = sio.imread(file)

                vmin, vmax = 0, 255

                images.append(image)
                axs[c, r].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[c, r].set_xticks([])
                axs[c, r].set_yticks([])
                for _, spine in axs[c, r].spines.items():
                    spine.set_visible(False)

                if transpose:
                    if r == 0:
                        title = self.model_names[model]
                        axs[c, r].set_ylabel(title, size=title_size)
                else:
                    if c == 0:
                        title = self.model_names[model]
                        axs[c, r].set_title(title, size=title_size)

        if show_metrics and bool(self.results):
            metric_name = True
            if transpose:
                offset = image.shape[0] * 0.1
            else:
                offset = image.shape[0] * 0.1

            x_offset = 3

            if num_img > 8:
                fontsize = 3
            else:
                fontsize = 5

            for c, column in enumerate(columns):
                for r, row in enumerate(rows):
                    for m, metric in enumerate(self.config.metrics):
                        if transpose:
                            model = column
                            idx = row
                            idx = (
                                r  # metrics are only available for the selected images
                            )
                        else:
                            model = row
                            idx = column
                            idx = (
                                c  # metrics are only available for the selected images
                            )

                        if model in ["gt", "noise"]:
                            continue
                        try:
                            evaluation = self.results[model][metric][idx]
                        except KeyError as exc:
                            raise KeyError(
                                f"No metric {metric} for model {model}, run_metrics first for {model}"
                            ) from exc

                        if metric_name:
                            axs[c, r].text(
                                x_offset,
                                offset + offset * m,
                                f"{metric}: {evaluation:.3f}",
                                color=metric_color,
                                fontsize=fontsize,
                            )
                        else:
                            axs[c, r].text(
                                x_offset,
                                offset + offset * m,
                                f"{evaluation:.3f}",
                                color=metric_color,
                                fontsize=fontsize,
                            )

        comparison_plot_path = self.eval_folder_path / "comparison.pdf"
        if save:
            try:
                fig.savefig(
                    comparison_plot_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi
                )
                fig.savefig(
                    comparison_plot_path.with_suffix(".png"),
                    bbox_inches="tight",
                    pad_inches=0.05,
                    dpi=dpi,
                )
                print(f"Succesfully saved plot to {comparison_plot_path}")
            except Exception as e:
                print(f"Could not save file to {comparison_plot_path} because of {e}")

    def plot_results(
        self,
        models: list = None,
        plot_metrics: list = None,
        box_colors: list = None,
        upper_labels: bool = True,
        legend: bool = False,
        distance: float = None,
        width: float = None,
        title: str = None,
        double_column: bool = False,
    ):
        """Box plot of results using calculated metrics.
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html

        Args:
            models (list, optional): List of models to plot. Defaults to None.
            plot_metrics (list, optional): List of metrics to plot. Defaults to None.
            box_colors (list, optional): List of colors for the boxes. Defaults to None.
            upper_labels (bool, optional): Show metrics in top of plot, above the bars.
                Defaults to True.
            legend (bool, optional): Show legend. Defaults to False.
            distance (float, optional): Distance between the boxes. Defaults to None.
            width (float, optional): Width of the boxes. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            double_column (bool, optional): Use double column width (for paper).
                Defaults to False.
        """
        if plot_metrics is None:
            plot_metrics = ["psnr", "ssim"]
        if box_colors is None:
            box_colors = [
                "darkkhaki",
                "royalblue",
                "mediumseagreen",
                "darkorange",
            ]
        if distance is None:
            distance = 0.7
        if width is None:
            width = 0.4
        if title is None:
            title = ""

        plt.style.use(["science", "ieee"])
        plt.rcParams.update(
            {
                "figure.dpi": "300",
                "xtick.direction": "out",
                "xtick.major.size": 3,
                "xtick.major.width": 0.5,
                "xtick.minor.size": 1.5,
                "xtick.minor.width": 0.5,
                "xtick.minor.visible": False,
                "xtick.top": False,
            }
        )
        if double_column:
            plt.rcParams.update({"figure.figsize": (3.3 * 2, 2.5)})

        fig, ax = plt.subplots()
        if models is None:
            models = list(self.results.keys())
            if "noisy" in models:
                models.remove("noisy")
        data = {}
        axes = {}
        offset = 1.0
        for model in models:
            metrics = self.results[model].keys()
            metrics = [m for m in plot_metrics if m in metrics]
            for metric in metrics:
                if not metric in data.keys():
                    data[metric] = []
                if len(axes.keys()) == 0:
                    axes[metric] = ax
                elif not metric in axes.keys():
                    axes[metric] = ax.twinx()
                    axes[metric].spines.right.set_position(("axes", offset))
                    offset += 0.2
                data[metric].append(self.results[model][metric])

        bp = {}
        N = len(models)
        M = len(metrics)
        offset = 0
        linewidth = 0.8
        for metric in metrics:
            bp[metric] = axes[metric].boxplot(
                data[metric],
                sym="x",
                flierprops=dict(color="black", markersize=1),
                boxprops=dict(linewidth=linewidth),
                whiskerprops=dict(linewidth=linewidth),
                capprops=dict(linewidth=linewidth),
                widths=width,
                medianprops=dict(linewidth=linewidth, color="black"),
                positions=np.arange(1 + offset, N * M + 1 + offset, M),
            )
            offset += distance

        for i, model in enumerate(models):
            for color, metric in zip(box_colors, metrics):
                box = bp[metric]["boxes"][i]
                box_x = []
                box_y = []
                for j in range(5):
                    box_x.append(box.get_xdata()[j])
                    box_y.append(box.get_ydata()[j])
                box_coords = np.column_stack([box_x, box_y])
                axes[metric].add_patch(Polygon(box_coords, facecolor=color))

        if upper_labels:
            ticks = ax.get_xticks()
            i = 0
            for color, metric in zip(box_colors, metrics):
                averages = data[metric]
                for average in averages:
                    upper_label = str(round(np.mean(average), 2))

                    ax.text(
                        ticks[i],
                        0.95,
                        upper_label,
                        transform=ax.get_xaxis_transform(),
                        horizontalalignment="center",
                        size="6",
                        fontdict={"fontname": "monospace"},
                        weight="bold",
                        color=color,
                    )

                    i += 1

        # Set the axes ranges and axes labels
        if "ssim" in metrics:
            axes["ssim"].set_ylim(0, 1)

        start_1 = 1
        start_end = M * N - 1
        width = (M - 1) * distance
        ticks = np.arange(start_1 + width / 2, start_end + width / 2 + 1, M)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [self.model_names[model] for model in models], rotation=45, fontsize=8
        )

        ax.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title=title,
            # xlabel='Model',
        )
        for axs, color in zip(axes, box_colors):
            axes[axs].set_ylabel(axs.upper(), color=color, fontweight="bold")
            # give a little more space for upper labels
            if upper_labels:
                ylims = axes[axs].get_ylim()
                axes[axs].set_ylim(ylims[0], ylims[1] * 1.06)

        if legend:
            custom_lines = [
                Line2D([0], [0], color=mcolors.to_rgb(color), lw=4)
                for color in box_colors[:M]
            ]
            ax.legend(custom_lines, metrics, loc="upper left")

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

        fig.tight_layout()
        self.metrics_plot_path = Path(self.eval_folder_path) / "metrics.pdf"
        try:
            fig.savefig(
                self.metrics_plot_path, bbox_inches="tight", pad_inches=0.05, dpi=600
            )
            fig.savefig(
                self.metrics_plot_path.with_suffix(".png"),
                bbox_inches="tight",
                pad_inches=0.05,
                dpi=600,
            )
            print(f"Succesfully saved plot to {self.metrics_plot_path}")
        except Exception as e:
            print(f"Could not save file to {self.metrics_plot_path} because of {e}")


class Monitor(Callback):
    def __init__(self, model, config, num_img=None, **kwargs):
        self.config = config
        self.eval_freq = config.eval_freq
        self.epochs = config.epochs
        self.model = model
        self.log_dir = config.log_dir
        self.duration = 5  # seconds
        if num_img is None:
            self.num_img = config.num_img
        else:
            self.num_img = num_img

        self.model_name = config.model_name.lower()

        self.random_latent_vectors = self.model.get_latent_vector(self.num_img)

        if self.config.image_range is not None:
            self.vmin, self.vmax = self.config.image_range

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_freq == 0 or ((epoch + 1) == self.epochs):
            try:
                self.plot_batch()

                self.file_name = "generated images"
                wandbimage = wandb.Image(self.fig, caption=self.file_name)
                wandb.log({self.file_name: wandbimage})
                plt.close()
            except AssertionError as ae:
                print(f"Could not plot batch at epoch {epoch} due to:\n{ae}")

    def on_train_end(self, logs=None):
        image_dir = Path(self.log_dir) / "media" / "images"
        anim_file = "training_progress.gif"
        filenames = sorted(
            list(image_dir.glob(f"{self.file_name}*.png")),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        fps = len(filenames) / self.duration
        with imageio.get_writer(image_dir / anim_file, mode="I", fps=fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
        print(f"Succesfully saved animation {anim_file} to {image_dir}")

    def plot_batch(self):
        # currently only supporting square grid of images
        n_rows = np.sqrt(self.num_img).astype(int)
        self.fig, self.axs = plt.subplots(n_rows, n_rows, figsize=(8, 8))
        self.generated_images = self.model.sample(self.random_latent_vectors)

        for img, ax in zip(self.generated_images, self.axs.ravel()):
            img = np.squeeze(img)
            if len(img.shape) == 1:
                ax.plot(img.T)
            elif len(img.shape) == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            elif len(img.shape) == 3:
                if self.config.color_mode == "rgb":
                    ax.imshow(img)
                elif self.config.color_mode == 2:
                    img = img[..., 0]
                    ax.imshow(img, vmin=0, vmax=255, cmap="gray")

            else:
                raise ValueError(
                    f"Image shape {img.shape} not compatible with plotting."
                )
            ax.axis("off")

        self.fig.tight_layout()
