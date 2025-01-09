"""Inference script for generative models and various inverse tasks
Author(s): Tristan Stevens
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict

from datasets import get_dataset
from generators.models import get_model
from sweeper import Sweeper
from utils.callbacks import EvalDataset, Monitor
from utils.checkpoints import ModelCheckpoint
from utils.gpu_config import set_gpu_usage
from utils.inverse import (
    animate_multiple_denoisers,
    get_denoiser,
    get_list_of_denoisers,
    plot_multiple_denoisers,
)
from utils.runs import assert_run_exists, init_config
from utils.utils import (
    add_args_to_config,
    load_config_from_yaml,
    set_random_seed,
    update_dict,
)


def get_inference_args():
    """Parse input arguments for inference script."""
    task_choices = [
        "denoise",
        "sample",
        "evaluate",
        "show_dataset",
        "plot_results",
        "run_metrics",
    ]
    model_choices = get_list_of_denoisers()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        default="mnist",
        type=str,
        help="experiment name located at ./configs/inference/<experiment>)",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="denoise",
        choices=task_choices,
        help="which task to run",
        type=str,
    )
    parser.add_argument(
        "-n", "--num_img", default=None, type=int, help="number of images"
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        required=False,
        choices=model_choices,
        help="list of models to run",
    )
    parser.add_argument(
        "-s",
        "--sweep",
        default=False,
        type=str,
        help="sweep config located at ./configs/sweeps/<sweep>)",
    )
    parser.add_argument(
        "-ef",
        "--eval_folder",
        default=None,
        type=str,
        help="eval file located at ./results/<eval_folder>)",
    )

    args = parser.parse_args()
    return args


def inference_setup(args):
    """Setup inference and load dataset with args and config parameters."""

    # Load inference config -> inf_cfg
    if Path(args.experiment).is_file():
        path = Path(args.experiment)
        args.experiment = path.stem
    else:
        path = Path(f"./configs/inference/{args.experiment}")
    path = path.with_suffix(".yaml")

    inf_cfg = load_config_from_yaml(path)

    inf_cfg = add_args_to_config(args, inf_cfg)

    if "run_id" in inf_cfg:
        run_ids = inf_cfg.run_id
    else:
        raise ValueError("please provide a run_id")

    if args.task != "plot_results":
        set_gpu_usage(inf_cfg.device)

    set_random_seed(inf_cfg.seed)

    # Initialize a config object with only dataset params (uniform across experiments)
    dataset_cfg = init_config(run_ids["sgm"], just_dataset=True)
    # Merge training and inference configs dataset_cfg + inf_cfg -> cfg
    cfg = update_dict(dataset_cfg, inf_cfg)

    if args.task in ["plot_results", "sample", "run_metrics"]:
        dataset = None
    else:
        _, dataset = get_dataset(cfg)

    return cfg, dataset, run_ids


def sample(config, run_ids):
    """Sample"""
    for model_name in config.models:
        run_id = run_ids.get(model_name)
        if run_id is None:
            print(
                f"skipping {model_name} as it is not a generative model or does not exist"
            )
            continue
        model_config = init_config(run_id, config)
        model_config = edict({**model_config, **model_config.get(model_name)})
        assert_run_exists(run_id, model_name)

        model = get_model(model_config, training=False)
        ckpt = ModelCheckpoint(model, config=model_config)
        ckpt.restore(model_config.get("checkpoint_file"))

        if ckpt.model_library == "torch":
            model.set_actnorm_init()
            model.eval()

        cb = Monitor(model, model_config)
        cb.plot_batch()


def denoise(config, run_ids, dataset):
    """Single inference for solving inverse task with generative models."""
    denoiser_list = []
    for denoiser in config.models:
        # merge cfg with training_cfg of denoiser -> config
        run_id = run_ids.get(denoiser)
        assert_run_exists(run_id, denoiser)
        model_config = init_config(run_id, config)
        Denoiser = get_denoiser(denoiser)

        # reuse corruptor across denoisers if corruptor is not model specific
        # essentially corruptor is only initialized in first denoiser and
        # passed onto the subsequent denoisers.
        corruptor = None
        if len(denoiser_list) > 0:
            if denoiser_list[-1].corruptor.model is None:
                corruptor = denoiser_list[-1].corruptor

        denoiser = Denoiser(config=model_config, dataset=dataset, corruptor=corruptor)

        # for single denoiser plot regular
        if len(config.models) == 1:
            denoised_samples = denoiser()
            if model_config.zoom:
                denoiser.plot(zoom=model_config.zoom, show_metrics=False)
            if model_config.keep_track:
                denoiser.animate(duration=5)

        # if multiple denoisers, reuse the inference data
        # and plot together
        else:
            if len(denoiser_list) == 0:
                # set inference data once and reuse later
                noisy_samples, target_samples = denoiser.set_data()
                # store random measurement matrix A and apply later
                if "cs" in denoiser.corruptor.name:
                    A = denoiser.corruptor.A

            if "cs" in denoiser.corruptor.name:
                denoiser.corruptor.A = A

            # solve inverse task
            denoised_samples = denoiser(
                noisy_samples,
                target_samples,
                plot=False,
            )

        denoiser_list.append(denoiser)

    # plot results of multiple denoisers side by side
    if len(config.models) > 1:
        plot_multiple_denoisers(denoiser_list, figsize=model_config.get("figsize"))
        if model_config.get("keep_track") and (
            set(config.models) & {"sgm", "gan", "glow"}
        ):
            animate_multiple_denoisers(denoiser_list, duration=5)


def run_sweep(config, run_ids, dataset):
    """Sweep"""
    if len(config.models) > 1:
        raise ValueError("Can only choose one model to perform sweep.")
    denoiser = config.models[0]

    run_id = run_ids.get(denoiser)
    assert_run_exists(run_id, denoiser)
    config = init_config(run_id, config)
    config.denoiser = denoiser

    sw = Sweeper(
        config=config,
        sweep_config=config.sweep,
        dataset=dataset,
    )
    sw.run()
    plt.close("all")


def evaluate(config, run_ids, dataset):
    """Evaluate"""
    denoisers = config.models
    cb = EvalDataset(
        dataset=dataset,
        config=config,
        inference=True,
        name=config.experiment,
        eval_folder=config.get("eval_folder"),
    )
    config.metrics = "all"
    config.keep_track = False
    if "glow" in denoisers:
        config.glow.show_optimization_progress = False
    gen_dataset = True
    for denoiser in denoisers:
        run_id = run_ids.get(denoiser)
        assert_run_exists(run_id, denoiser)
        model_config = init_config(run_id, config)

        Denoiser = get_denoiser(denoiser)
        denoiser = Denoiser(config=model_config, dataset=dataset, verbose=True)
        if gen_dataset:
            cb.gen_eval_dataset(denoiser)
            gen_dataset = False
            if "cs" in denoiser.corruptor.name:
                A = denoiser.corruptor.A
                np.save(cb.eval_folder_path / "measurement_matrix.npy", A)

        if "cs" in denoiser.corruptor.name:
            denoiser.corruptor.A = A
        cb.eval_inverse_problem(denoiser)

    if config.paired_data:
        cb.save_results()
        cb.plot_results(config.models)


def show_dataset(config, dataset):
    """Show dataset."""
    # disable corruptor model as we do not need it for
    # just showing dataset.
    config.disable_corruptor_model = True
    config.batch_size = config.num_img
    cb = EvalDataset(dataset=dataset, config=config)
    cb.plot_batch()


def plot_results(config):
    """Plot results."""
    config.disable_corruptor_model = True
    config.corruptor = None
    cb = EvalDataset(
        dataset=None, config=config, inference=True, eval_folder=config.eval_folder
    )
    if config.paired_data:
        try:
            cb.load_results()
            cb.plot_results(config.models)
        except:
            print("Couldnt find results to plot. Please run_metrics first.")
    cb.plot_comparison(config.models, config.num_img)


def run_metrics(config):
    """Compute metrics on evaluation dataset."""
    cb = EvalDataset(
        dataset=None, config=config, inference=True, eval_folder=config.eval_folder
    )
    config.metrics = "all"
    cb.run_metrics()
    cb.save_results()
    cb.plot_results(config.models)
    cb.plot_comparison(config.models, config.num_img)


if __name__ == "__main__":
    # argparse arguments -> args
    args = get_inference_args()

    # setup data paths and wandb env variables
    os.environ["WANDB_MODE"] = "disabled"

    args.data_paths = None

    # load inference config file and dataset
    # also load run_ids (wandb string / path to model weights and config)
    config, dataset, run_ids = inference_setup(args)

    # Switch statemtent for tasks
    if args.task == "denoise":
        if config.sweep:
            run_sweep(config, run_ids, dataset)
        else:
            denoise(config, run_ids, dataset)

    elif args.task == "sample":
        sample(config, run_ids)

    elif args.task == "evaluate":
        evaluate(config, run_ids, dataset)

    elif args.task == "show_dataset":
        show_dataset(config, dataset)

    elif args.task == "plot_results":
        plot_results(config)

    elif args.task == "run_metrics":
        run_metrics(config)

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
