"""Run script listing and managing all training runs.
Author(s): Tristan Stevens
"""

from pathlib import Path

import wandb
import yaml
from easydict import EasyDict as edict
from huggingface_hub import login, snapshot_download

runs = {}


def yellow(text):
    """Color text yellow."""
    return f"\033[33m{text}\033[0m"


def print_run_info(run_id):
    """Print run_id info in a nice way for debugging

    Args:
        run_id (str): string of a run_id

    Raises:
        ValueError: run_id cannot be found

    Returns:
        idx: index of the given run for all the runs.
    """
    idx = next((i for i, d in enumerate(runs) if d["run_id"] == run_id), None)
    if idx is None:
        raise ValueError(f"Cannot find run: {run_id}. Please add to runs dict.")
    run = runs[idx]
    dataset = run["dataset"]
    size = run["size"]
    group = run["group"]
    model = run["model"]
    msg = f" Choosing run {run_id} trained on {dataset} with image size {size}"
    msg_full = msg + f"\n {model} is a {group} model"

    print(len(msg) * "=")
    print(msg_full)
    print(len(msg) * "=")
    print()

    if idx is None:
        raise ValueError(f"Run id {run_id} not found!")

    return idx


def assert_run_exists(run_id, model=None):
    """Check if run_id exists among defined runs.

    Args:
        run_id (str): either string for run_id (defined in runs)
            or a path to model checkpoint folder. Can also be None,
            but then an error is given if the model is recognized in
            the runs. If the model is not recognized, run_id is ignored.
            For example, bm3d is not a trained model, and therefore no
            assertion should be thrown.
        model (str, optional): string with model name. Defaults to None.
            Can be used to relax the assertion, when the model name is
            not recognized.
    """
    ids = [run["run_id"] for run in runs]

    # model is not recognized in the existing runs -> no assertion error
    if model is not None:
        models = [run["model"] for run in runs]
        if model not in models:
            return
    # run_id is a checkpoint folder -> no assertion error
    if Path(run_id).exists():
        return
    # run_id is a huggingface model -> no assertion error
    if run_id.startswith("hf://"):
        return

    # run_id is not a path, and the model is known, therefore
    # the run_id should be listed in the runs, if not -> assertion error.
    assert run_id in ids, f"Unknown run id {run_id} for model {model}"


def init_config(run_id=None, update_config=None, just_dataset=False, verbose=True):
    """Combines / loads a config and merges with existing config.

    Args:
        run_id (str, optional): wandb run_id to load training config from.
            Or can also be a path to where config (*.yaml) is stored.
            Finally can also be a string `hf://<repo_id>` to download
            a model from huggingface hub.
            Defaults to None.
        update_config (dict, optional): if run_id is provided, loaded config
            is combined with update_config. Else, just use update_config.
            Defaults to None.
        just_dataset (bool, optional): Just load parameters from config dict
            that are necessary for loading dataset. Defaults to False.
        verbose (bool, optional): enable print statements. Defaults to True.

    Raises:
        ValueError: Cannot find config from run_id path / wandb string.

    Returns:
        dict: config object / dict.
    """
    if run_id:
        # Download from huggingface hub
        if run_id.startswith("hf://"):
            repo_id = run_id.split("hf://")[1]
            login(new_session=False)

            url = "https://huggingface.co/" + repo_id
            print(f"Using model from huggingface hub: {yellow(url)}")

            run_id = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
            )

        # get config from wandb if not a path
        if not Path(run_id).exists():
            # assert_run_exists(run_id)
            if not just_dataset:
                if verbose:
                    print_run_info(run_id)
            try:
                api = wandb.Api()
                run = api.run(f"deep_generative/{run_id}")
            except Exception as e:
                raise ValueError(
                    f"Using wandb directories, but cannot find run {run_id} "
                    f"for checkpoints and config. Are you sure {run_id} is a folder "
                    "with a *.yaml file or a valid wandb run id?"
                ) from e
            group = run.group
            assert group in [
                "supervised",
                "generative",
            ], f"Run type of type {group} is not supported."
            if verbose:
                print(f"wandb: Loaded config from {run.job_type} run {run.name}\n")

            config = run.config
            config["log_dir"] = Path(run.dir) / "files"
        else:
            run = Path(run_id)
            config_file = list(run.glob("*.yaml"))
            if len(config_file) != 1:
                raise ValueError(
                    "Folder can / should only contain a " "single .yaml config file"
                )
            with open(config_file[0]) as yml:
                config = yaml.load(yml, Loader=yaml.FullLoader)
            config["log_dir"] = run

        if update_config:
            config = {**config, **update_config}

        if just_dataset:
            keep_keys = [
                "data_root",
                "dataset_name",
                "image_size",
                "batch_size",
                "paired_data",
                "image_range",
                "color_mode",
                "seed",
            ]
            config = {k: config.get(k) for k in keep_keys}

        config = edict(config)

    else:
        config = edict(update_config)

    return config
