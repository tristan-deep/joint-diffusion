"""Training script for generative models
Author(s): Tristan Stevens
"""
import argparse
from pathlib import Path

import matplotlib
from keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

import wandb
from datasets import get_dataset
from generators.models import get_model
from utils.callbacks import EvalDataset, Monitor
from utils.checkpoints import ModelCheckpoint
from utils.git_info import get_git_summary
from utils.gpu_config import set_gpu_usage
from utils.utils import (check_model_library, random_augmentation,
                         set_random_seed)

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="configs/training/glow_celeba.yaml",
    type=str,
    help="relative path to config file",
)
parser.add_argument("-e", "--run_eagerly", default=False, type=bool, help="run eagerly")
args = parser.parse_args()

# handle absolute / relative paths
if not Path(args.config).exists():
    path = Path(args.config).with_suffix(".yaml")
    args.config = f"./configs/training/{path}"

print(f"Using config file: {args.config}")

run = wandb.init(
    project="deep_generative",
    group="generative",
    config=args.config,
    job_type="train",
    allow_val_change=True,
)

print(f"wandb: {run.job_type} run {run.name}\n")

config = wandb.config
config.update({"log_dir": run.dir})
set_gpu_usage(config.get("device"))
set_random_seed(config.seed)
config.update({"git": get_git_summary()})

dataset, test_dataset = get_dataset(config)
if config.get("augmentation"):
    print("Training with augmented dataset")
    dataset = dataset.map(random_augmentation(config))

model = get_model(config, run_eagerly=args.run_eagerly, plot_summary=True)
model_library = check_model_library(model)
print(f"Monitoring loss: {model.monitor_loss}")

callbacks = [
    EvalDataset(model=model, dataset=dataset, config=config),
    Monitor(model=model, config=config),
    ModelCheckpoint(model=model, config=config),
]

if model_library == "tensorflow":
    callbacks += [
        WandbCallback(),
        ReduceLROnPlateau(monitor=model.monitor_loss, factor=0.3, verbose=1),
    ]

if model_library == "pytorch":
    callbacks += []

model.fit(
    dataset,
    epochs=config.epochs,
    callbacks=callbacks,
    steps_per_epoch=config.get("steps_per_epoch"),
)

run.finish()
