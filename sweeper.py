"""Sweep over hyperparameters using ax platform
Author(s): Tristan Stevens
"""
import os

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10000"

import json
from pathlib import Path

import numpy as np
from ax.service.ax_client import AxClient
from easydict import EasyDict as edict

from utils.inverse import get_denoiser
from utils.utils import (
    get_date_filename,
    get_date_string,
    load_config_from_yaml,
)


class Sweeper:
    """Sweeper class for hyperparemeter tuning experiments."""

    def __init__(
        self,
        config,
        sweep_config,
        dataset,
        name="tuning_experiment",
        allow_plot=True,
    ):
        """Init sweeper.

        Args:
            config (dict): config object with general parameters.
            sweep_config (dict): config object with sweep parameters.
            dataset (tf.Dataset): tensorflow dataset.
            name (str, optional): name of the experiment.
                Defaults to 'tuning_experiment'.
            allow_plot (bool, optional): plot intermediate results.
                Defaults to True.
        """
        self.sweep_id = "sweep_" + get_date_string("%Y-%m-%d_%H%M%S")

        self.config = config

        if Path(sweep_config).is_file():
            path = sweep_config
        else:
            path = f"./configs/sweeps/{sweep_config}.yaml"
        self.sweep_config = load_config_from_yaml(path)

        self.objective = self.sweep_config.objective
        self.name = name
        self.allow_plot = allow_plot

        Denoiser = get_denoiser(config.denoiser)

        self.denoiser = Denoiser(
            config=config,
            dataset=dataset,
            metrics=self.objective,
            sweep_id=self.sweep_id,
        )

        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            name=name,
            parameters=self.sweep_config.parameters,
            objective_name=self.objective,
            minimize=self.denoiser.metrics.minimize[self.objective],
        )

        self.noisy_samples, self.target_samples = self.denoiser.set_data()

    def __str__(self):
        return f"Sweeper: {self.sweep_id}"

    def run(self):
        """Run and evaluate all trials of hyperparameter optimization."""
        # Run number of optimization steps
        for _ in range(self.sweep_config.num_iterations):
            selected_parameters, trial_index = self.ax_client.get_next_trial()

            try:
                objective = self.evaluate(selected_parameters)
                if np.isnan(objective[self.objective][0]):
                    raise AssertionError("nan objective")

            except AssertionError as e:
                print(f"Skipping {selected_parameters} because of assertion {e}")
                # setting metric to worse case value and uncertainty
                if self.denoiser.metrics.minimize[self.objective]:
                    objective = 100
                else:
                    objective = -100
                objective = {self.objective: (objective, 100.0)}

            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=objective,
            )

        self.export()

    def evaluate(self, parameters):
        """Evaluate given parameters and return objective for optimization."""
        # TODO: add check whether parameter is even available in denoiser
        # this to prevent updating parameters that don't exist

        # update denoiser config with new parameters
        self.denoiser.config = edict(self.denoiser.config | parameters)

        if self.config.model_name.lower() == "score":
            self.denoiser.set_sampler()

        _ = self.denoiser(
            noisy_samples=self.noisy_samples,
            target_samples=self.target_samples,
            plot=self.allow_plot,
            preprocess=False,
        )

        objective = np.mean(self.denoiser.eval_denoised[self.objective])
        return {self.objective: (objective, 0.0)}

    def export(self, output_path=None):
        """Export best found parameters to json file."""
        best_parameters, _ = self.ax_client.get_best_parameters()

        print(f"best_parameters: {best_parameters}")

        if output_path is None:
            output_path = get_date_filename(
                Path(self.config.log_dir, "sweeps")
                / (self.name + "_best_parameters.json"),
            )

        with open(Path(output_path), "w") as f:
            json.dump(best_parameters, fp=f)

        print(f"Succesfully saved best parameters to {output_path}")
