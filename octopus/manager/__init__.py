"""OctoManager."""

import copy
import math
import os
from os import cpu_count
from pathlib import Path

import ray
from attrs import define, field, validators

from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.logger import LogGroup, get_logger
from octopus.manager.ray_parallel import init_ray, run_parallel_outer_ray, shutdown_ray
from octopus.modules import modules_inventory

logger = get_logger()


@define
class OctoManager:
    """Manages the execution of Octopus experiments."""

    base_experiments: list[OctoExperiment] = field(
        validator=[validators.instance_of(list)],
    )
    configs: OctoConfig = field(
        validator=[validators.instance_of(OctoConfig)],
    )

    def run_outer_experiments(self):
        """Run outer experiments."""
        # start local ray instance
        self._init_ray()
        self._log_execution_info()
        self._validate_experiments()

        single_exp = self.configs.manager.run_single_experiment_num

        # run single experiment
        if single_exp != -1:
            self._run_single_experiment(single_exp)
        # run multiple experiments
        elif self.configs.manager.outer_parallelization:
            self._run_parallel_ray()
        else:
            self._run_sequential()

        # close manager
        self._close()

    def _init_ray(self):
        """Initialize ray."""
        # reserve num_workers for outer processes
        num_workers = min(self.configs.study.n_folds_outer, cpu_count())
        if self.configs.manager.run_single_experiment_num != -1:
            num_workers = 1
        # start exactly ONE local head here; export its address for children
        init_ray(start_local_if_missing=True, num_cpus=cpu_count() - num_workers)

    def _close(self):
        # shutdown ray instance
        shutdown_ray()

    def _log_execution_info(self):
        num_workers = min(self.configs.study.n_folds_outer, cpu_count())
        logger.info(
            f"Preparing execution of experiments | "
            f"Outer parallelization: {self.configs.manager.outer_parallelization} | "
            f"Run single experiment: {self.configs.manager.run_single_experiment_num} |"
            f"Number of outer folds: {self.configs.study.n_folds_outer} | "
            f"Number of logical CPUs: {cpu_count()} | "
            f"Number of outer fold workers: {num_workers}"
        )

    def _validate_experiments(self):
        if not self.base_experiments:
            logger.error("No experiments defined")
            raise ValueError("No experiments defined")

    def _run_single_experiment(self, exp_num):
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single experiment: {exp_num}")
        self.create_execute_mlmodules(self.base_experiments[exp_num])

    def _run_sequential(self):
        logger.set_log_group(LogGroup.PROCESSING)
        for cnt, base_experiment in enumerate(self.base_experiments):
            logger.info(f"Running Outerfold: {cnt}")
            self.create_execute_mlmodules(base_experiment)

    def _run_parallel_ray(self):
        # Choose concurrency similar to your previous joblib setting
        num_workers = min(self.configs.study.n_folds_outer, os.cpu_count() or 1)

        # Wrap your existing per-experiment logic into a callable that matches (base_experiment, index)
        def create_execute_mlmodules_fn(base_experiment, index: int):
            # Keep your logging the same
            logger.set_log_group(LogGroup.PROCESSING, f"EXP {index}")
            logger.info("Starting execution")
            try:
                # Your original logic
                self.create_execute_mlmodules(base_experiment)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"EXP {index}")
                logger.info("Completed successfully")
                return True  # or any meaningful result
            except Exception as e:
                logger.exception(f"Exception occurred while executing task {index}: {e!s}")
                return None  # or raise to fail the task

        # Run with Ray
        results = run_parallel_outer_ray(
            base_experiments=self.base_experiments,
            create_execute_mlmodules=create_execute_mlmodules_fn,
            num_workers=num_workers,
        )
        return results

    def create_execute_mlmodules(self, base_experiment: OctoExperiment):
        """Create and execute ml modules."""
        # Child connects to ray the local ray instance
        # using the address exported by the main process
        # If running inside a Ray worker, this will be True and we won't try to init again.
        if not ray.is_initialized():
            # Only used when you call this function outside of Ray.
            init_ray(address=os.environ.get("RAY_ADDRESS"), start_local_if_missing=False)

        exp_path_dict: dict[int, Path] = {}

        for element in self.configs.sequence.sequence_items:
            self._log_sequence_item_info(element)

            # load from sequence item
            if element.load_sequence_item:
                self._load_existing_experiment(base_experiment, element)
            # create new experiment
            else:
                experiment = self._create_new_experiment(base_experiment, element)
                path_study_sequence = self._create_sequence_directory(experiment)
                path_save = self._get_save_path(path_study_sequence, experiment)
                exp_path_dict[experiment.sequence_id] = path_save

                self._update_experiment_if_needed(experiment, exp_path_dict)
                self._run_and_save_experiment(experiment, path_study_sequence, path_save)

    def _log_sequence_item_info(self, element):
        logger.info(
            f"Processing sequence item: {element.sequence_id} | "
            f"Input item: {element.input_sequence_id} | "
            f"Module: {element.module} | "
            f"Description: {element.description} | "
            f"Load existing sequence item: {element.load_sequence_item}"
        )

    def _create_new_experiment(self, base_experiment: OctoExperiment, element):
        experiment = copy.deepcopy(base_experiment)
        experiment.ml_module = element.module
        experiment.ml_config = element
        experiment.id = f"{experiment.id}_{element.sequence_id}"
        experiment.sequence_id = element.sequence_id
        experiment.input_sequence_id = element.input_sequence_id
        experiment.path_sequence_item = Path(f"experiment{experiment.experiment_id}", f"sequence{element.sequence_id}")
        experiment.num_assigned_cpus = self._calculate_assigned_cpus()
        return experiment

    def _calculate_assigned_cpus(self):
        if self.configs.manager.outer_parallelization:
            n_outer = self.configs.study.n_folds_outer
            # reserve 2 CPUs for the outer processes
            return math.floor((cpu_count() - 2) / n_outer)
        elif self.configs.manager.run_single_experiment_num != -1:
            return cpu_count() - 1
        else:
            return cpu_count() - 1

    def _create_sequence_directory(self, experiment):
        path_study_sequence = experiment.path_study.joinpath(experiment.path_sequence_item)
        path_study_sequence.mkdir(parents=True, exist_ok=True)
        return path_study_sequence

    def _get_save_path(self, path_study_sequence, experiment):
        return path_study_sequence.joinpath(f"exp{experiment.experiment_id}_{experiment.sequence_id}.pkl")

    def _update_experiment_if_needed(self, experiment, exp_path_dict):
        """Update from input item.

        Not for item with base input.
        """
        if experiment.input_sequence_id >= 0:
            self._update_from_input_item(experiment, exp_path_dict)
        experiment.feature_groups = experiment.calculate_feature_groups(experiment.feature_columns)

    def _run_and_save_experiment(self, experiment, path_study_sequence, path_save):
        logger.info(f"Running experiment: {experiment.id}")
        experiment.to_pickle(path_save)

        module = self._get_ml_module(experiment)
        experiment = module.run_experiment()

        if experiment.results:
            # save predictions and feature importance for all keys
            for key in experiment.results:
                # save predictions
                experiment.results[key].create_prediction_df().to_parquet(
                    path_study_sequence.joinpath(
                        f"predictions_{experiment.experiment_id}_{experiment.sequence_id}_{key}.parquet"
                    )
                )

                # save feature importance
                experiment.results[key].create_feature_importance_df().to_parquet(
                    path_study_sequence.joinpath(
                        f"feature-importance_{experiment.experiment_id}_{experiment.sequence_id}_{key}.parquet"
                    )
                )

        experiment.to_pickle(path_save)

    def _get_ml_module(self, experiment):
        if experiment.ml_module in modules_inventory:
            return modules_inventory[experiment.ml_module](experiment)
        else:
            raise ValueError(f"ml_module {experiment.ml_module} not supported")

    def _load_existing_experiment(self, base_experiment, element):
        path_study_sequence = base_experiment.path_study.joinpath(
            f"experiment{base_experiment.experiment_id}",
            f"sequence{element.sequence_id}",
        )
        path_load = path_study_sequence.joinpath(f"exp{base_experiment.experiment_id}_{element.sequence_id}.pkl")

        if not path_load.exists():
            raise FileNotFoundError("Sequence item to be loaded does not exist")

        experiment = OctoExperiment.from_pickle(path_load)
        logger.info(f"Loaded existing experiment from: {path_load}")
        return experiment

    def _update_from_input_item(self, experiment, path_dict):
        """Update experiment properties using input item.

        Properties updated currently, but could be expanded later:
            - selected features
            - prior feature importances
        """
        input_path = path_dict[experiment.input_sequence_id]

        if not input_path.exists():
            raise FileNotFoundError("Sequence item to be loaded does not exist")

        input_experiment = OctoExperiment.from_pickle(input_path)

        experiment.feature_columns = input_experiment.selected_features
        experiment.prior_results = input_experiment.results

        logger.info(f"Prior results keys: {input_experiment.prior_results.keys()}")
