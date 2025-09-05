"""OctoManager."""

import copy
import math
from os import cpu_count
from pathlib import Path
from typing import Dict, List

from attrs import define, field, validators
from joblib import Parallel, delayed

from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.modules import modules_inventory

from .logger import LogGroup, get_logger

logger = get_logger()


@define
class OctoManager:
    """Manages the execution of Octopus experiments."""

    base_experiments: List[OctoExperiment] = field(
        validator=[validators.instance_of(list)],
    )
    configs: OctoConfig = field(
        validator=[validators.instance_of(OctoConfig)],
    )

    def run_outer_experiments(self):
        """Run outer experiments."""
        self._log_execution_info()
        self._validate_experiments()

        single_exp = self.configs.manager.run_single_experiment_num

        # run single experiment
        if single_exp != -1:
            self._run_single_experiment(single_exp)
        # run multiple experiments
        elif self.configs.manager.outer_parallelization:
            self._run_parallel()
        else:
            self._run_sequential()

    def _log_execution_info(self):
        num_workers = min(self.configs.study.n_folds_outer, cpu_count())
        logger.info(
            f"Preparing execution of experiments | "
            f"Outer parallelization: {self.configs.manager.outer_parallelization} | "
            f"Run single experiment: {self.configs.manager.run_single_experiment_num} |"
            f" Number of outer folds: {self.configs.study.n_folds_outer} | "
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

    def _run_parallel(self):
        num_workers = min(self.configs.study.n_folds_outer, cpu_count())
        logger.info(f"Starting parallel execution with {num_workers} workers")
        with Parallel(n_jobs=num_workers) as parallel:
            parallel(delayed(self._execute_task)(base_experiment, index) for index, base_experiment in enumerate(self.base_experiments))

    def _execute_task(self, base_experiment, index):
        logger.set_log_group(LogGroup.PROCESSING, f"EXP {index}")
        logger.info("Starting execution")
        try:
            self.create_execute_mlmodules(base_experiment)
            logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"EXP {index}")
            logger.info("Completed successfully")
        except Exception as e:
            logger.exception(f"Exception occurred while executing task {index}: {str(e)}")

    def create_execute_mlmodules(self, base_experiment: OctoExperiment):
        """Create and execute ml modules."""
        exp_path_dict: Dict[int, Path] = {}

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
            return math.floor(cpu_count() / self.configs.study.n_folds_outer)
        elif self.configs.manager.run_single_experiment_num != -1:
            return cpu_count()
        else:
            return cpu_count()

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

        # save predictions
        experiment.results["best"].create_prediction_df().to_parquet(
            path_study_sequence.joinpath(f"predictions_{experiment.experiment_id}_{experiment.sequence_id}.parquet")
        )

        # save feature importances
        experiment.results["best"].create_feature_importance_df().to_parquet(
            path_study_sequence.joinpath(f"feature-importance_{experiment.experiment_id}_{experiment.sequence_id}.parquet")
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

        print("Prior results keys: ", input_experiment.prior_results.keys())
