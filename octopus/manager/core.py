"""OctoManager for managing Octopus experiments."""

import copy
import math
import os
from pathlib import Path

import ray
from attrs import define, field, validators

from octopus.experiment import OctoExperiment
from octopus.logger import LogGroup, get_logger
from octopus.manager.ray_parallel import init_ray, run_parallel_outer_ray, shutdown_ray
from octopus.modules import modules_inventory
from octopus.task import Task

logger = get_logger()


@define
class OctoManager:
    """Manages the execution of Octopus experiments."""

    base_experiments: list[OctoExperiment] = field(
        validator=[validators.instance_of(list)],
    )
    tasks: list[Task] = field(
        validator=[validators.instance_of(list)],
    )
    outer_parallelization: bool = field(
        default=True,
        validator=[validators.instance_of(bool)],
    )
    run_single_experiment_num: int = field(
        default=-1,
        validator=[validators.instance_of(int)],
    )

    @property
    def num_available_cpus(self) -> int:
        """Get available CPUs after reservation."""
        total_cpus = os.cpu_count()
        if total_cpus is None:
            raise RuntimeError("Could not determine number of CPUs available on the system.")
        return total_cpus

    @property
    def num_outer_workers(self) -> int:
        """Calculate number of parallel outer workers."""
        if self.run_single_experiment_num != -1:
            return 1
        return min(len(self.base_experiments), self.num_available_cpus)

    def run_outer_experiments(self):
        """Run outer experiments."""
        # start local ray instance
        self._init_ray()
        self._log_execution_info()
        self._validate_experiments()

        single_exp = self.run_single_experiment_num

        # run single experiment
        if single_exp != -1:
            self._run_single_experiment(single_exp)
        # run multiple experiments
        elif self.outer_parallelization:
            self._run_parallel_ray()
        else:
            self._run_sequential()

        # close manager
        self._close()

    def _init_ray(self):
        """Initialize Ray with available CPUs.

        CPUs reserved via config.manager.reserve_cpus are excluded.
        """
        init_ray(start_local_if_missing=True, num_cpus=self.num_available_cpus)

    def _close(self):
        """Shutdown ray instance."""
        shutdown_ray()

    def _log_execution_info(self):
        """Log execution configuration and CPU allocation."""
        logger.info(
            f"Preparing execution of experiments | "
            f"Outer parallelization: {self.outer_parallelization} | "
            f"Run single experiment: {self.run_single_experiment_num} | "
            f"Number of outer folds: {len(self.base_experiments)} | "
            f"Available CPUs: {self.num_available_cpus} | "
            f"Outer workers: {self.num_outer_workers} | "
            f"CPUs per experiment: {self._calculate_assigned_cpus()}"
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
        """Run experiments in parallel using Ray."""

        def create_execute_mlmodules_fn(base_experiment: OctoExperiment, index: int):
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
            num_workers=self.num_outer_workers,
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

        for element in self.tasks:
            self._log_workflow_task_info(element)

            # load from workflow task
            if element.load_task:
                self._load_existing_experiment(base_experiment, element)
            # create new experiment
            else:
                experiment = self._create_new_experiment(base_experiment, element)
                path_study_workflow = self._create_workflow_directory(experiment)
                path_save = self._get_save_path(path_study_workflow, experiment)
                exp_path_dict[experiment.task_id] = path_save

                self._update_experiment_if_needed(experiment, exp_path_dict)
                self._run_and_save_experiment(experiment, path_study_workflow, path_save)

    def _log_workflow_task_info(self, element):
        logger.info(
            f"Processing workflow task: {element.task_id} | "
            f"Input item: {element.depends_on_task} | "
            f"Module: {element.module} | "
            f"Description: {element.description} | "
            f"Load existing workflow task: {element.load_task}"
        )

    def _create_new_experiment(self, base_experiment: OctoExperiment, element):
        experiment = copy.deepcopy(base_experiment)
        experiment.ml_module = element.module
        experiment.ml_config = element
        experiment.id = f"{experiment.id}_{element.task_id}"
        experiment.task_id = element.task_id
        experiment.depends_on_task = element.depends_on_task
        # Note: attrs strips underscore from init param, so we assign directly to the private field
        experiment._task_path = Path(f"outersplit{experiment.experiment_id}", f"workflowtask{element.task_id}")
        experiment.num_assigned_cpus = self._calculate_assigned_cpus()
        return experiment

    def _calculate_assigned_cpus(self) -> int:
        """Calculate CPUs assigned to each individual experiment for inner parallelization.

        Each experiment uses these CPUs to parallelize its internal trainings (Bag).

        Strategy:
        - Outer parallelization: Distribute CPUs (after reservation) evenly across outer workers
        - Sequential/Single: Use all CPUs (after reservation)

        Returns:
            Number of CPUs each experiment can use for inner parallelization.
        """
        if self.outer_parallelization:
            # Distribute CPUs evenly among outer workers
            cpus_per_experiment = max(1, math.floor(self.num_available_cpus / self.num_outer_workers))
            return cpus_per_experiment
        else:
            # Sequential or single experiment: use all available CPUs
            return self.num_available_cpus

    def _create_workflow_directory(self, experiment):
        path_study_workflow = experiment.path_study.joinpath(experiment.task_path)
        path_study_workflow.mkdir(parents=True, exist_ok=True)
        return path_study_workflow

    def _get_save_path(self, path_study_workflow, experiment):
        return path_study_workflow.joinpath(f"exp{experiment.experiment_id}_{experiment.task_id}.pkl")

    def _update_experiment_if_needed(self, experiment, exp_path_dict):
        """Update from input item.

        Not for item with base input.
        """
        if experiment.depends_on_task >= 0:
            self._update_from_input_item(experiment, exp_path_dict)
        experiment.feature_groups = experiment.calculate_feature_groups(experiment.feature_columns)

    def _run_and_save_experiment(self, experiment, path_study_workflow, path_save):
        logger.info(f"Running experiment: {experiment.id}")
        experiment.to_pickle(path_save)

        module = self._get_ml_module(experiment)
        experiment = module.run_experiment()

        if experiment.results:
            # save predictions and feature importance for all keys
            for key in experiment.results:
                # save predictions
                experiment.results[key].create_prediction_df().to_parquet(
                    path_study_workflow.joinpath(
                        f"predictions_{experiment.experiment_id}_{experiment.task_id}_{key}.parquet"
                    )
                )

                # save feature importance
                experiment.results[key].create_feature_importance_df().to_parquet(
                    path_study_workflow.joinpath(
                        f"feature-importance_{experiment.experiment_id}_{experiment.task_id}_{key}.parquet"
                    )
                )

        experiment.to_pickle(path_save)

    def _get_ml_module(self, experiment):
        if experiment.ml_module in modules_inventory:
            return modules_inventory[experiment.ml_module](experiment)
        else:
            raise ValueError(f"ml_module {experiment.ml_module} not supported")

    def _load_existing_experiment(self, base_experiment, element):
        path_study_workflow = base_experiment.path_study.joinpath(
            f"outersplit{base_experiment.experiment_id}",
            f"workflowtask{element.task_id}",
        )
        path_load = path_study_workflow.joinpath(f"exp{base_experiment.experiment_id}_{element.task_id}.pkl")

        if not path_load.exists():
            raise FileNotFoundError("Workflow task to be loaded does not exist")

        experiment = OctoExperiment.from_pickle(path_load)
        logger.info(f"Loaded existing experiment from: {path_load}")
        return experiment

    def _update_from_input_item(self, experiment, path_dict):
        """Update experiment properties using input item.

        Properties updated currently, but could be expanded later:
            - selected features
            - prior feature importances
        """
        input_path = path_dict[experiment.depends_on_task]

        if not input_path.exists():
            raise FileNotFoundError("Workflow task to be loaded does not exist")

        input_experiment = OctoExperiment.from_pickle(input_path)

        experiment.feature_columns = input_experiment.selected_features
        experiment.prior_results = input_experiment.results

        logger.info(f"Prior results keys: {input_experiment.prior_results.keys()}")
