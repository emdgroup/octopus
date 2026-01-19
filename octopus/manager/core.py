"""OctoManager for managing Octopus experiments."""

import math
import os

from attrs import define, field, validators
from upath import UPath

from octopus.experiment import OctoExperiment
from octopus.logger import get_logger
from octopus.manager.execution import (
    ExecutionStrategy,
    ParallelRayStrategy,
    SequentialStrategy,
    SingleExperimentStrategy,
)
from octopus.manager.ray_parallel import init_ray, shutdown_ray
from octopus.manager.workflow_runner import WorkflowTaskRunner
from octopus.task import Task

logger = get_logger()


def get_available_cpus() -> int:
    """Get available CPUs on the system."""
    total_cpus = os.cpu_count()
    if total_cpus is None:
        raise RuntimeError("Could not determine number of CPUs.")
    return total_cpus


@define(frozen=True)
class ResourceConfig:
    """Immutable configuration for CPU resources.

    Attributes:
        num_cpus: Total available CPUs on the system.
        num_workers: Number of parallel outer workers.
        cpus_per_experiment: CPUs allocated to each experiment for inner parallelization.
        outer_parallelization: Whether outer parallelization is enabled.
        run_single_experiment_num: Index of single experiment to run (-1 for all).
        num_experiments: Total number of experiments in the study.
    """

    num_cpus: int
    num_workers: int
    cpus_per_experiment: int
    outer_parallelization: bool
    run_single_experiment_num: int
    num_experiments: int

    @classmethod
    def create(
        cls,
        num_experiments: int,
        outer_parallelization: bool,
        run_single_experiment_num: int,
        num_cpus: int | None = None,
    ) -> "ResourceConfig":
        """Create ResourceConfig with computed values.

        Args:
            num_experiments: Total number of experiments in the study.
            outer_parallelization: Whether to run experiments in parallel.
            run_single_experiment_num: Index of single experiment to run (-1 for all).
            num_cpus: Total CPUs available (auto-detected if None).

        Returns:
            ResourceConfig with computed worker and CPU allocation.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if num_experiments <= 0:
            raise ValueError(f"num_experiments must be positive, got {num_experiments}")

        if run_single_experiment_num < -1:
            raise ValueError(
                f"run_single_experiment_num must be -1 (all experiments) or a valid index >= 0, "
                f"got {run_single_experiment_num}"
            )
        if run_single_experiment_num >= num_experiments:
            raise ValueError(
                f"run_single_experiment_num ({run_single_experiment_num}) must be less than "
                f"num_experiments ({num_experiments})"
            )

        # Get or validate num_cpus
        if num_cpus is None:
            num_cpus = get_available_cpus()
        elif num_cpus <= 0:
            raise ValueError(f"num_cpus must be positive, got {num_cpus}")

        # Calculate effective number of experiments for resource allocation
        effective_num_experiments = 1 if run_single_experiment_num != -1 else num_experiments

        # Calculate resource allocation
        num_workers = min(effective_num_experiments, num_cpus)
        if num_workers == 0:
            raise ValueError(
                f"Cannot allocate resources: num_workers computed as 0 "
                f"(effective_num_experiments={effective_num_experiments}, num_cpus={num_cpus})"
            )

        cpus_per_experiment = max(1, math.floor(num_cpus / num_workers)) if outer_parallelization else num_cpus

        return cls(
            num_cpus=num_cpus,
            num_workers=num_workers,
            cpus_per_experiment=cpus_per_experiment,
            outer_parallelization=outer_parallelization,
            run_single_experiment_num=run_single_experiment_num,
            num_experiments=num_experiments,
        )

    def __str__(self) -> str:
        """Return string representation of resource configuration."""
        return (
            f"Parallelization: {self.outer_parallelization} | "
            f"Single exp: {self.run_single_experiment_num} | "
            f"Outer folds: {self.num_experiments} | "
            f"CPUs: {self.num_cpus} | "
            f"Workers: {self.num_workers} | "
            f"CPUs/exp: {self.cpus_per_experiment}"
        )


@define
class OctoManager:
    """Orchestrates the execution of Octopus experiments."""

    base_experiments: list[OctoExperiment] = field(validator=[validators.instance_of(list)])
    workflow: list[Task] = field(validator=[validators.instance_of(list)])
    log_dir: UPath = field(
        validator=[validators.instance_of(UPath)],
    )
    outer_parallelization: bool = field(default=True, validator=[validators.instance_of(bool)])
    run_single_experiment_num: int = field(default=-1, validator=[validators.instance_of(int)])

    def run_outer_experiments(self) -> None:
        """Run outer experiments."""
        if not self.base_experiments:
            logger.error("No experiments defined")
            raise ValueError("No experiments defined")

        # Initialize Ray upfront to ensure worker setup hooks are registered before any workflows execute.
        # This is critical for:
        # 1. Inner parallelization: ML modules (e.g., Octo, AutoGluon) may spawn Ray workers for their
        #    internal operations (bagging, hyperparameter tuning) even when outer_parallelization=False
        # 2. Safety checks: The worker setup hook (_check_parallelization_disabled) must be configured
        #    before any Ray workers start, to detect and prevent thread-level parallelization issues
        # 3. Lifecycle clarity: Explicit init → run → shutdown at the manager level makes the
        #    Ray lifecycle predictable and easier to reason about
        init_ray(start_local_if_missing=True)

        resources = ResourceConfig.create(
            num_experiments=len(self.base_experiments),
            outer_parallelization=self.outer_parallelization,
            run_single_experiment_num=self.run_single_experiment_num,
        )
        logger.info(f"Preparing execution | {resources}")

        try:
            runner = WorkflowTaskRunner(self.workflow, resources.cpus_per_experiment, self.log_dir)
            strategy = self._select_strategy(resources.num_workers)
            strategy.execute(self.base_experiments, runner.run)
        finally:
            shutdown_ray()

    def _select_strategy(self, num_workers: int) -> ExecutionStrategy:
        """Select execution strategy based on configuration.

        Args:
            num_workers: Number of parallel workers for ParallelRayStrategy.

        Returns:
            Appropriate execution strategy based on configuration.
        """
        if self.run_single_experiment_num != -1:
            return SingleExperimentStrategy(self.run_single_experiment_num)
        if self.outer_parallelization:
            return ParallelRayStrategy(num_workers, self.log_dir)
        return SequentialStrategy()
