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
    """

    num_cpus: int
    num_workers: int
    cpus_per_experiment: int

    @classmethod
    def create(
        cls,
        num_experiments: int,
        outer_parallelization: bool,
        num_cpus: int | None = None,
    ) -> "ResourceConfig":
        """Create ResourceConfig with computed values."""
        if num_cpus is None:
            num_cpus = get_available_cpus()

        num_workers = min(num_experiments, num_cpus)
        cpus_per_experiment = max(1, math.floor(num_cpus / num_workers)) if outer_parallelization else num_cpus

        return cls(
            num_cpus=num_cpus,
            num_workers=num_workers,
            cpus_per_experiment=cpus_per_experiment,
        )

    def log(self, outer_parallelization: bool, run_single: int, num_experiments: int) -> None:
        """Log resource configuration."""
        logger.info(
            f"Preparing execution | "
            f"Parallelization: {outer_parallelization} | "
            f"Single exp: {run_single} | "
            f"Outer folds: {num_experiments} | "
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
        )
        resources.log(
            outer_parallelization=self.outer_parallelization,
            run_single=self.run_single_experiment_num,
            num_experiments=len(self.base_experiments),
        )

        try:
            runner = WorkflowTaskRunner(self.workflow, resources, self.log_dir)
            strategy = self._select_strategy(resources)
            strategy.execute(self.base_experiments, runner.run)
        finally:
            shutdown_ray()

    def _select_strategy(self, resources: ResourceConfig) -> ExecutionStrategy:
        """Select execution strategy based on configuration."""
        if self.run_single_experiment_num != -1:
            return SingleExperimentStrategy(self.run_single_experiment_num)
        if self.outer_parallelization:
            return ParallelRayStrategy(resources, self.log_dir)
        return SequentialStrategy()
