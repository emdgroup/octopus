"""Execution strategies for running experiments."""

from typing import TYPE_CHECKING, Protocol

from attrs import define
from upath import UPath

from octopus.logger import LogGroup, get_logger
from octopus.manager.ray_parallel import run_parallel_outer_ray

if TYPE_CHECKING:
    from collections.abc import Callable

    from octopus.experiment import OctoExperiment

logger = get_logger()


class ExecutionStrategy(Protocol):
    """Protocol for experiment execution strategies."""

    def execute(
        self,
        experiments: list["OctoExperiment"],
        run_fn: "Callable[[OctoExperiment], None]",
    ) -> None:
        """Execute experiments using this strategy."""
        ...


@define
class SingleExperimentStrategy:
    """Run a single experiment by index."""

    experiment_index: int

    def execute(
        self,
        experiments: list["OctoExperiment"],
        run_fn: "Callable[[OctoExperiment], None]",
    ) -> None:
        """Execute only the experiment at experiment_index."""
        logger.set_log_group(LogGroup.PROCESSING)
        logger.info(f"Running single experiment: {self.experiment_index}")
        run_fn(experiments[self.experiment_index])


@define
class SequentialStrategy:
    """Run experiments one after another."""

    def execute(
        self,
        experiments: list["OctoExperiment"],
        run_fn: "Callable[[OctoExperiment], None]",
    ) -> None:
        """Execute all experiments sequentially."""
        logger.set_log_group(LogGroup.PROCESSING)
        for idx, experiment in enumerate(experiments):
            logger.info(f"Running outer split: {idx}")
            run_fn(experiment)


@define
class ParallelRayStrategy:
    """Run experiments in parallel using Ray."""

    num_workers: int
    log_dir: UPath

    def execute(
        self,
        experiments: list["OctoExperiment"],
        run_fn: "Callable[[OctoExperiment], None]",
    ) -> None:
        """Execute all experiments in parallel using Ray."""

        def wrapped_run(experiment: "OctoExperiment", index: int):
            logger.set_log_group(LogGroup.PROCESSING, f"EXP {index}")
            logger.info("Starting execution")
            try:
                run_fn(experiment)
                logger.set_log_group(LogGroup.PREPARE_EXECUTION, f"EXP {index}")
                logger.info("Completed successfully")
                return True
            except Exception as e:
                logger.exception(f"Exception in task {index}: {e!s}")
                return None

        run_parallel_outer_ray(
            base_experiments=experiments,
            create_execute_mlmodules=wrapped_run,
            log_dir=self.log_dir,
            num_workers=self.num_workers,
        )
