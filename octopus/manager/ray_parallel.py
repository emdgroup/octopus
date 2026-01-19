"""Ray parallelization for outer and inner loops."""

import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import ray
import threadpoolctl
from ray import ObjectRef
from upath import UPath

from octopus.logger import set_logger_filename

if TYPE_CHECKING:
    from octopus.experiment import OctoExperiment


# =============================================================================
# Ray Lifecycle
# =============================================================================


def init_ray(
    address: str | None = None,
    num_cpus: int | None = None,
    start_local_if_missing: bool = False,
    **kwargs,
) -> None:
    """Initialize Ray for the current process.

    Args:
        address: Ray head address. If None, uses RAY_ADDRESS or RAY_HEAD_ADDRESS env vars.
        num_cpus: CPU limit when starting locally.
        start_local_if_missing: Start local Ray if no address available.
        **kwargs: Extra args forwarded to ray.init.

    Raises:
        RuntimeError: If no address is available and start_local_if_missing is False.
    """
    if ray.is_initialized():
        return

    addr = address or os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")
    if addr:
        ray.init(
            address=addr,
            runtime_env={"worker_process_setup_hook": _check_parallelization_disabled},
            **kwargs,
        )
        return

    if start_local_if_missing:
        ray.init(
            num_cpus=num_cpus,
            runtime_env={"worker_process_setup_hook": _check_parallelization_disabled},
            **kwargs,
        )
        return

    raise RuntimeError(
        "No Ray address provided. Set RAY_ADDRESS env, pass address='auto', or call init_ray(..., start_local_if_missing=True) once in the driver."
    )


def shutdown_ray() -> None:
    """Shut down Ray if initialized."""
    if ray.is_initialized():
        ray.shutdown()
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_ADDRESS", None)


def setup_ray_for_external_library() -> None:
    """Set RAY_ADDRESS to allow external libraries to use our Ray instance."""
    if ray.is_initialized():
        ray_address = ray.get_runtime_context().gcs_address
        if ray_address:
            os.environ["RAY_ADDRESS"] = ray_address
    else:
        os.environ.pop("RAY_ADDRESS", None)


def _check_parallelization_disabled() -> None:
    """Verify thread-level parallelization is disabled to prevent oversubscription."""
    from octopus.modules import _PARALLELIZATION_ENV_VARS  # noqa: PLC0415

    for lib in threadpoolctl.threadpool_info():
        if lib["num_threads"] > 1:
            raise RuntimeError(
                f"Active thread-level parallelization detected in {lib}. Please disable thread-level parallelization."
            )

    for env_var in _PARALLELIZATION_ENV_VARS:
        if os.environ.get(env_var, None) != "1":
            raise RuntimeError(
                f"Environment variable {env_var} is set to {os.environ.get(env_var)}. "
                "Please set it to 1 to disable thread-level parallelization."
            )


# =============================================================================
# Parallel Execution
# =============================================================================


def _setup_worker_logging(log_dir: UPath):
    """Setup logging for Ray worker processes."""
    # We could log to individual files, e.g. per task:
    # task_id = ray.get_runtime_context().get_task_id()
    # worker_log_file = log_dir / f"octo_worker.{task_id}.log"
    # but for now every worker just logs into the same file
    worker_log_file = log_dir / "octo_manager.log"
    set_logger_filename(log_file=worker_log_file)


def run_parallel_outer_ray[T](
    base_experiments: Iterable["OctoExperiment"],
    create_execute_mlmodules: Callable[["OctoExperiment", int], T],
    log_dir: UPath,
    num_workers: int,
) -> list[T]:
    """Execute experiments in parallel using Ray with backpressure control.

    Args:
        base_experiments: Experiments to process.
        create_execute_mlmodules: Function(experiment, index) -> result.
        log_dir: Directory to store individual Ray worker logs.
        num_workers: Maximum concurrent tasks.

    Returns:
        Results in same order as input experiments.
    """
    init_ray(start_local_if_missing=True)

    @ray.remote(num_cpus=0)
    def outer_task(idx: int, experiment: "OctoExperiment", log_dir: UPath):
        _setup_worker_logging(log_dir)
        return idx, create_execute_mlmodules(experiment, idx)

    items = list(base_experiments)
    if not items:
        return []

    # Backpressure: limit concurrent tasks
    max_concurrent = max(1, min(num_workers, len(items)))
    results: list[T | None] = [None] * len(items)
    inflight: list[ObjectRef] = []
    next_i = 0

    # Start initial batch
    while next_i < len(items) and len(inflight) < max_concurrent:
        inflight.append(outer_task.remote(next_i, items[next_i], log_dir))
        next_i += 1

    # Process completions and submit new tasks
    while inflight:
        done, inflight = ray.wait(inflight, num_returns=1)
        idx, res = ray.get(done[0])
        results[idx] = res
        if next_i < len(items):
            inflight.append(outer_task.remote(next_i, items[next_i], log_dir))
            next_i += 1

    return cast("list[T]", results)


def run_parallel_inner(trainings: Iterable[Any], log_dir: UPath, num_cpus: int = 1) -> list[Any]:
    """Run training.fit() for each item in parallel.

    Args:
        trainings: Objects with fit() method.
        log_dir: Directory to store individual Ray worker logs.
        num_cpus: CPUs per training task.

    Returns:
        Results from each training.fit() in input order.

    Raises:
        RuntimeError: If Ray is not initialized.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Call init_ray() first.")

    @ray.remote(num_cpus=num_cpus)
    def execute_training(training: Any, idx: int, log_dir: UPath) -> Any:
        _setup_worker_logging(log_dir)
        return training.fit()

    futures = [execute_training.remote(training, idx, log_dir) for idx, training in enumerate(trainings)]
    return ray.get(futures)
