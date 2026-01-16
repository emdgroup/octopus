"""Ray parallelization for outer and inner loops."""

import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import ray
import threadpoolctl
from ray import ObjectRef

if TYPE_CHECKING:
    from octopus.experiment import OctoExperiment


def _get_env_address() -> str | None:
    """Return Ray address from environment or None if not set."""
    return os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")


def _check_parallelization_disabled() -> None:
    """Raise an error if any kind of active parallelization (OMP, MKL, threadpools, ...) can be detected.

    This is required to prevent accidental OMP paralleliztion inside ray processes that can lead to oversubscription.
    Used as a worker_process_setup_hook in distributed Ray clusters.
    """
    from octopus.modules import _PARALLELIZATION_ENV_VARS  # noqa: PLC0415

    for lib in threadpoolctl.threadpool_info():
        if lib["num_threads"] > 1:
            raise RuntimeError(
                f"Active thread-level parallelization detected in {lib}. "
                "This may lead to resource oversubscription and slow execution. "
                "Please disable thread-level parallelization by setting respective "
                "environment variables."
            )

    for env_var in _PARALLELIZATION_ENV_VARS:
        if os.environ.get(env_var, None) != "1":
            raise RuntimeError(
                f"Environment variable {env_var} is set to {os.environ.get(env_var)}. "
                "This may lead to resource oversubscription and slow execution. "
                "Please set it to 1 to disable thread-level parallelization."
            )


def _verify_parallelization_disabled() -> None:
    """Verify that parallelization environment variables are properly configured.

    Only checks environment variables, not actual threadpool state, because:
    - Ray workers are separate processes that inherit env vars, not driver threadpool state
    - Environment variables persist across test boundaries
    - Checking driver threadpool state can give false positives in test environments
    - Tests confirm Ray workers correctly use single-threaded libraries when env vars are set

    Used in the driver process before starting Ray to verify configuration.
    """
    from octopus.modules import _PARALLELIZATION_ENV_VARS  # noqa: PLC0415

    for env_var in _PARALLELIZATION_ENV_VARS:
        val = os.environ.get(env_var)
        if val != "1":
            raise RuntimeError(
                f"Environment variable {env_var} is set to {val}. "
                "This may lead to resource oversubscription in Ray workers. "
                f"Please set {env_var}=1 to ensure single-threaded execution."
            )


def init_ray(
    address: str | None = None,
    num_cpus: int | None = None,
    start_local_if_missing: bool = False,
    **kwargs,
) -> None:
    """Initialize Ray for the current process.

    Connects to an existing cluster if an address is provided or set via
    environment variables; otherwise, optionally starts a local Ray instance.

    For local Ray instances, parallelization settings are verified in the driver
    process before starting Ray, and workers inherit the driver's environment.
    This avoids the overhead of runtime_env package reinstallation.

    Args:
        address: Ray head address (e.g., "auto", "127.0.0.1:6379"). If None, uses
            env vars RAY_ADDRESS or RAY_HEAD_ADDRESS if set.
        num_cpus: CPU limit when starting a local Ray instance (only used if starting locally).
        start_local_if_missing: If True and no address is available, start a local Ray instance.
        **kwargs: Extra args forwarded to ray.init (e.g., log_to_driver, namespace).

    Raises:
        RuntimeError: If no address is available and start_local_if_missing is False.
    """
    if ray.is_initialized():
        return

    addr = address or _get_env_address()

    if addr:
        # Connecting to existing Ray instance (local or remote)
        # Workers connecting to the local instance will inherit the driver's environment
        ray.init(address=addr, **kwargs)
        return

    if start_local_if_missing:
        # Starting new local Ray instance - verify settings in driver
        # Workers will inherit driver environment, avoiding package reinstallation
        _verify_parallelization_disabled()
        ray.init(num_cpus=num_cpus, **kwargs)
        return

    raise RuntimeError(
        "No Ray address provided. Set RAY_ADDRESS env, pass address='auto', "
        "or call init_ray(..., start_local_if_missing=True) once in the driver."
    )


def shutdown_ray() -> None:
    """Shut down Ray if initialized. Safe to call multiple times."""
    if ray.is_initialized():
        ray.shutdown()
    # Clear RAY_ADDRESS to avoid stale references after shutdown
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_ADDRESS", None)


def setup_ray_for_external_library() -> None:
    """Configure environment to enable external libraries to use the existing Ray instance.

    Sets RAY_ADDRESS to the current Ray GCS address, preventing external libraries
    (e.g., AutoGluon, Ray Tune) from creating separate Ray instances that would
    cause resource conflicts.

    Should be called before using external libraries that may use Ray.
    """
    if ray.is_initialized():
        ray_address = ray.get_runtime_context().gcs_address
        if ray_address:
            os.environ["RAY_ADDRESS"] = ray_address
    else:
        # If Ray is not initialized, clear the RAY_ADDRESS to avoid stale references
        os.environ.pop("RAY_ADDRESS", None)


def run_parallel_outer_ray[T](
    base_experiments: Iterable["OctoExperiment"],
    create_execute_mlmodules: Callable[["OctoExperiment", int], T],
    num_workers: int,
) -> list[T]:
    """Execute create_execute_mlmodules(base_experiment, index) in parallel using Ray.

    Preserves input order and limits concurrency to num_workers. Outer tasks reserve
    0 CPUs so inner Ray work can use available CPUs.

    Args:
        base_experiments: Items to process.
        create_execute_mlmodules: Function called as create_execute_mlmodules(base_experiment, index).
            If your function only accepts (base_experiment), wrap it (e.g., lambda be, i: f(be)).
        num_workers: Maximum number of concurrent outer tasks.

    Returns:
        Results from create_execute_mlmodules in the same order as base_experiments.
    """
    # Ensure Ray is ready in the driver (connect or start local)
    init_ray(start_local_if_missing=True)

    # ff outerjobs do non-trivial CPU tasks - use a small fractional CPU for outers,
    # e.g., num_cpus=0.1. This limits oversubscription.
    @ray.remote(num_cpus=0)
    def _outer_task(idx: int, base_exp: "OctoExperiment"):
        # Do not re-initialize Ray here; workers already have a Ray context.
        return idx, create_execute_mlmodules(base_exp, idx)

    items = list(base_experiments)
    n = len(items)
    if n == 0:
        return []

    max_concurrent = max(1, min(num_workers, n))
    results: list[T | None] = [None] * n
    inflight: list[ObjectRef] = []
    next_i = 0

    # Prime up to max_concurrent tasks
    while next_i < n and len(inflight) < max_concurrent:
        inflight.append(_outer_task.remote(next_i, items[next_i]))
        next_i += 1

    # Drain with backpressure; fill results by original index to preserve order
    while inflight:
        done, inflight = ray.wait(inflight, num_returns=1)
        idx, res = ray.get(done[0])
        results[idx] = res
        if next_i < n:
            inflight.append(_outer_task.remote(next_i, items[next_i]))
            next_i += 1

    return cast("list[T]", results)


def _execute_training(training: Any, idx: int) -> Any:
    """Call training.fit() and return the result."""
    return training.fit()


def run_parallel_inner(trainings: Iterable[Any], num_cpus: int = 1) -> list[Any]:
    """Run training.fit() for each item in parallel using Ray and preserve input order.

    Args:
        trainings: Iterable of training-like objects. Each object must implement a fit() method.
        num_cpus: Number of CPUs to allocate per training task (default: 1).

    Returns:
        List[Any]: Results from each training.fit(), in the same order as the input iterable.

    Raises:
        RuntimeError: If Ray is not initialized in this process.

    Notes:
        Exceptions raised by individual trainings will propagate as
        ray.exceptions.RayTaskError when awaiting results with ray.get(). If you prefer
        to handle or log per-item errors and continue, wrap each training so its fit()
        catches/logs exceptions and returns a sentinel (e.g., None).
    """
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray is not initialized in this process. Call init_ray(...) in the driver, "
            "or ensure this runs inside a Ray task."
        )

    # Create the remote function with configurable num_cpus
    execute_training_ray = ray.remote(num_cpus=num_cpus)(_execute_training)

    futures = [execute_training_ray.remote(training, idx) for idx, training in enumerate(trainings)]
    return ray.get(futures)
