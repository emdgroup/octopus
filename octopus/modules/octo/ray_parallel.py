"""Parallel execution of training using Ray."""

# parallel_ray.py
import ray

from octopus.logger import get_logger

logger = get_logger()


def init_ray(address=None, num_cpus=None, **kwargs):
    """Initialize Ray once, safe to call multiple times.

    Parameters:
    - address: Connect to an existing cluster (e.g., "auto") or None for local.
    - num_cpus: Limit CPUs for the local Ray runtime. Ignored if connecting to a cluster.
    - **kwargs: Any additional ray.init keyword arguments (e.g., runtime_env, log_to_driver).
    """
    if ray.is_initialized():
        logger.info("Ray is already initialized.")
        return

    try:
        if address is not None:
            ray.init(address=address, **kwargs)
            logger.info(f"Ray initialized with address={address}.")
        else:
            ray.init(num_cpus=num_cpus, **kwargs)
            logger.info(f"Ray initialized locally with num_cpus={num_cpus}.")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise


def shutdown_ray():
    """Safely shut down Ray. No-op if Ray is not initialized."""
    if not ray.is_initialized():
        logger.info("Ray is not initialized; nothing to shut down.")
        return
    try:
        ray.shutdown()
        logger.info("Ray has been shut down.")
    except Exception as e:
        logger.error(f"Error during Ray shutdown: {e}")
        raise


def _execute_training(training, idx):
    """Execute a single training (Ray task body)."""
    try:
        result = training.fit()
        logger.info(f"Training {idx} completed successfully.")
        return result
    except Exception as e:
        print(f"Exception during training{idx}: {e}")
        print(f"Exception type: {type(e).__name__}")
        return None


@ray.remote(num_cpus=1)  # Ensure each training task reserves at most 1 CPU
def _execute_training_ray(training, idx):
    """Ray remote function for training execution with CPU limit of 1."""
    return _execute_training(training, idx)


def run_parallel_trainings(trainings):
    """Submit all trainings to Ray and return their results in the SAME ORDER as the input.

    Implementation details:
    - We build the futures list in input order and pass that ordered list to ray.get.
    - Ray guarantees that ray.get(list_of_futures) returns results in the same order.
    - For extra safety and clarity, we also map results back to their original indices.
    """
    if not ray.is_initialized():
        # Default local init; callers can explicitly call init_ray() earlier to configure differently.
        init_ray()

    futures_with_idx = [
        (_execute_training_ray.remote(training, idx), idx)
        for idx, training in enumerate(trainings)
    ]

    # Ray returns results in the same order as the list we provide to ray.get
    results_in_submission_order = ray.get([f for f, _ in futures_with_idx])

    # Explicitly place results back by their original indices to ensure order matches `trainings`
    ordered_results = [None] * len(trainings)
    for (__, idx), res in zip(futures_with_idx, results_in_submission_order):
        ordered_results[idx] = res

    return ordered_results
