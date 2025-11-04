"""Manager module for Octopus experiments."""

from octopus.manager.core import OctoManager
from octopus.manager.ray_parallel import (
    init_ray,
    run_parallel_inner,
    run_parallel_outer_ray,
    setup_ray_for_external_library,
    shutdown_ray,
)

__all__ = [
    "OctoManager",
    "init_ray",
    "run_parallel_inner",
    "run_parallel_outer_ray",
    "setup_ray_for_external_library",
    "shutdown_ray",
]
