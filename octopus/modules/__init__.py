"""Init modules."""

import os
import platform

import threadpoolctl

# Import optional dependencies conditionally
# This prevents Ray worker deserialization failures when optional dependencies are not available

# AutoGluon (optional dependency)
try:
    from .autogluon import AGCore, AutoGluon
except ImportError:
    AGCore = None  # type: ignore
    AutoGluon = None  # type: ignore

# Boruta (optional dependency)
try:
    from .boruta import Boruta, BorutaCore
except ImportError:
    Boruta = None  # type: ignore[assignment,misc]
    BorutaCore = None

# SFS (optional dependency - uses mlxtend)
try:
    from .sfs import Sfs, SfsCore
except ImportError:
    Sfs = None  # type: ignore[assignment,misc]
    SfsCore = None

# Core modules (always available)
from .efs import Efs, EfsCore
from .mrmr import Mrmr, MrmrCore
from .octo import Octo, OctoCore
from .rfe import Rfe, RfeCore
from .rfe2 import Rfe2, Rfe2Core
from .roc import Roc, RocCore

# Define the __all__ variable to specify what is available to import from this package
__all__ = ["AutoGluon", "Boruta", "Efs", "Mrmr", "Octo", "Rfe", "Rfe2", "Roc", "Sfs"]

# Type for the modules inventory dictionary
ModulesInventoryType = dict[str, type]

# Inventory for all available modules (core modules)
modules_inventory: ModulesInventoryType = {
    "octo": OctoCore,
    "mrmr": MrmrCore,
    "rfe": RfeCore,
    "rfe2": Rfe2Core,
    "roc": RocCore,
    "efs": EfsCore,
}

# Add optional modules to inventory only if they're available
if AGCore is not None:
    modules_inventory["autogluon"] = AGCore

if BorutaCore is not None:
    modules_inventory["boruta"] = BorutaCore

if SfsCore is not None:
    modules_inventory["sfs"] = SfsCore

_PARALLELIZATION_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for env_var in _PARALLELIZATION_ENV_VARS:
    if (num_threads := os.environ.setdefault(env_var, "1")) != "1":
        if platform.system() == "Darwin":
            print(
                f"Warning: {env_var} is set to {num_threads} on macOS. "
                "This may lead to issues/crashes in some libraries. "
                f"Consider setting {env_var}=1 for better stability."
            )
        else:
            print(
                f"Warning: {env_var} is set to {num_threads}. "
                "This may lead to resource oversubscription and slow execution. "
                f"Consider setting {env_var}=1 or at least perform "
                "a thorough threading performance evaluation."
            )

_THREADPOOL_LIMIT = threadpoolctl.threadpool_limits(limits=1)

del os
del platform
del threadpoolctl
