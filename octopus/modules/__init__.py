# __init__.py
"""Init modules."""

from .autogluon import AGCore, AutoGluon
from .boruta import Boruta, BorutaCore
from .efs import Efs, EfsCore
from .mrmr import Mrmr, MrmrCore
from .octo import Octo, OctoCore
from .rfe import Rfe, RfeCore
from .rfe2 import Rfe2, Rfe2Core
from .roc import Roc, RocCore
from .sfs import Sfs, SfsCore

# Define the __all__ variable to specify what is available to import from this package
__all__ = ["AutoGluon", "Boruta", "Efs", "Mrmr", "Octo", "Rfe", "Rfe2", "Roc", "Sfs"]

# Type for the modules inventory dictionary
ModulesInventoryType = dict[str, type]

# Inventory for all available modules
modules_inventory: ModulesInventoryType = {
    "autogluon": AGCore,
    "octo": OctoCore,
    "mrmr": MrmrCore,
    "rfe": RfeCore,
    "rfe2": Rfe2Core,
    "roc": RocCore,
    "sfs": SfsCore,
    "efs": EfsCore,
    "boruta": BorutaCore,
}
