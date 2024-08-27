# __init__.py
"""Init modules."""

from typing import Dict, Type

# Import the core classes from their respective modules
from .mrmr.core import MrmrCore

# Import the classes from their respective modules
from .mrmr.module import Mrmr
from .octo.core import OctoCore
from .octo.module import Octo
from .rfe.core import RfeCore
from .rfe.module import Rfe
from .roc.core import RocCore
from .roc.module import Roc
from .sfs.core import SfsCore
from .sfs.module import Sfs
from .efs.core import EfsCore
from .efs.module import Efs

# Define the __all__ variable to specify what is available to import from this package
__all__ = ["Mrmr", "Octo", "Rfe", "Roc", "Sfs", "Efs"]

# Type for the modules inventory dictionary
ModulesInventoryType = Dict[str, Type]

# Inventory for all available modules
modules_inventory: ModulesInventoryType = {
    "octo": OctoCore,
    "mrmr": MrmrCore,
    "rfe": RfeCore,
    "roc": RocCore,
    "sfs": SfsCore,
    "efs": EfsCore,
}
