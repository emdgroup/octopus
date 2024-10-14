# __init__.py
"""Init modules."""

from typing import Dict, Type

from .boruta.core import BorutaCore
from .boruta.module import Boruta
from .efs.core import EfsCore
from .efs.module import Efs

# Import the core classes from their respective modules
from .mrmr.core import MrmrCore
from .mrmr.module import Mrmr
from .octo.core import OctoCore
from .octo.module import Octo
from .rfe.core import RfeCore
from .rfe.module import Rfe
from .rfe2.core import Rfe2Core
from .rfe2.module import Rfe2
from .roc.core import RocCore
from .roc.module import Roc
from .sfs.core import SfsCore
from .sfs.module import Sfs

# Define the __all__ variable to specify what is available to import from this package
__all__ = ["Mrmr", "Octo", "Rfe", "Rfe2", "Roc", "Sfs", "Efs", "Boruta"]

# Type for the modules inventory dictionary
ModulesInventoryType = Dict[str, Type]

# Inventory for all available modules
modules_inventory: ModulesInventoryType = {
    "octo": OctoCore,
    "mrmr": MrmrCore,
    "rfe": RfeCore,
    "rfe2": Rfe2Core,
    "roc": RocCore,
    "sfs": SfsCore,
    "efs": EfsCore,
    "boruta": BorutaCore,
}
