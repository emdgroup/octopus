"""Modules inventory."""

try:
    from octopus.modules.autosklearn.autosk import Autosklearn
except ImportError:
    print("Info: Auto-sklearn not available in this conda environment")

from octopus.modules.mrmr.core import MrmrCore
from octopus.modules.octo.core import OctoCore
from octopus.modules.rfe.core import RfeCore
from octopus.modules.roc.core import RocCore
from octopus.modules.sfs.core import SfsCore

# inventory for all available modules
modules_inventory = {
    "autosklearn": Autosklearn,
    "octo": OctoCore,
    "mrmr": MrmrCore,
    "rfe": RfeCore,
    "roc": RocCore,
    "sfs": SfsCore,
}
