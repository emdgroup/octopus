"""Modules inventory."""

try:
    from octopus.modules.autosk import Autosklearn
except ImportError:
    print("Info: Auto-sklearn not available in this conda environment")

from octopus.modules.mrmr import Mrmr
from octopus.modules.octo.octofull import OctoFull
from octopus.modules.rfe import RfeModule
from octopus.modules.roc import RocModule
from octopus.modules.sfs import SfsModule

# inventory for all available modules
modules_inventory = {
    "autosklearn": Autosklearn,
    "octofull": OctoFull,
    "mrmr": Mrmr,
    "rfe": RfeModule,
    "sfs": SfsModule,
    "roc": RocModule,
}
