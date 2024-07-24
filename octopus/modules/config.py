"""Modules inventory."""

try:
    from octopus.modules.autosk import Autosklearn
except ImportError:
    print("Info: Auto-sklearn not available in this conda environment")

from octopus.modules.mrmr.module import MrmrModule
from octopus.modules.octo.octofull import OctoFull

# inventory for all available modules
modules_inventory = {
    "autosklearn": Autosklearn,
    "octofull": OctoFull,
    "mrmr": MrmrModule,
}
