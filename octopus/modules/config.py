"""Modules inventory."""

try:
    from octopus.modules.autosk import Autosklearn
except ImportError:
    print("Info: Auto-sklearn not available in this conda environment")

from octopus.modules.mrmr.core import MrmrCore
from octopus.modules.octo.core import OctoCore

# inventory for all available modules
modules_inventory = {
    "autosklearn": Autosklearn,
    "octo": OctoCore,
    "mrmr": MrmrCore,
}
