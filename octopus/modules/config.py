"""Modules inventory."""

try:
    from octopus.modules.autosk import Autosklearn
except ImportError:
    print("Auto-Sklearn not installed in this conda environment")

from octopus.modules.linear_regression import LinearRegressionAve, LinearRegressionUni
from octopus.modules.octo.octofull import OctoFull

# inventory for all available modules
modules_inventory = {
    "autosklearn": Autosklearn,
    "octofull": OctoFull,
    "linear_regression_ave": LinearRegressionAve,
    "linear_regression_uni": LinearRegressionUni,
}
