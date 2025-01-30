"""Octopus - AutoML for small Datasets."""

import sys

from octopus.dashboard.run import OctoDash  # noqa: F401
from octopus.data.config import OctoData  # noqa: F401

# from octopus.experiment import OctoExperiment  # noqa: F401
# from octopus.manager import OctoManager  # noqa: F401
from octopus.ml import OctoML  # noqa: F401

if not sys.version_info >= (3, 12):
    raise ValueError("Minimum required Python version is 3.12")
