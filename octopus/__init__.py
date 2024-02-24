"""Octopus - AutoML for small Datasets."""

import sys

from octopus.analytics.run import OctoAnalitics  # noqa: F401
from octopus.config import OctoConfig  # noqa: F401
from octopus.data import OctoData  # noqa: F401

# from octopus.experiment import OctoExperiment  # noqa: F401
# from octopus.manager import OctoManager  # noqa: F401
from octopus.ml import OctoML  # noqa: F401

if not sys.version_info >= (3, 8):
    raise ValueError("Minimum required Python version is 3.8")
    # Python 3.8 is only required for Auto-Sklearn, otherwise
    # Python 3.10 would be a suitable choice as it is available
    # in Uptimze and is supported till 2026
