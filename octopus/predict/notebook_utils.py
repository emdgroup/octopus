"""Utility functions for Jupyter/Marimo notebooks."""

import json
import re
from pathlib import Path

import pandas as pd

from octopus.experiment import OctoExperiment


def setup_notebook():
    """Setup notebook environment with required imports and configurations.

    This function configures pandas display options and returns commonly used
    objects for notebook analysis workflows.

    Returns:
        tuple: Common objects needed for notebook operations:
            - OctoExperiment: Experiment class for loading and analyzing experiments
            - Path: pathlib.Path class for file path operations
            - json: json module for reading configuration files
            - pd: pandas module for data manipulation
            - re: re module for regular expressions

    Example:
        >>> from octopus.predict.notebook_utils import setup_notebook
        >>> OctoExperiment, Path, json, pd, re = setup_notebook()
    """
    # Configure pandas display options
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    return OctoExperiment, Path, json, pd, re
