"""Workflow script for the housing."""

import os
import socket
from pathlib import Path

import pandas as pd

from octopus import OctoData
from octopus.dashboard.run import OctoDash

# Conda and Host information
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


path_study = Path("./studies/20240110B")
octo_dashboard = OctoDash(path_study)
octo_dashboard.run()
