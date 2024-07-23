"""Workflow script for the housing."""

import os
import socket
from pathlib import Path

from octopus.dashboard.run import OctoDash

# Conda and Host information
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


path_study = Path("./studies/20240322A_MBOS6_octofull_5x5_ETREE")
octo_dashboard = OctoDash(path_study)
octo_dashboard.run()
