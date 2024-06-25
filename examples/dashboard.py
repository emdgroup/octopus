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


# California housing dataset
# load data from csv and perform pre-processing
data_df = (
    pd.read_csv(os.path.join(os.getcwd(), "datasets", "california_housing_prices.csv"))
    .reset_index()
    .astype(
        {
            "housing_median_age": int,
            "total_rooms": int,
            "population": int,
            "households": int,
            "median_income": int,
            "median_house_value": int,
        }
    )
    .loc[0:100, :]
)

# define input for Octodata
data_input = {
    "data": data_df,
    "sample_id": "index",
    "row_id": "index",
    "target_columns": ["median_house_value"],
    "datasplit_type": "sample",
    "feature_columns": [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        # "total_bedrooms",
        "population",
        "households",
        "median_income",
        # "ocean_proximity": str,
    ],
}


# create OctoData object
data = OctoData(**data_input)

# eda = OctoEDA(data)
# eda.run()


folder_path = Path("studies").joinpath("classification_2")

octo_dashboard = OctoDash(folder_path)
octo_dashboard.run()
