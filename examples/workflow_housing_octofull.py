"""Workflow script for the housing."""

import os
import socket

import attrs
import pandas as pd

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.octo.config import OctopusFullConfig

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
    "target_columns": {"median_house_value": int},
    "datasplit_type": "sample",
    "feature_columns": {
        "longitude": float,
        "latitude": float,
        "housing_median_age": int,
        "total_rooms": int,
        # "total_bedrooms": int,
        "population": int,
        "households": int,
        "median_income": int,
        # "ocean_proximity": str,
    },
}


# create OctoData object
data = OctoData(**data_input)

# configure study
config_study = {
    "study_name": "housing_octofull_test_5",
    "output_path": "./studies/",
    "production_mode": True,
    "ml_type": "regression",
    "n_folds_outer": 5,
    "target_metric": "R2",
    "metrics": ["MSE", "MAE", "R2"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    "outer_parallelization": True,
}

# define processing sequence
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # datasplit
    n_folds_inner=5,
    datasplit_seed_inner=0,
    # model training
    optuna_seed=5,
    models=["RidgeRegressor"],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    # max_outl=5,
    # parallelization
    inner_parallelization=True,
    n_workers=5,
    # HPO
    global_hyperparameter=True,
    n_trials=50,
    # max_features=70,
    # remove_trials=False,
)

config_sequence = [attrs.asdict(sequence_item_1)]

print(config_sequence)

# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
