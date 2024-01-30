"""Workflow script for the housing."""
import os
import socket

import attrs
import pandas as pd

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.octofull import OctopusFullConfig

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
    # .loc[0:100, :]
)

# define input for Octodata
data_input = {
    "data": data_df.iloc[:600, :],
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
    "study_name": "housing_octofull_test",
    "output_path": "./studies/",
    "production_mode": False,
    "ml_type": "regression",
    "k_outer": 5,
    "target_metric": "R2",
    "metrics": ["MSE", "MAE", "R2"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # ['parallel', 'sequential']-type of execution of outer loop experiments
    "outer_parallelization": True,
    # only process first outer loop experiment, for quick testing
    "ml_only_first": True,
}

config_sequence = [
    {
        "ml_module": "octofull",
        "description": "step1_octofull",
        "config": {
            "datasplit_seed_inner": 0,  # data split seed for inner loops
            "k_inner": 5,  # number of inner folds
            "ml_jobs": 1,  # number of parallel jobs for ML
            "ml_seed": 0,  # seed to make ML algo deterministic
            "dim_red_methods": [""],  # Optuna
            "ml_model_types": [
                "ExtraTreesRegressor",
                "RandomForestRegressor",
                # "XGBRegressor",
            ],  # Optuna
            "max_outl": 5,  # Optuna
            # ["parallel","sequential"]
            "execution_type": "parallel",
            "num_workers": 5,
            "HPO_type": "global",  # ["global","individual"]
            "HPO_max_features": 70,
            "HPO_remove_trials": False,
            "HPO_trials": 5,  # number of HPO trial
        },
    },
]


sequence_1 = OctopusFullConfig(
    models=["ExtraTreesRegressor", "RandomForestRegressor"],
    description="step1_octofull",
    global_hyperparameter=True,
    max_features=70,
    inner_parallelization=True,
    n_trials=5,
)

config_sequence = [attrs.asdict(sequence_1)]

# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
