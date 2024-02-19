"""Workflow script for the diabetes regression example."""

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

# Regression Analysis on Diabetes Dataset
# http://statweb.lsu.edu/faculty/li/teach/exst7142/diabetes.html
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
# https://automl.github.io/auto-sklearn/master/examples/20_basic/example_regression.html

# load data from csv and perform pre-processing
data_df = pd.read_csv(
    os.path.join(os.getcwd(), "datasets", "diabetes.csv"), index_col=0
)


# define input for Octodata
data_input = {
    "data": data_df,
    "sample_id": "patient_id",  # sample_id may contain duplicates
    "row_id": "patient_id",  # must be unique!
    # ['sample','group_sample', 'group_sample_and_features']
    "datasplit_type": "group_sample_and_features",
    "disable_checknan": True,
    "target_columns": {"progression": float},
    "feature_columns": {
        "age": float,
        "sex": float,
        "bmi": float,
        "bp": float,
        "s1": float,
        "s2": float,
        "s3": float,
        "s4": float,
        "s5": float,
        "s6": float,
    },
}

# create OctoData object
data = OctoData(**data_input)


# define inputs for OctoConfig
# configure study
config_study = {
    # OctoML
    "study_name": "20231221B",
    "output_path": "./studies/",
    "production_mode": False,
    # ['classification','regression','timetoevent']
    "ml_type": "regression",
    "n_folds_outer": 5,
    "target_metric": "R2",
    "metrics": ["MSE", "MAE", "R2"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # outer loop parallelization
    "outer_parallelization": True,
    # only process first outer loop experiment, for quick testing
    "ml_only_first": True,
}

# define processing sequence
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # datasplit
    n_folds_inner=5,
    datasplit_seed_inner=0,
    # model training
    models=["ExtraTreesRegressor", "RandomForestRegressor"],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    max_outl=5,
    # parallelization
    inner_parallelization=True,
    n_workers=5,
    # HPO
    global_hyperparameter=True,
    n_trials=5,
    max_features=70,
    remove_trials=False,
)

config_sequence = [attrs.asdict(sequence_item_1)]


# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)
# print(oml)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
