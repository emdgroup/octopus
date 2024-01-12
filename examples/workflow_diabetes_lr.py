"""Workflow script for the diabetes regression example."""
import os

import pandas as pd

from octopus import OctoConfig, OctoData, OctoML

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
    "target_columns": {"progression": float},
    "datasplit_type": "sample",
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

# configure study
config_study = {
    "study_name": "20231220B",
    "output_path": "./studies/",
    "ml_type": "regression",  # prediction_task
    "k_outer": 5,
    "target_metric": "MAE",  # only metric?
    "metrics": ["MSE", "MAE", "R2"],  # what is this for? calculate these metrics
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    "ml_execution": "sequential",  # computation / mode?
    # only process first outer loop experiment, for quick testing
    # just "first_only"???
    "ml_only_first": True,
}

config_sequence = [
    {
        "ml_module": "linear_regression_ave",  # module
        "description": "step1_autosklearn",
        "config": {
            "optuna_trails": 50,
            "alpha": [0.001, 1],
            "fit_intercept": [True, False],
        },
    }
]


# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("done")
print("done")
