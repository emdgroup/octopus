"""Workflow script for the housing."""

import os
import socket

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules.octo.sequence import Octo

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

### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. For this classification approach,
# we also define a stratification column.
octo_data = OctoData(
    data=data_df,
    target_columns=["median_house_value"],
    feature_columns=[
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
    sample_id="index",
    datasplit_type="sample",
)

### Create Configuration

# We create three types of configurations:
# 1. `ConfigStudy`: Sets the name, machine learning type (classification),
# and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use one sequence with the `RandomForestClassifier` model.

config_study = ConfigStudy(
    name="Housing",
    ml_type="regression",
    target_metric="R2",
    metrics=["MSE", "MAE", "R2"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    start_with_empty_study=True,
    path="./studies/",
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=1,
    production_mode=False,
)

config_sequence = ConfigSequence(
    [
        # Step1: octo
        Octo(
            description="step1_octo",
            # datasplit
            n_folds_inner=5,
            datasplit_seed_inner=0,
            # model training
            optuna_seed=5,
            models=["RandomForestRegressor"],
            model_seed=0,
            n_jobs=1,
            dim_red_methods=[""],
            fi_methods_bestbag=["permutation"],
            # max_outl=5,
            # parallelization
            inner_parallelization=False,
            n_workers=5,
            # HPO
            global_hyperparameter=True,
            n_trials=5,
            max_features=70,
            # remove_trials=False,
        ),
        # Step2: ....
        Octo(
            description="step2_octo",
            # datasplit
            n_folds_inner=5,
            datasplit_seed_inner=0,
            # model training
            optuna_seed=5,
            models=["RandomForestRegressor"],
            model_seed=0,
            n_jobs=1,
            dim_red_methods=[""],
            fi_methods_bestbag=["permutation"],
            # max_outl=5,
            # parallelization
            inner_parallelization=False,
            n_workers=5,
            # HPO
            global_hyperparameter=True,
            n_trials=5,
            max_features=70,
            # remove_trials=False,
        ),
    ]
)

### Execute the Machine Learning Workflow

# We add the data and the configurations defined earlier
# and run the machine learning workflow.
octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_sequence=config_sequence,
)
octo_ml.create_outer_experiments()
octo_ml.run_outer_experiments()

print("Workflow completed")

# This completes the basic example for using Octopus Classification
# with the Titanic dataset. The workflow involves loading and preprocessing
# the data, creating necessary configurations, and executing the machine
# learning pipeline.
