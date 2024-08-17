"""Workflow script for the diabetes regression example."""

import os
import socket

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Mrmr, Octo

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


### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. For this classification approach,
# we also define a stratification column.
octo_data = OctoData(
    data=data_df,
    disable_checknan=True,
    target_columns=["progression"],
    feature_columns=[
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ],
    sample_id="patient_id",
    datasplit_type="group_sample_and_features",
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
    name="Diabetes",
    ml_type="regression",
    target_metric="R2",
    metrics=["MSE", "MAE", "R2"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    start_with_empty_study=True,
    path="./studies/",
    silently_overwrite_study=True,
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=1,
)

config_sequence = ConfigSequence(
    [
        # Step1: octo
        Octo(
            description="step1_octofull",
            # datasplit
            n_folds_inner=5,
            datasplit_seed_inner=0,
            load_sequence_item=False,
            # model training
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            model_seed=0,
            n_jobs=1,
            dim_red_methods=[""],
            max_outl=5,
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            global_hyperparameter=True,
            n_trials=5,
            max_features=70,
        ),
        # Step2: MRMR
        Mrmr(
            description="step2_mrmr",
            # number of features selected by MRMR
            n_features=6,
            # what correlation type should be used
            correlation_type="rdc",
            # feature importance type (mean/count)
            feature_importance_type="mean",
            # feature importance method (permuation/shap/internal)
            feature_importance_method="permutation",
        ),
        # Step3: octo
        Octo(
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
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            global_hyperparameter=True,
            n_trials=10,
            max_features=70,
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

# This completes the basic example for using Octopus Regression
# with the Diabetes dataset. The workflow involves loading and preprocessing
# the data, creating necessary configurations, and executing the machine
# learning pipeline.
