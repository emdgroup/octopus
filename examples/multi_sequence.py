## Example for using a multi sequence

# TBD

### Necessary imports for this example

import os

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules.mrmr.sequence import Mrmr
from octopus.modules.octo.sequence import Octo

# Regression Analysis on Diabetes Dataset
# http://statweb.lsu.edu/faculty/li/teach/exst7142/diabetes.html
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
# https://automl.github.io/auto-sklearn/master/examples/20_basic/example_regression.html

# load data from csv and perform pre-processing
data_df = pd.read_csv(
    os.path.join(os.getcwd(), "datasets", "diabetes.csv"), index_col=0
)

### Create OctoData Object

# TBD
octo_data = OctoData(
    data=data_df,
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

# TDB

config_study = ConfigStudy(
    name="example_multisequence",
    ml_type="regression",
    target_metric="R2",
    overwrite_existing_study=True,
)

config_manager = ConfigManager(
    outer_parallelization=False,
    run_single_experiment_num=1,
)

config_sequence = ConfigSequence(
    [
        Octo(
            description="step1_octofull",
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=1,
            max_features=70,
        ),
        Mrmr(
            description="step2_mrmr",
            n_features=6,
            correlation_type="rdc",
        ),
        Octo(
            description="step1_octofull",
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=1,
            max_features=70,
        ),
    ]
)

### Execute the Machine Learning Workflow

octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_sequence=config_sequence,
)
octo_ml.create_outer_experiments()
octo_ml.run_outer_experiments()
