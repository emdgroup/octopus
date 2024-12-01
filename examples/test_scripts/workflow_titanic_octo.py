"""Workflow script for the titanic example."""

import os
import socket

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo

# Conda and Host information
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())

# load data from csv and perform pre-processing
# all features should be numeric (and not bool)
data_df = (
    pd.read_csv(
        os.path.join(os.getcwd(), "datasets", "titanic_openml.csv"), index_col=0
    )
    .astype({"age": float})
    .assign(
        age=lambda df_: df_["age"].fillna(df_["age"].median()).astype(int),
        embarked=lambda df_: df_["embarked"].fillna(df_["embarked"].mode()[0]),
        fare=lambda df_: df_["fare"].fillna(df_["fare"].median()),
    )
    .astype({"survived": bool})
    .pipe(
        lambda df_: df_.reindex(
            columns=["survived"] + list([a for a in df_.columns if a != "survived"])
        )
    )
    .pipe(
        lambda df_: df_.reindex(
            columns=["name"] + list([a for a in df_.columns if a != "name"])
        )
    )
    .pipe(pd.get_dummies, columns=["embarked", "sex"], drop_first=True, dtype=int)
)

### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. For this classification approach,
# we also define a stratification column.
octo_data = OctoData(
    data=data_df,
    target_columns=["survived"],
    feature_columns=[
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked_Q",
        "embarked_S",
        "sex_male",
    ],
    sample_id="name",
    datasplit_type="group_sample_and_features",
    stratification_column="survived",
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
    name="Titanic",
    ml_type="classification",
    target_metric="ACCBAL",
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
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
    run_single_experiment_num=0,
)

config_sequence = ConfigSequence(
    [
        # Step1: octo
        Octo(
            description="step_1_octo",
            n_folds_inner=5,
            # model selection
            models=[
                # "TabPFNClassifier",
                "ExtraTreesClassifier",
                # "RandomForestClassifier",
                # "CatBoostClassifier",
                # "XGBClassifier",
                # "LogisticRegressionClassifier",
            ],
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            global_hyperparameter=False,
            n_trials=30,
            # ensemble selection
            ensemble_selection=False,
            ensel_n_save_trials=50,
            # mrmr
            # mrmr_feature_numbers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        # Step2: ....
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
