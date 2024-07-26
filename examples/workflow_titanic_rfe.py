"""Workflow script for the titanic example."""

import os
import socket

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Sfs

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

# define input for Octodata
# all features need to be numeric
data_input = {
    "data": data_df,
    "sample_id": "name",  # sample_id may contain duplicates
    "target_columns": ["survived"],
    "stratification_column": ["survived"],
    "datasplit_type": "group_sample_and_features",
    "feature_columns": [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked_Q",
        "embarked_S",
        "sex_male",
    ],
}

# create OctoData object
octo_data = OctoData(**data_input)


# define inputs for OctoConfig
# configure study
config_study = ConfigStudy(
    name="Sfs",
    ml_type="classification",
    target_metric="AUCROC",
)

# configure manager
config_manager = ConfigManager(outer_parallelization=False, run_single_experiment_num=0)


# define processing sequence
config_sequence = ConfigSequence([Sfs(description="Sfs", cv=5)])

octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_sequence=config_sequence,
)
octo_ml.create_outer_experiments()

octo_ml.run_outer_experiments()

print("Workflow completed")
