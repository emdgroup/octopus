"""Workflow script for the titanic example."""

import os
import socket

import attrs
import pandas as pd

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.rfe.rfe import Rfe

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
data = OctoData(**data_input)


# define inputs for OctoConfig
# configure study
config_study = {
    # OctoML
    "study_name": "Titanic-RFE",
    "output_path": "./studies/",
    "production_mode": False,
    "ml_type": "classification",  # ['classification','regression','timetoevent']
    "n_folds_outer": 5,
    "target_metric": "AUCROC",
    "metrics": ["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # outer loop parallelization
    "outer_parallelization": True,
    # only process first outer loop experiment, for quick testing
    "run_single_experiment_num": 1,
}

# define processing sequence
sequence_item_1 = Rfe(
    description="RFE",
    cv=5,  # number of CV folds
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

# use dashboard
# path_study = Path(config_study["output_path"]).joinpath(config_study["study_name"])
# octo_dashboard = OctoDash(path_study)
# octo_dashboard.run()
