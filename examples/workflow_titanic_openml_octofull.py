"""Workflow script for the titanic example."""

import os
import socket

import attrs
import numpy as np
import pandas as pd

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.octo.config import OctopusFullConfig

# Conda and Host information
print()
print("########### General Info ##########")
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())
print()
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

my_list = ["test"] + ["G"] * 1308
data_df["abc"] = 1

# define input for Octodata
# all features need to be numeric
data_input = {
    "data": data_df,
    "sample_id": "age",  # sample_id may contain duplicates
    "target_columns": ["age"],
    "stratification_column": ["survived"],
    "datasplit_type": "sample",
    "feature_columns": [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked_Q",
        "embarked_S",
        "sex_male",
        "age",
        "abc",
        "body",
    ],
}
data = OctoData(**data_input)

print()

df_data = pd.DataFrame(
    {
        "target_0": [0, 1, 0, 1],
        "target_1": [5, 1, 5, 1],
        "target_2": [5, 1, 5, 1],
        "feature_0": [1, 2, 3, 4],
        "feature_1": [5, 2, 1, 4],
        "feature_2": [5, 3, 3, 3],
        "feature_3": [1, 2, 3, 3],
        "feature_nan": [5, 6, 7, np.nan],
        "feature_inf": [5, 5, 5, np.inf],
        "feature_str": [5, 6, 7, "1"],
        "feature_bool": [5, 6, 7, True],
        "sample_id": [0, 1, 2, 2],
        "sample_id_unique": [0, 1, 2, 3],
        "id": [10, 11, 12, 13],
    }
)
# create OctoData object
# data = OctoData(**data_input)
octo_data = OctoData(
    data=df_data,
    target_columns=["target_0"],
    feature_columns=["feature_str", "feature_bool"],
    sample_id="sample_id",
    datasplit_type="sample",
    data_quality_check=True,
)

# define inputs for OctoConfig
# configure study
config_study = {
    # OctoML
    "study_name": "20240110B",
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
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # datasplit
    n_folds_inner=5,
    # model training
    models=[
        # "TabPFNClassifier",
        "ExtraTreesClassifier",
        # "RandomForestClassifier",
        # "CatBoostClassifier",
        # "XGBClassifier",
    ],
    # parallelization
    inner_parallelization=True,
    n_workers=5,
    # HPO
    global_hyperparameter=True,
    n_trials=1,
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
