"""Workflow script T2E analysis using German Breast Cancer Study group 2."""

import os
import socket

import attrs

# OPENBLASE config needs to be before pandas, autosk
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.octo.config import OctopusFullConfig

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


# load test dataset from Martin from csv and perform pre-processing
# stored in ./datasets_local/ to avoid accidental uploading to github
data = pd.read_csv("./datasets/gbs2.csv", index_col=0)
# data pre-processing
# check for NaNs
assert not pd.isna(data).any().any()

# one-hot encoding of categorical columns
columns = ["horTh", "menostat", "tgrade"]
df_list = [data]
for column in columns:
    df_list.append(pd.get_dummies(data[column], prefix=column, drop_first=True))
data_processed = pd.concat(df_list, axis=1)
for column in columns:
    data_processed.drop(column, axis=1, inplace=True)

# convert boolean columns to int
# Find the boolean columns
boolean_columns = data_processed.select_dtypes(include=bool).columns

# Convert boolean columns to int
data_processed[boolean_columns] = data_processed[boolean_columns].astype(int)
# create patient ID
data_processed.reset_index(inplace=True)
data_processed.rename(columns={"index": "patient"}, inplace=True)


data_input = {
    "data": data_processed,
    "sample_id": "patient",
    "target_columns": {
        "Event": int,
        "Duration": float,
    },
    "target_asignments": {"event": "Event", "duration": "Duration"},
    "datasplit_type": "sample",
    "feature_columns": {
        "age": float,
        "estrec": float,
        "pnodes": float,
        "progrec": float,
        "tsize": float,
        "horTh_yes": int,
        "menostat_Pre": int,
        "tgrade_II": int,
        "tgrade_III": int,
    },
}


# create OctoData object
data = OctoData(**data_input)

# configure study
config_study = {
    "study_name": "20240226A_survival_octofull_5x5_global_xtratree",
    "output_path": "./studies/",
    "production_mode": False,
    "ml_type": "timetoevent",
    "n_folds_outer": 5,
    "target_metric": "CI",
    "metrics": ["CI"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # outer loop
    "outer_parallelization": True,
    # only run specific single experiment, for quick testing
    "run_single_experiment_num": 0,
}

# define processing sequence
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # datasplit
    n_folds_inner=5,
    datasplit_seed_inner=0,
    # model training
    models=["ExtraTreesSurv"],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    max_outl=0,
    # parallelization
    inner_parallelization=False,
    n_workers=5,
    # HPO
    optuna_seed=0,
    n_optuna_startup_trials=10,
    resume_optimization=False,
    global_hyperparameter=True,
    n_trials=10,
    save_trials=False,
    max_features=0,
    penalty_factor=10.0,
)

config_sequence = [attrs.asdict(sequence_item_1)]
# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
