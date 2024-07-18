"""Workflow script for test dataset Martin."""

import os
import socket

import attrs

# OPENBLASE config needs to be before pandas, autosk
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd  # noqa E402

from octopus import OctoConfig, OctoData, OctoML  # noqa E402
from octopus.modules.mrmr import MrmrConfig  # noqa E402
from octopus.modules.octo.config import OctopusFullConfig  # noqa E402

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


# load test dataset from Martin from csv and perform pre-processing
# stored in ./datasets_local/ to avoid accidental uploading to github
# pylint: disable=invalid-name
file_path = (
    "./datasets_local/baseline_dataframe_OS6_20230724A_mb_clini_haema"
    "_3random(2-4)_treatmentarm(1)_strat.csv"
)

data = pd.read_csv(file_path, index_col=0)
data.columns = data.columns.astype(str)


features = pd.read_csv(
    "./datasets_local/20221109_compl90_remcorr_trmtarm_3noise.csv", index_col=0
)
features = features["features"].astype(str).tolist()

target_column = ["OS_DURATION_6MONTHS"]
sample_column = "SUBJECT_ID"
stratification_column = ["STRAT_OS6_TRT_NUM"]


# pre-process data
print("Number of samples with target values:", len(data[target_column]))

# define data_input
data_input = {
    "data": data,
    "sample_id": sample_column,
    "target_columns": target_column,
    "datasplit_type": "sample",
    "feature_columns": features,
    "stratification_column": stratification_column,
}

# create OctoData object
data = OctoData(**data_input)

# configure study
config_study = {
    "study_name": "20240322A_MBOS6_octofull_5x5_ETREE",
    "output_path": "./studies/",
    "production_mode": False,
    "start_with_empty_study": False,
    "ml_type": "classification",
    "n_folds_outer": 5,
    "target_metric": "AUCROC",
    "metrics": ["AUCROC"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # outer loop
    "outer_parallelization": False,
    # only run specific single experiment, for quick testing
    # "run_single_experiment_num": 4,
}

# define processing sequence
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # loading of existing results
    load_sequence_item=True,
    # datasplit
    n_folds_inner=5,
    datasplit_seed_inner=0,
    # model training
    models=[
        # "ExtraTreesClassifier",
        # "RandomForestClassifier",
        # "CatBoostClassifier",
        "XGBClassifier",
    ],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    max_outl=0,
    fi_methods_bestbag=["permutation"],
    # parallelization
    inner_parallelization=True,
    n_workers=5,
    # HPO
    optuna_seed=0,
    n_optuna_startup_trials=10,
    resume_optimization=False,
    global_hyperparameter=True,
    n_trials=500,
    max_features=70,
    penalty_factor=1.0,
)


# define processing sequence
sequence_item_2 = MrmrConfig(
    description="step2_mrmr",
    # number of features selected by MRMR
    n_features=50,
    # what correlation type should be used
    correlation_type="rdc",
    # feature importance type (mean/count)
    feature_importance_type="mean",
    # feature importance method (permuation/shap/internal)
    feature_importance_method="permutation",
)


# define processing sequence
sequence_item_3 = OctopusFullConfig(
    description="step2_octofull",
    # loading of existing results
    load_sequence_item=False,
    # datasplit
    n_folds_inner=5,
    datasplit_seed_inner=0,
    # model training
    models=[
        "TabPFNClassifier",
        # "ExtraTreesClassifier",
        # "RandomForestClassifier",
        # "CatBoostClassifier",
        # "XGBClassifier",
    ],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    max_outl=0,
    # fi_methods_bestbag=["permutation"],
    # parallelization
    inner_parallelization=True,
    n_workers=5,
    # HPO
    optuna_seed=0,
    n_optuna_startup_trials=10,
    resume_optimization=False,
    global_hyperparameter=True,
    n_trials=1,
    # max_features=70,
    # penalty_factor=1.0,
)


config_sequence = [
    attrs.asdict(sequence_item_1),
    attrs.asdict(sequence_item_2),
    attrs.asdict(sequence_item_3),
]
# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
