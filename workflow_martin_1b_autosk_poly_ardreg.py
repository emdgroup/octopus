"""Workflow script for test dataset Martin."""

import os
import socket

# OPENBLASE config needs to be before pandas, autosk
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd  # noqa: E402  # pylint: disable=C0413
from sklearn.preprocessing import (  # noqa: E402  # pylint: disable=C0413
    PolynomialFeatures,
)

from octopus import OctoConfig, OctoData, OctoML  # noqa: E402  # pylint: disable=C0413

# Conda and Host information
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())
# show directory name


# load test dataset from Martin from csv and perform pre-processing
# stored in ./datasets_local/ to avoid accidental uploading to github
data = pd.read_csv("./datasets_local/df_data_lights.csv", index_col=0)

# definitions given by Martin
ls_numbers = [
    "NumAtoms",
    "NumHeavyAtoms",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAmideBonds",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumBridgeheadAtoms",
    "NumHBA",
    "NumHBD",
    "NumHeteroatoms",
    "NumHeterocycles",
    "NumRotatableBonds",
    "NumRings",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumSpiroAtoms",
    "FractionCSP3",
]
ls_props = ["molwt"]
ls_graph = ["BalabanJ", "BertzCT"]
ls_morgan_fp = [f"morgan_{i}" for i in range(3 * 1024)]
ls_rd_fp = [f"rdfp_{i}" for i in range(2048)]
ls_features = ls_numbers + ls_props + ls_graph + ls_morgan_fp + ls_rd_fp
ls_targets = ["T_SETARAM"]
id_data = ["MATERIAL_ID"]

# pre-process data
# there are NaNs in the target column
target_column = data[ls_targets[0]]
non_nan_targets = ~pd.isna(target_column)  # pylint: disable=E1130
target_column = target_column[non_nan_targets]
print("Number of samples with target values:", len(target_column))
data_reduced = data[non_nan_targets].reset_index(drop=True)
# check for NaNs
data_relevant = data_reduced[ls_features + ls_targets + id_data]
assert not pd.isna(data_relevant).any().any()

# add polynomial features
data_poly_input = data_reduced[ls_numbers + ls_props + ls_graph]
poly = PolynomialFeatures(degree=2, interaction_only=True)
data_poly = pd.DataFrame(poly.fit_transform(data_poly_input))
data_poly.columns = data_poly.columns.astype(str)  # column names must be string
ls_poly = data_poly.columns.tolist()
data_final = pd.concat(
    [data_poly, data_reduced[ls_morgan_fp + ls_rd_fp + ls_targets + id_data]], axis=1
)
ls_final = ls_poly + ls_morgan_fp + ls_rd_fp


# reduce constant features
def find_constant_columns(df):
    """Find constant columns."""
    constant_columns = []
    for column in df.columns:
        if df[column].nunique() == 1:
            constant_columns.append(column)
    return constant_columns


ls_features_const = find_constant_columns(data_final)
# Remove constant columns from other_cols list
print("Number of original features:", len(ls_final))
ls_final = [col for col in ls_final if col not in ls_features_const]
print("Number of features after removal of const. features:", len(ls_final))


# define data_input, use data_reduced
data_input = {
    "data": data_final,
    "sample_id": id_data[0],
    "target_columns": {ls_targets[0]: data_final[ls_targets[0]].dtype},
    "datasplit_type": "sample",
    "feature_columns": dict(),
}

# for feature in ls_features:
#    data_input["feature_columns"][feature] = data_reduced[feature].dtype
for feature in ls_final:
    data_input["feature_columns"][feature] = data_final[feature].dtype


# create OctoData object
data = OctoData(**data_input)

# configure study
config_study = {
    "study_name": "20240215F_Martin_wf1_poly_autosk_ardreg_all_1h",
    "output_path": "./studies/",
    "production_mode": False,
    "ml_type": "regression",
    "n_folds_outer": 7,
    "target_metric": "MAE",
    "metrics": ["MSE", "MAE", "R2"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    # outer loop
    "outer_parallelization": True,
    # only process first outer loop experiment, for quick testing
    # "run_single_experiment_num": 3,
}


# define processing sequence
config_sequence = [
    {
        "module": "autosklearn",
        "description": "step1_autosklearn",
        "config": {
            "time_left_for_this_task": 60 * 60,
            "per_run_time_limit": 12 * 60,
            "n_jobs": 1,
            "include": {
                # regressor:[
                # "adaboost",
                # "ard_regression",
                # "decision_tree",
                # "extra_trees",
                # "gaussian_process",
                # "gradient_boosting",
                # "k_nearest_neighbors",
                # "libsvm_svr",
                # "mlp",
                # "random_forest",]
                "regressor": ["ard_regression"],
                # ["no_preprocessing","polynomial","select_percentile_classification"],
                "feature_preprocessor": [
                    "no_preprocessing"
                ],  # no feature preprocessing
            },
            "ensemble_kwargs": {"ensemble_size": 1},  # no ensembling
            # "memory_limit": 6144,
            "initial_configurations_via_metalearning": 0,  # no meta learning
            # 'resampling_strategy':'holdout',
            # 'resampling_strategy_arguments':None,
            "resampling_strategy": "cv",
            "resampling_strategy_arguments": {"folds": 6},
        },
    },
]

# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
