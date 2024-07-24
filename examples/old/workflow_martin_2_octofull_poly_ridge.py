"""Workflow script for test dataset Martin."""

import os
import socket

import attrs

# OPENBLASE config needs to be before pandas, autosk
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from octopus import OctoConfig, OctoData, OctoML
from octopus.modules.octo.sequence import OctopusFullConfig

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


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


data_input = {
    "data": data_final,
    "sample_id": id_data[0],
    "target_columns": ls_targets,
    "datasplit_type": "sample",
    "feature_columns": ls_final,
}


# create OctoData object
data = OctoData(**data_input)

# configure study
config_study = {
    "study_name": "20240214F_Martin_wf2_octofull_7x6_poly_global_ridge",
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
    # only run specific single experiment, for quick testing
    # "run_single_experiment_num": 0,
}

# define processing sequence
sequence_item_1 = OctopusFullConfig(
    description="step1_octofull",
    # datasplit
    n_folds_inner=6,
    datasplit_seed_inner=0,
    # model training
    models=["RidgeRegressor"],
    model_seed=0,
    n_jobs=1,
    dim_red_methods=[""],
    max_outl=0,
    # parallelization
    inner_parallelization=True,
    n_workers=6,
    # HPO
    optuna_seed=0,
    n_optuna_startup_trials=10,
    resume_optimization=False,
    global_hyperparameter=True,
    n_trials=50,
    max_features=70,
)

config_sequence = [attrs.asdict(sequence_item_1)]
# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
print("Workflow completed")
