"""Workflow script for the diabetes regression example."""
import os
import socket

import pandas as pd

from octopus import OctoConfig, OctoData, OctoML

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


# define input for Octodata
data_input = {
    "data": data_df,
    "sample_id": "patient_id",  # sample_id may contain duplicates
    "row_id": "patient_id",  # must be unique!
    # ['sample','group_sample', 'group_sample_and_features']
    "datasplit_type": "group_sample_and_features",
    "disable_checknan": True,
    "target_columns": {"progression": float},
    "feature_columns": {
        "age": float,
        "sex": float,
        "bmi": float,
        "bp": float,
        "s1": float,
        "s2": float,
        "s3": float,
        "s4": float,
        "s5": float,
        "s6": float,
    },
}

# create OctoData object
data = OctoData(**data_input)


# define inputs for OctoConfig
# configure study
config_study = {
    # OctoML
    "study_name": "20231221B",
    "output_path": "./studies/",
    "production_mode": False,
    # ['classification','regression','timetoevent']
    "ml_type": "regression",
    "n_folds_outer": 5,
    "target_metric": "MAE",
    "metrics": ["MSE", "MAE", "R2"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    "outer_parallelization": True,
    # only process first outer loop experiment, for quick testing
    "ml_only_first": False,
}

# configure modules and model sequences
# https://www.kaggle.com/code/rupakroy/auto-sklearn-regression
# https://github.com/automl/auto-sklearn/tree/development/autosklearn/pipeline/components/regression

config_sequence = [
    {
        "module": "autosklearn",
        "description": "step1_autosklearn",
        "config": {
            "time_left_for_this_task": 1 * 60,
            "per_run_time_limit": 30,
            "include": {
                # ["decision_tree", "lda", "sgd"]
                "regressor": ["ard_regression"],
                # ["no_preprocessing","polynomial","select_percentile_classification"],
                "feature_preprocessor": ["no_preprocessing"],
            },
            "ensemble_kwargs": {"ensemble_size": 1},
            "initial_configurations_via_metalearning": 0,
            # 'resampling_strategy':'holdout',
            # 'resampling_strategy_arguments':None,
            "resampling_strategy": "cv",
            "resampling_strategy_arguments": {"folds": 5},
        },
    },
    #    'octopusfull':{
    #            'description':'Step2_octopus_full',
    #            'ml_module':'octopus_full',
    #            'datasplit_seed_inner':0, # data split seed for inner loops
    #            'ml_seed':0, # seed to make ML algo deterministic
    #            'dim_red_methods':['pca', 'ica'], # ['pca','ica','wgcna', etc..]
    #             dimension reduction methods
    #            'ml_model_types:':['extratree', 'xgboost'],
    #            'num_outl':0, # outlier detection in training dataset,
    #                            could be part of HPO
    #            'n_jobs':4, # number of parallel jobs for ML
    #            # whether to use class weights, if yes class weights are part of HPO
    #            'class_weights':False,
    #            'HPO_method':'Optuna_050722A',  # allow for different
    # HPO methods and versions
    #            'HOP_trials':50, # number of HPO trial
    #            },
]


# create study config
octo_config = OctoConfig(config_manager, config_sequence, **config_study)

# create ML object
oml = OctoML(data, octo_config)
# print(oml)

oml.create_outer_experiments()

oml.run_outer_experiments()

print("Workflow completed")
