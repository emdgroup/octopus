"""Workflow script for the titanic example."""
import os
import socket

import pandas as pd

from octopus import OctoConfig, OctoData, OctoML

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
    "target_columns": {"survived": bool},
    "stratification_column": {"survived": bool},
    "datasplit_type": "group_sample_and_features",
    "feature_columns": {
        "pclass": int,
        "age": int,
        "sibsp": int,
        "parch": int,
        "fare": float,
        "embarked_Q": int,
        "embarked_S": int,
        "sex_male": int,
    },
}

# create OctoData object
data = OctoData(**data_input)


# define inputs for OctoConfig
# configure study
config_study = {
    # OctoML
    "study_name": "20240110B",
    "output_path": "./studies/",
    "production_mode": False,
    "ml_type": "classification",  # ['classification','regression','timetoevent']
    "k_outer": 5,
    "target_metric": "AUCROC",
    "metrics": ["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    "datasplit_seed_outer": 1234,
}

# configure manager
config_manager = {
    "ml_execution": "parallel",  # ['parallel', 'sequential']
    # only process first outer loop experiment, for quick testing
    "ml_only_first": True,
}

# configure modules and model sequences
config_sequence = [
    {
        "ml_module": "autosklearn",
        "description": "step1_autosklearn",
        "config": {
            "time_left_for_this_task": 1 * 60,
            "per_run_time_limit": 30,
            "include": {
                "classifier": ["extra_trees"],  # ["decision_tree", "lda", "sgd"]
                "feature_preprocessor": ["no_preprocessing"],
                # ["no_preprocessing","polynomial","select_percentile_classification"],
            },
            "ensemble_kwargs": {"ensemble_size": 1},
            "initial_configurations_via_metalearning": 0,
            #'resampling_strategy':'holdout',
            #'resampling_strategy_arguments':None,
            "resampling_strategy": "cv",
            "resampling_strategy_arguments": {"folds": 5},
        },
    },
    #            {'ml_module':'autosklearn',
    #            'description':'step2_autosklearn',
    #            'config':{
    #                'time_left_for_this_task':1*60,
    #                'per_run_time_limit':30,
    #                'include':{
    #                    "classifier": ["extra_trees"], #["decision_tree", "lda", "sgd"]
    #                    "feature_preprocessor":["no_preprocessing"],
    #                'ensemble_kwargs':{"ensemble_size": 1},
    #                'initial_configurations_via_metalearning':0,
    #                #'resampling_strategy':'holdout',
    #                #'resampling_strategy_arguments':None,
    #                'resampling_strategy':'cv',
    #                'resampling_strategy_arguments':{'folds': 5},
    #                }
    #            } ,
    #    'octopusfull':{
    #            'description':'Step2_octopus_full',
    #            'ml_module':'octopus_full',
    #            'datasplit_seed_inner':0, # data split seed for inner loops
    #            'ml_seed':0, # seed to make ML algo deterministic
    #            'dim_red_methods':['pca', 'ica'], # ['pca','ica','wgcna', etc..]
    #            'ml_model_types:':['extratree', 'xgboost'],
    #            'num_outl':0,
    #            'n_jobs':4, # number of parallel jobs for ML
    #            'class_weights':False,
    #            'HPO_method':'Optuna_050722A',  # allow for
    # different HPO methods and versions
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
