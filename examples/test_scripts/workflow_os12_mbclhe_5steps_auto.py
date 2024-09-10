"""Workflow script for test dataset Martin."""

import os
import socket

# OPENBLASE config needs to be before pandas, autosk
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Mrmr, Octo, Rfe, Roc

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


# load test dataset from Martin from csv and perform pre-processing
# stored in ./datasets_local/ to avoid accidental uploading to github
# pylint: disable=invalid-name


################# Select dataset
timepoint = 12  # OS12

# for rad+mb+clhe
data_inventory = {
    6: "./datasets_local/baseline_dataframe_OS6_20240906A_mb_clhe_3random(2-4)_treatmentarm(1)_strat.csv",
    9: "./datasets_local/baseline_dataframe_OS9_20240906A_mb_clhe_3random(2-4)_treatmentarm(1)_strat.csv",
    12: "./datasets_local/baseline_dataframe_OS12_20240906A_mb_clhe_3random(2-4)_treatmentarm(1)_strat.csv",
    15: "./datasets_local/baseline_dataframe_OS15_20240906A_mb_clhe_3random(2-4)_treatmentarm(1)_strat.csv",
}

file_path = data_inventory[timepoint]
print("File path: ", file_path)
data = pd.read_csv(file_path, index_col=0)
data.columns = data.columns.astype(str)

###############Select treatment arm
# #data = (data[data["TREATMENT"] == "Pembrolizumab"]).copy()
# #print("Number of samples (Pembro only): ", len(data))

# #data = (data[data["TREATMENT"] == "Bintrafusp alfa"]).copy()
# #print("Number of samples (Bintra only): ", len(data))


##############Select features
# features dict
feat_inventory = {
    # "rad": "./datasets_local/20240906A_rad_trmt.csv",
    "mb": "./datasets_local/20240906A_mb_trmt_3noise.csv",
    "clhe": "./datasets_local/20240906A_clhe_trmt.csv",
    # "rad_mb": "./datasets_local/20240906A_rad_mb_trmt_3noise.csv",
    # "rad_clhe": "./datasets_local/20240906A_rad_clhe_trmt.csv",
    "mb_clhe": "./datasets_local/20240906A_mb+clhe_trmt_3noise.csv",
    # "rad_mb_clhe": "./datasets_local/20240906A_rad_mb_clhe_trmt_3noise.csv",
}

## iterate through feature dicts

for key, feature_file in feat_inventory.items():
    dataset_key = str(key)
    features = pd.read_csv(
        feature_file,
        index_col=0,
    )

    # from here, create and run study
    print("Number features in feature file:", len(features))
    print(
        "Number of unique features in feature file:",
        len(set(features["features"].tolist())),
    )

    features = features["features"].astype(str).tolist()

    target_column = [f"OS_DURATION_{int(timepoint)}MONTHS"]
    sample_column = "SUBJECT_ID"
    stratification_column = [
        f"STRAT_OS{int(timepoint)}_TRT_NUM"
    ]  # "OS_DURATION_6MONTHS"

    # pre-process data
    print("Number of samples with target values:", len(data[target_column]))

    ### Create OctoData Object

    # We define the data, target columns, feature columns, sample ID to identify groups,
    # and the data split type. For this classification approach,
    # we also define a stratification column.
    octo_data = OctoData(
        data=data,
        target_columns=target_column,
        feature_columns=features,
        sample_id=sample_column,
        datasplit_type="sample",
        stratification_column=stratification_column,
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
        name=f"MBOS{int(timepoint)}_mbclhe_5steps_{dataset_key}",
        ml_type="classification",
        target_metric="AUCROC",
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
        # run_single_experiment_num=1,
    )

    config_sequence = ConfigSequence(
        [
            # Step0:
            Roc(
                # loading of existing results
                load_sequence_item=False,
                description="step_0_ROC",
                threshold=0.85,
                correlation_type="spearmanr",
                filter_type="f_statistics",  # "mutual_info"
            ),
            # Step1: octo
            Octo(
                description="step_1_octo",
                # loading of existing results
                load_sequence_item=False,
                # datasplit
                n_folds_inner=5,
                # model selection
                models=[
                    # "TabPFNClassifier",
                    "ExtraTreesClassifier",
                    # "RandomForestClassifier",
                    # "CatBoostClassifier",
                    # "XGBClassifier",
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
                n_trials=700,
                max_features=70,
                penalty_factor=1.0,
                # ensemble selection
                # ensemble_selection=True,
                # ensel_n_save_trials=75,
            ),
            # Step2: MRMR
            Mrmr(
                description="step2_mrmr",
                # loading of existing results
                load_sequence_item=False,
                # model_name
                model_name="best",
                # number of features selected by MRMR
                n_features=60,
                # what correlation type should be used
                correlation_type="rdc",  # "rdc"
                # relevance type
                relevance_type="permutation",
                # feature importance type (mean/count)
                feature_importance_type="mean",
                # feature importance method (permuation/shap/internal)
                feature_importance_method="permutation",
            ),
            # Step3: octo
            Octo(
                description="step_3_octo",
                # loading of existing results
                load_sequence_item=False,
                # datasplit
                n_folds_inner=5,
                # model selection
                models=[
                    # "TabPFNClassifier",
                    "ExtraTreesClassifier",
                    # "RandomForestClassifier",
                    # "CatBoostClassifier",
                    # "XGBClassifier",
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
                n_trials=100,
                max_features=60,
                penalty_factor=1.0,
                # ensemble selection
                # ensemble_selection=True,
                # ensel_n_save_trials=75,
            ),
            # Step4: rfe
            Rfe(
                description="rfe",
                # loading of existing results
                load_sequence_item=False,
                model="RandomForestClassifier",
                cv=5,
                mode="Mode1",
            ),
            # Step5: octo
            Octo(
                description="step_5_octo",
                # loading of existing results
                load_sequence_item=False,
                # datasplit
                n_folds_inner=5,
                # model selection
                models=[
                    # "TabPFNClassifier",
                    # "ExtraTreesClassifier",
                    "RandomForestClassifier",
                    # "CatBoostClassifier",
                    # "XGBClassifier",
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
                n_trials=25,
                max_features=40,
                penalty_factor=1.0,
                # ensemble selection
                # ensemble_selection=True,
                # ensel_n_save_trials=75,
            ),
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
