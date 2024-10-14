"""Workflow script for CLP use case."""

import os
import socket
from pathlib import Path

# OPENBLASE config needs to be before pandas, autosk
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Mrmr, Octo, Rfe

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


# load test dataset from Martin from csv and perform pre-processing
# stored in ./datasets_local/ to avoid accidental uploading to github
# pylint: disable=invalid-name


################# Select dataset
file_path = Path("./datasets_local/CLP_on_treat_OS6m.csv")
print("File path: ", file_path)
data = pd.read_csv(file_path, index_col=0)
data.columns = data.columns.astype(str)
data["STRAT_OS6m_TRT_NUM"] = data["STRAT_OS6m_TRT_NUM"].astype(int)

###############Select treatment arm
# #data = (data[data["TREATMENT"] == "Pembrolizumab"]).copy()
# #print("Number of samples (Pembro only)yes: ", len(data))

# #data = (data[data["TREATMENT"] == "Bintrafusp alfa"]).copy()
# #print("Number of samples (Bintra only): ", len(data))


##############Define features
drop_cols = [
    "SUBJECT ID",
    "TIME POINT",
    "OS_6m_binary",
    "TREAT_binary",
    "STRAT_OS6m_TRT",
    "STRAT_OS6m_TRT_NUM",
]
features = list(set(data.columns) - set(drop_cols))
features = [str(x) for x in features]

target_column = ["OS_6m_binary"]
sample_column = "SUBJECT ID"
stratification_column = ["STRAT_OS6m_TRT_NUM"]

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
    name="20241011C_CLP_OS6m_ontreat",
    ml_type="classification",
    target_metric="AUCROC",
    metrics=["ACCBAL", "ACC", "LOGLOSS"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    path="./studies/",
    silently_overwrite_study=False,
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    # run_single_experiment_num=0,
)

config_sequence = ConfigSequence(
    [
        # step0
        Octo(
            description="step_0_octo",
            # loading of existing results
            load_sequence_item=False,
            # datasplit
            n_folds_inner=5,
            # model selection
            models=[
                # "TabPFNClassifier",
                # "ExtraTreesClassifier",
                # "RandomForestClassifier",
                # "CatBoostClassifier",
                "XGBClassifier",
            ],
            max_outl=3,
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            n_trials=700,
            max_features=70,
            penalty_factor=1.0,
            # ensemble selection
            ensemble_selection=True,
            ensel_n_save_trials=75,
        ),
        # Step1: MRMR
        Mrmr(
            description="step1_mrmr",
            # loading of existing results
            load_sequence_item=False,
            # number of features selected by MRMR
            n_features=40,
            # what correlation type should be used
            correlation_type="rdc",
            # relevance type
            relevance_type="f-statistics",
        ),
        # Step2: octo
        Octo(
            description="step_2_octo",
            # loading of existing results
            load_sequence_item=False,
            # datasplit
            n_folds_inner=5,
            # model selection
            models=[
                # "TabPFNClassifier",
                # "ExtraTreesClassifier",
                # "RandomForestClassifier",
                # "CatBoostClassifier",
                "XGBClassifier",
            ],
            max_outl=3,
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            n_trials=100,
            max_features=40,
            penalty_factor=1.0,
            # ensemble selection
            ensemble_selection=True,
            ensel_n_save_trials=75,
        ),
        # Step3: rfe
        Rfe(
            description="step3_rfe",
            # loading of existing results
            load_sequence_item=False,
            model="RandomForestClassifier",
            cv=5,
        ),
        # Step4: octo
        Octo(
            description="step_4_octo",
            # loading of existing results
            load_sequence_item=False,
            # datasplit
            n_folds_inner=5,
            # model selection
            models=[
                # "TabPFNClassifier",
                # "ExtraTreesClassifier",
                # "RandomForestClassifier",
                # "CatBoostClassifier",
                "XGBClassifier",
            ],
            max_outl=3,
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            n_trials=100,
            max_features=40,
            penalty_factor=1.0,
            # ensemble selection
            ensemble_selection=True,
            ensel_n_save_trials=75,
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
