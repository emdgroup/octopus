"""Workflow script for test dataset Martin."""

import os
import socket


# OPENBLASE config needs to be before pandas, autosk
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Mrmr, Octo, Roc

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
    name="MBOS6_test2",
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
    run_single_experiment_num=1,
)

config_sequence = ConfigSequence(
    [
        # Step0:
        Roc(
            description="ROC",
            threshold=0.8,
            correlation_type="spearmanr",
            filter_type="f_statistics",  # "mutual_info"
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
