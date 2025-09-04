"""20250826A Workflow script for Octopus intro."""

## Basic example for using Octopus Classification

# This example demonstrates how to use Octopus to create a machine learning classification model.
# We will use the famous Titanic dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os
import socket

# import pandas as pd
from sklearn.datasets import load_breast_cancer

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo

# Check if this is a smoke test
SMOKE_TEST = os.environ.get("SMOKE_TEST", "").lower() in ["true", "1", "yes"]
N_TRIALS = 5 if SMOKE_TEST else 50

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
print("Working directory: ", os.getcwd())

### Load and Preprocess Data

# First, we load the Titanic dataset and preprocess it
# to ensure it's clean and suitable for analysis.

### Load the diabetes dataset
breast_cancer = load_breast_cancer(as_frame=True)

df = breast_cancer["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(breast_cancer["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

### Create OctoData Object
octo_data = OctoData(
    data=df,
    target_columns=["target"],
    feature_columns=features,
    sample_id="index",
    datasplit_type="sample",
    stratification_column="target",
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
    name=f"20250826A_octo_intro",
    ml_type="classification",
    target_metric="ACCBAL",
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    start_with_empty_study=True,
    path="./studies/",
    silently_overwrite_study=False,
    ignore_data_health_warning=True,
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=0,
)

config_sequence = ConfigSequence(
    [
        # Step0:
        # Roc(
        #     # loading of existing results
        #     load_sequence_item=False,
        #     description="step_0_ROC",
        #     threshold=0.85,
        #     correlation_type="spearmanr",
        #     filter_type="f_statistics",  # "mutual_info"
        # ),
        # Step1: octo
        Octo(
            description="step_1_octo",
            sequence_id=0,
            input_sequence_id=-1,
            # loading of existing results
            load_sequence_item=False,
            # datasplit
            n_folds_inner=5,
            # model selection
            models=[
                # "TabPFNClassifier",
                "ExtraTreesClassifier",
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
            n_trials=20,
            max_features=12,
            penalty_factor=1.0,
            # ensemble selection
            # ensemble_selection=False,
            # ensel_n_save_trials=75,
            # mrmr_feature_numbers=[3, 4, 5, 6],
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
