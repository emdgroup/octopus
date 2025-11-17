"""Multiclass Workflow script for Octopus using Wine dataset."""

# Multiclass classification example using Octopus

# This example demonstrates how to use Octopus to create a multiclass classification model.
# We will use the Wine dataset from sklearn for this purpose.
# The Wine dataset contains 3 classes (wine types) with 13 features.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

import os
import socket

from sklearn.datasets import load_wine

from octopus import (
    OctoData,
    OctoML,
)
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
print("Working directory: ", os.getcwd())

### Load and Preprocess Data

# First, we load the Wine dataset and preprocess it
# to ensure it's clean and suitable for analysis.

### Load the wine dataset
wine = load_wine(as_frame=True)

df = wine["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(wine["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

print("Dataset info:")
print(f"  Features: {len(features)}")
print(f"  Samples: {df.shape[0]}")
print(f"  Classes: {len(wine.target_names)} - {wine.target_names}")
print(f"  Target distribution: {df['target'].value_counts().sort_index().to_dict()}")

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
# 1. `ConfigStudy`: Sets the name, machine learning type (multiclass),
# and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigWorkflow`: Defines the workflows to be executed. In this example,
# we use one workflow with multiclass classification models.

config_study = ConfigStudy(
    name="multiclass_wine",
    ml_type="multiclass",
    target_metric="AUCROC_MACRO",
    metrics=["AUCROC_MACRO", "AUCROC_WEIGHTED", "ACCBAL_MC"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    start_with_empty_study=True,
    path="./studies/",
    silently_overwrite_study=True,
    ignore_data_health_warning=True,
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=0,
)

config_workflow = ConfigWorkflow(
    [
        # Step1: octo multiclass
        Octo(
            description="step_1_octo_multiclass",
            task_id=0,
            depends_on_task=-1,
            # loading of existing results
            load_task=False,
            # datasplit
            n_folds_inner=5,
            # model selection - using models that work well with multiclass
            models=[
                "ExtraTreesClassifier",
                "RandomForestClassifier",
                "XGBClassifier",
                "CatBoostClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,
            fi_methods_bestbag=["permutation"],
            # parallelization
            inner_parallelization=True,
            n_workers=5,
            # HPO
            n_trials=20,
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
    config_workflow=config_workflow,
)
octo_ml.run_study()

print("Multiclass workflow completed")
print("Results saved to: ./studies/multiclass_wine/")
