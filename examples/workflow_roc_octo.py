"""Example: Use of ROC and Octo modules for breast cancer classification."""

# This example demonstrates how to use Octopus with ROC (Remove Outliers and Correlations)
# and Octo modules for binary classification on the breast cancer dataset.
# The workflow includes:
# 1. ROC module for feature correlation analysis and filtering
# 2. Octo module for model training and hyperparameter optimization

import os
import socket

from sklearn.datasets import load_breast_cancer

from octopus import (
    OctoData,
    OctoML,
)
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo, Roc

print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
print("Working directory: ", os.getcwd())

### Load and Preprocess Data

# Load the breast cancer dataset from sklearn
# This is a binary classification dataset with 30 features
# Target: 0 = malignant, 1 = benign

breast_cancer = load_breast_cancer(as_frame=True)

df = breast_cancer["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(breast_cancer["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

### Create OctoData Object

# Configure the data object with target and feature columns
# Use stratified sampling to maintain class balance across folds
octo_data = OctoData(
    data=df,
    target_columns=["target"],
    feature_columns=features,
    sample_id="index",
    datasplit_type="sample",
    stratification_column="target",
)

### Create Configuration

# Configure the study parameters for breast cancer classification
config_study = ConfigStudy(
    name="example_roc_octo",
    ml_type="classification",
    target_metric="ACCBAL",  # Balanced accuracy for binary classification
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    datasplit_seed_outer=1234,
    n_folds_outer=5,
    start_with_empty_study=True,
    path="./studies/",
    silently_overwrite_study=True,
    ignore_data_health_warning=True,
)

# Configure parallel execution settings
config_manager = ConfigManager(
    # Enable outer loop parallelization for faster execution
    outer_parallelization=True,
    # Process only first outer loop experiment for quick testing
    run_single_experiment_num=0,
)

# Define the two-step sequence: ROC filtering followed by Octo training
config_workflow = ConfigWorkflow(
    [
        # Step 0: ROC - Remove highly correlated features and apply statistical filtering
        Roc(
            description="step_0_roc",
            task_id=0,
            depends_on_task=-1,  # First step, no input dependency
            load_task=False,
            threshold=0.85,  # Remove features with correlation > 0.85
            correlation_type="spearmanr",  # Use Spearman correlation
            filter_type="f_statistics",  # Apply F-statistics filtering
        ),
        # Step 1: Octo - Train models on filtered features from ROC step
        Octo(
            description="step_1_octo",
            task_id=1,
            depends_on_task=0,  # Use output from ROC step
            load_task=False,
            # Cross-validation settings
            n_folds_inner=5,
            # Model selection - using ExtraTreesClassifier for this example
            models=[
                "ExtraTreesClassifier",
                # Additional models can be uncommented as needed:
                # "TabPFNClassifier",
                # "RandomForestClassifier",
                # "GaussianProcessClassifier",
                # "CatBoostClassifier",
                # "XGBClassifier",
            ],
            model_seed=0,
            n_jobs=1,
            max_outl=0,  # No outlier removal
            fi_methods_bestbag=["permutation"],  # Feature importance method
            # Parallelization settings
            inner_parallelization=True,
            n_workers=5,
            # Hyperparameter optimization with Optuna
            optuna_seed=0,
            n_optuna_startup_trials=10,
            resume_optimization=False,
            n_trials=12,  # Number of hyperparameter optimization trials
            max_features=12,  # Maximum number of features to select
            penalty_factor=1.0,
            # Ensemble selection settings (commented out)
            # ensemble_selection=False,
            # ensel_n_save_trials=75,
            # mrmr_feature_numbers=[3, 4, 5, 6],
        ),
    ]
)

### Execute the Machine Learning Workflow

# Initialize and run the complete ROC + Octo workflow
octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_workflow=config_workflow,
)
octo_ml.run_study()

print("ROC + Octo workflow completed successfully")
