"""Example for using a multi workflow."""

# TBD

### Necessary imports for this example
from sklearn.datasets import load_diabetes

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Mrmr, Octo

# Regression Analysis on Diabetes Dataset
# http://statweb.lsu.edu/faculty/li/teach/exst7142/diabetes.html
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
# https://automl.github.io/auto-sklearn/master/examples/20_basic/example_regression.html

### Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)

### Create OctoData Object
octo_data = OctoData(
    data=diabetes["frame"].reset_index(),
    target_columns=["target"],
    feature_columns=diabetes["feature_names"],
    sample_id="index",
    datasplit_type="sample",
)

### Create Configuration

# TDB

config_study = ConfigStudy(
    name="example_multiworkflow",
    ml_type="regression",
    target_metric="R2",
    silently_overwrite_study=True,
    ignore_data_health_warning=True,
)

config_manager = ConfigManager(
    outer_parallelization=False,
    run_single_experiment_num=1,
)

config_workflow = ConfigWorkflow(
    [
        Octo(
            description="step1_octofull",
            task_id=0,
            depends_on_task=-1,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=2,
            max_features=70,
        ),
        Mrmr(
            description="step2_mrmr",
            task_id=1,
            depends_on_task=0,
            n_features=6,
            correlation_type="rdc",
        ),
        Octo(
            description="step1_octofull",
            task_id=2,
            depends_on_task=1,
            models=["ExtraTreesRegressor", "RandomForestRegressor"],
            n_trials=1,
            max_features=70,
        ),
    ]
)

### Execute the Machine Learning Workflow

octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_workflow=config_workflow,
)
octo_ml.run_study()
