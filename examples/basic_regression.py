"""Basic example for using Octopus regression."""

# This example demonstrates how to use Octopus to create a machine learning regression model.
# We will use the famous California housing dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
from sklearn.datasets import load_diabetes

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigStudy, ConfigWorkflow
from octopus.modules import Octo

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

# We create three types of configurations:
# 1. `ConfigStudy`: Sets the name, machine learning type (classification), and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigWorkflow`: Defines the workflows to be executed. In this example,
# we use one workflow with the `RandomForestRegressor` and `XGBRegressor` model.

config_study = ConfigStudy(
    name="basic_regression_example",
    ml_type="regression",
    target_metric="MAE",
    ignore_data_health_warning=True,
    silently_overwrite_study=True,
)

config_manager = ConfigManager(outer_parallelization=True, run_single_experiment_num=-1)

config_workflow = ConfigWorkflow(
    [
        Octo(
            task_id=0,
            depends_on_task=-1,
            description="step_1_neural_network",
            models=[
                "TabularNNRegressor",
                # "ExtraTreesRegressor",
            ],
            n_trials=40,
        ),
    ]
)

## Execute the Machine Learning Workflow

# We add the data and the configurations defined earlier
# and run the machine learning workflow.

octo_ml = OctoML(
    octo_data,
    config_study=config_study,
    config_manager=config_manager,
    config_workflow=config_workflow,
)
octo_ml.run_study()

print("Workflow completed")
