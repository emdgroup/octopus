"""Example for using custom hyperparameters in Octopus regression."""

# This example demonstrates how to use Octopus with custom hyperparameters.
# Instead of letting Optuna automatically search the hyperparameter space,
# you can define your own hyperparameter ranges for the models.
# We will use the diabetes dataset for this purpose.

### Necessary imports for this example
from sklearn.datasets import load_diabetes

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.models.hyperparameter import Hyperparameter
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
# 1. `ConfigStudy`: Sets the name, machine learning type (regression), and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use `RandomForestRegressor` with custom hyperparameter ranges defined using
# the `Hyperparameter` class.

config_study = ConfigStudy(
    name="use_own_hyperparameters_example",
    ml_type="regression",
    target_metric="MAE",
    ignore_data_health_warning=True,
    silently_overwrite_study=True,
)

config_manager = ConfigManager(outer_parallelization=False, run_single_experiment_num=0)

config_sequence = ConfigSequence(
    [
        Octo(
            sequence_id=0,
            models=["RandomForestRegressor"],
            n_trials=3,
            hyperparameters={
                "RandomForestRegressor": [
                    Hyperparameter(type="int", name="max_depth", low=2, high=32),
                    Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                ]
            },
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
    config_sequence=config_sequence,
)
octo_ml.run_study()

print("Workflow completed")

# This example demonstrates how to use custom hyperparameters with Octopus.
# The key difference from the basic example is the use of the `hyperparameters` parameter
# in the Octo configuration, where you can define custom hyperparameter ranges
# for each model using the Hyperparameter class.
