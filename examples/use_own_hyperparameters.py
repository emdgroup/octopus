"""Example for using custom hyperparameters in Octopus regression."""

# This example demonstrates how to use Octopus with custom hyperparameters.
# Instead of letting Optuna automatically search the hyperparameter space,
# you can define your own hyperparameter ranges for the models.
# We will use the diabetes dataset for this purpose.

### Necessary imports for this example
from sklearn.datasets import load_diabetes

from octopus import OctoStudy
from octopus.models.hyperparameter import IntHyperparameter
from octopus.modules import Octo

### Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)

### Create and run OctoStudy with custom hyperparameters
study = OctoStudy(
    name="use_own_hyperparameters_example",
    ml_type="regression",
    target_metric="MAE",
    feature_columns=diabetes["feature_names"],
    target_columns=["target"],
    sample_id="index",
    ignore_data_health_warning=True,
    outer_parallelization=False,
    run_single_experiment_num=0,
    workflow=[
        Octo(
            task_id=0,
            models=["RandomForestRegressor"],
            n_trials=3,
            hyperparameters={
                "RandomForestRegressor": [
                    IntHyperparameter(name="max_depth", low=2, high=32),
                    IntHyperparameter(name="min_samples_split", low=2, high=100),
                ]
            },
        ),
    ],
)

study.fit(data=diabetes["frame"].reset_index())

print("Workflow completed")

# This example demonstrates how to use custom hyperparameters with Octopus.
# The key difference from the basic example is the use of the `hyperparameters` parameter
# in the Octo configuration, where you can define custom hyperparameter ranges
# for each model using the Hyperparameter class.
