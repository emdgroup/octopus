"""Basic example for using Octopus regression."""

# This example demonstrates how to use Octopus to create a machine learning regression model.
# We will use the famous diabetes dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
from sklearn.datasets import load_diabetes

from octopus import OctoStudy

### Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)

### Create and run OctoStudy
# Note: ml_type is now automatically inferred from the data
# The target column is numeric with many unique values, so it will be inferred as "regression"
study = OctoStudy(
    name="basic_regression",
    target_metric="MAE",
    feature_columns=diabetes["feature_names"],
    target_columns=["target"],
    sample_id="index",
)

study.fit(data=diabetes["frame"].reset_index())

print("Workflow completed")
