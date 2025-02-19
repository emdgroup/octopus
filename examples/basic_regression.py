## Basic example for using Octopus regression

# This example demonstrates how to use Octopus to create a machine learning regression model.
# We will use the famous California housing dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo

### Load and Preprocess Data

# First, we load the Titanic dataset and preprocess it
# to ensure it's clean and suitable for analysis. We restrict it
# to only the first 100 entries to make this example faster.

data_df = (
    pd.read_csv(os.path.join(os.getcwd(), "datasets", "california_housing_prices.csv"))
    .reset_index()
    .astype(
        {
            "housing_median_age": int,
            "total_rooms": int,
            "population": int,
            "households": int,
            "median_income": int,
            "median_house_value": int,
        }
    )
    .loc[0:100, :]
)
### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. The columns total_bedrooms and ocean_proximity are not
# cleaned yet. Therefore we leave them out of the example.

octo_data = OctoData(
    data=data_df,
    target_columns=["median_house_value"],
    feature_columns=[
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "population",
        "households",
        "median_income",
        # "total_bedrooms",
        # "ocean_proximity",
    ],
    sample_id="index",
    datasplit_type="sample",
)

### Create Configuration

# We create three types of configurations:
# 1. `ConfigStudy`: Sets the name, machine learning type (classification), and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use one sequence with the `RandomForestRegressor` and `XGBRegressor` model.

config_study = ConfigStudy(
    name="basic_regression_example",
    ml_type="regression",
    target_metric="MSE",
    ignore_data_health_warning=True,
    silently_overwrite_study=True,
)

config_manager = ConfigManager(outer_parallelization=True, run_single_experiment_num=1)

config_sequence = ConfigSequence(
    [
        Octo(
            item_id=1,
            input_item_id=0,
            description="step_1",
            models=["RandomForestRegressor", "XGBRegressor"],
            n_trials=5,
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

octo_ml.create_outer_experiments()
octo_ml.run_outer_experiments()

# This completes the basic example for using Octopus regression
# with the Calfifornia housing dataset.
# The workflow involves loading and preprocessing
# the data, creating necessary configurations, and
# executing the machine learning pipeline.
