## Basic example for using Octopus Classification

# This example demonstrates how to use Octopus to create a machine learning classification model.
# We will use the famous Titanic dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
import os

import numpy as np
import pandas as pd

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo

### Load and Preprocess Data

# First, we load the Titanic dataset and preprocess it
# to ensure it's clean and suitable for analysis.

data_df = (
    pd.read_csv(
        os.path.join(os.getcwd(), "datasets", "titanic_openml.csv"), index_col=0
    )
    .astype({"age": float})
    .assign(
        age=lambda df_: df_["age"].fillna(df_["age"].median()).astype(int),
        embarked=lambda df_: df_["embarked"].fillna(df_["embarked"].mode()[0]),
        fare=lambda df_: df_["fare"].fillna(df_["fare"].median()),
    )
    .astype({"survived": bool})
    .pipe(
        lambda df_: df_.reindex(
            columns=["survived"] + list([a for a in df_.columns if a != "survived"])
        )
    )
    .pipe(
        lambda df_: df_.reindex(
            columns=["name"] + list([a for a in df_.columns if a != "name"])
        )
    )
    .pipe(pd.get_dummies, columns=["embarked", "sex"], drop_first=True, dtype=int)
)

# insert NaN
data_df.loc[0, "age"] = np.NaN
data_df.loc[19, "age"] = np.NaN
data_df.loc[22, "fare"] = np.NaN

### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. For this classification approach,
# we also define a stratification column.

octo_data = OctoData(
    data=data_df,
    target_columns=["survived"],
    feature_columns=[
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked_Q",
        "embarked_S",
        "sex_male",
    ],
    sample_id="name",
    datasplit_type="group_sample_and_features",
    stratification_column="survived",
)

### Create Configuration

# We create three types of configurations:
# 1. `ConfigStudy`: Sets the name, machine learning type (classification), and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use one sequence with the `RandomForestClassifier` model.

config_study = ConfigStudy(
    name="basic_classification",
    ml_type="classification",
    target_metric="AUCROC",
    imputation_method="mice",
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=0,
)

config_sequence = ConfigSequence(
    sequence_items=[
        Octo(description="step_1_octo", models=["RandomForestClassifier"], n_trials=3)
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
# the data, creating necessary configurations, and executing the machine learning pipeline.
