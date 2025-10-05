"""Basic example for using Octopus Classification."""

# This example demonstrates how to use Octopus to create a machine learning classification model.
# We will use the famous Titanic dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
from sklearn.datasets import load_breast_cancer

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Octo

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
# 1. `ConfigStudy`: Sets the name, machine learning type (classification), and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use one sequence with the `RandomForestClassifier` model.

config_study = ConfigStudy(
    name="basic_classification",
    ml_type="classification",
    target_metric="AUCROC",
    silently_overwrite_study=True,
)

config_manager = ConfigManager(outer_parallelization=True, run_single_experiment_num=0)

config_sequence = ConfigSequence(
    sequence_items=[
        Octo(
            description="step_1_octo",
            sequence_id=0,
            models=[
                "CatBoostClassifier",
                # "XGBClassifier",
                # "RandomForestClassifier",
                # "ExtraTreesClassifier",
                # "RandomForestClassifier",
            ],
            n_trials=5,
        )
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
octo_ml.run_study()

print("Workflow completed")

# This completes the basic example for using Octopus Classification
# with the Titanic dataset. The workflow involves loading and preprocessing
# the data, creating necessary configurations, and executing the machine learning pipeline.
