"""Basic example for using Octopus Classification with TabPFN."""

# This example demonstrates how to use Octopus to create a machine learning classification model
# using TabPFN (Tabular Prior-data Fitted Networks).
# We will use the breast cancer dataset for this purpose.
# Please ensure your dataset is clean, with no missing values (`NaN`),
# and that all features are numeric.

### Necessary imports for this example
from sklearn.datasets import load_breast_cancer

from octopus import OctoStudy
from octopus.modules import Octo

### Load and Preprocess Data
breast_cancer = load_breast_cancer(as_frame=True)

df = breast_cancer["frame"].reset_index()
df.columns = df.columns.str.replace(" ", "_")
features = list(breast_cancer["feature_names"])
features = [feature.replace(" ", "_") for feature in features]

### Create and run OctoStudy with TabPFN
study = OctoStudy(
    name="basic_classification_tabpfn",
    ml_type="classification",
    target_metric="AUCROC",
    feature_columns=features,
    target_columns=["target"],
    sample_id="index",
    stratification_column="target",
    outer_parallelization=True,
    run_single_experiment_num=0,
    tasks=[
        Octo(
            task_id=0,
            description="step_1_octo",
            models=[
                "TabPFNClassifier",
            ],
            inner_parallelization=True,
            n_trials=1,
            fi_methods_bestbag=["constant"],
        )
    ],
)

study.fit(data=df)

print("Workflow completed")
