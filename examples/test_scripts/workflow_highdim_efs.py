"""Workflow script for the HighDim example."""

# The AlonDS data set was collected by Alon et. al. and consists of 2000 genes measured
# on 62 patients: 40 diagnosed with colon cancer and 22 healthy patients. The patient
# status (tissue type) is described in ‘tissse_numberic’ and the gene values are given
# by the numeric variables ‘genes.1’ through ‘genes.2000’
#
# Link:https://rdrr.io/cran/HiDimDA/man/AlonDS.html
#
# Source: Alon, U., Barkai, N., Notterman, D.A., Gish, K., Ybarra, S., Mack, D. and
# Levine, A.J. (1999) “Broad patterns of gene expression revealed by clustering
# analysis of tumor and normal colon tissues probed by oligonucleotide arrays”,
# In: Proceedings National Academy of Sciences USA 96, 6745-6750.
# The data set is available at http://microarray.princeton.edu/oncology
#
# The direct link to the dataset is here:
# http://genomics-pubs.princeton.edu/oncology/affydata/index.html

# performance comparision here:
# https://github.com/AutoViML/featurewiz/blob/main/examples/Featurewiz_on_2000_variables.ipynb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from octopus import OctoData, OctoML
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.modules import Efs, Roc

# import data
df = pd.read_csv("./datasets/AlonDS2000.csv", index_col=0)
df = df.drop(columns=["tissue", "identity"]).reset_index()
TARGET = "tissue_numeric"
features = [x for x in df.columns if "genes" in str(x)]
print("Shape:", df.shape)

# convert features to log
df[features] = np.log10(df[features])
df.head()

# Reference solution
### HiDimDA using High Criticism method selected 46 genes
# as being important for this data set
HC_sel = [
    "genes.25",
    "genes.42",
    "genes.65",
    "genes.74",
    "genes.137",
    "genes.186",
    "genes.240",
    "genes.244",
    "genes.248",
    "genes.266",
    "genes.364",
    "genes.376",
    "genes.390",
    "genes.398",
    "genes.492",
    "genes.512",
    "genes.580",
    "genes.624",
    "genes.764",
    "genes.779",
    "genes.801",
    "genes.821",
    "genes.896",
    "genes.963",
    "genes.991",
    "genes.1001",
    "genes.1041",
    "genes.1059",
    "genes.1152",
    "genes.1324",
    "genes.1345",
    "genes.1405",
    "genes.1422",
    "genes.1472",
    "genes.1493",
    "genes.1581",
    "genes.1633",
    "genes.1634",
    "genes.1647",
    "genes.1670",
    "genes.1729",
    "genes.1769",
    "genes.1770",
    "genes.1771",
    "genes.1842",
    "genes.1899",
]
len(HC_sel)

RFC = RandomForestClassifier(n_estimators=200, random_state=99)
X = df[HC_sel]
y = df[TARGET]
scoresFW = cross_val_score(RFC, X, y, scoring="accuracy", cv=3, n_jobs=-1)
print(scoresFW)
print(f"Average Accuracy: {100 * np.mean(scoresFW):.3f}")
# [0.9047619  0.80952381 0.6       ]
# Average Accuracy: 77%

# Feature wizard solution
# - new environment, conda install feature_wizard (we get older version, but OK)
# from featurewiz import featurewiz
# features = featurewiz(df, target, corr_limit=0.9, verbose=2)
# RFC = RandomForestClassifier(n_estimators=200, random_state=99)
# X = df[features[0]]
# y = df[target]
# scoresFW = cross_val_score(RFC, X, y, scoring="accuracy", cv=3, n_jobs=-1)
# print(scoresFW)
# print("Average Accuracy: %0.0f%%" % (100 * np.mean(scoresFW)))
# [1.        0.9047619 0.75     ]
# Average Accuracy: 88%

### (B) Octopus EFS

### Create OctoData Object

# We define the data, target columns, feature columns, sample ID to identify groups,
# and the data split type. For this classification approach,
# we also define a stratification column.
octo_data = OctoData(
    data=df,
    target_columns=[TARGET],
    feature_columns=features,
    datasplit_type="group_sample_and_features",
    stratification_column=[TARGET],
    sample_id="index",
)

### Create Configuration

# We create three types of configurations:
# 1. `ConfigStudy`: Sets the name, machine learning type (classification),
# and target metric.

# 2. `ConfigManager`: Manages how the machine learning will be executed.
# We use the default settings.

# 3. `ConfigSequence`: Defines the sequences to be executed. In this example,
# we use one sequence with the `RandomForestClassifier` model.

config_study = ConfigStudy(
    name="HigDim",
    ml_type="classification",
    target_metric="ACC",
    metrics=["AUCROC", "ACCBAL", "ACC", "LOGLOSS"],
    datasplit_seed_outer=1234,
    n_folds_outer=3,
    start_with_empty_study=True,
    path="./studies/",
    silently_overwrite_study=True,
)

config_manager = ConfigManager(
    # outer loop parallelization
    outer_parallelization=True,
    # only process first outer loop experiment, for quick testing
    run_single_experiment_num=1,
)

config_sequence = ConfigSequence(
    [
        # Step0:
        Roc(
            description="ROC",
            threshold=0.8,
            correlation_type="spearmanr",
        ),
        # Step1: efs
        Efs(
            description="EFS",
            model="RandomForestClassifier",
            n_subsets=10,
            subset_size=20,
            cv=3,
            max_n_iterations=50,
            max_n_models=100,
        ),
        # Step2: ....
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
# the data, creating necessary configurations, and executing the machine
# learning pipeline.
