# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: octopus
#     language: python
#     name: python3
# ---

# %% [markdown] vscode={"languageId": "plaintext"}
# # Analyze Study (Classification Task)
# - version 0.1
# - 2025.01.09

# %% [markdown] vscode={"languageId": "plaintext"}
# ## ToDo
#
# - create predict directory
# - create utility functions in separate file
# - functionality:
#   1. study overview: which workflow tasks, number of splits
#   2. performance overview for certain given metric
#   3. provide feature lists for each task
# - aucpr -- baseline

# %% [markdown] vscode={"languageId": "plaintext"}
# ## Input

# %%
# INPUT: Select study
study_directory = "./studies/workflow_sequential_tasks/"

# %% [markdown]
# ## Imports

# %%
from octopus.predict import OctoPredict
from octopus.predict.notebook_utils import (
    show_selected_features,
    show_study_details,
    show_target_metric_performance,
    testset_performance_overview,
)

# %% [markdown]
# ## Show Study Details

# %%
# Call the utility function to display and validate study details
study_info = show_study_details(study_directory, expected_ml_type="classification")

# Extract key variables for use in subsequent cells
# path_study = study_info["path"]
# config = study_info["config"]
# ml_type = study_info["ml_type"]
# n_folds_outer = study_info["n_folds_outer"]
# workflow_tasks = study_info["workflow_tasks"]
# outersplit = study_info["outersplit_dirs"]
# expected_task_ids = study_info["expected_task_ids"]
# octo_workflow_lst = study_info["octo_workflow_tasks"]

# %% [markdown]
# ## Show Target Metric Performance for all  Tasks

# %%
# Display performance (target metric) for all workflow tasks
df_performance = show_target_metric_performance(study_info, details=False)

# %% [markdown]
# ## Show Selected Features Summary

# %%
# Display the number of selected features across outer splits and tasks
# Returns two tables: feature counts and feature frequency
# sort_task parameter sorts the frequency table by the specified task
selected_task=None
feature_numbers_table, feature_frequency_table = show_selected_features(df_performance, sort_task=selected_task)

# %% [markdown]
# ## Evaluate Model Performance on Test Dataset for a given Task
#

# %%
# Input: select task
selected_task = 0

# check availalbe results_keys in cell output below
results_key = "best"

print('Selected task:', selected_task)
print('Selected results_key:', results_key)
print()
# load predictor object
task_predictor = OctoPredict(study_path=study_info["path"], task_id=selected_task, results_key=results_key)


# %% [markdown]
# ### Testset Performance overview for Selected Metrics

# %%
# Input: selected metrics for performance overviwe
metrics = ["AUCROC", "ACCBAL", "ACC", "F1", "AUCPR", "NEGBRIERSCORE"]
print("Selected metrics: ", metrics)

# %%
testset_performance=testset_performance_overview(predictor=task_predictor, metrics=metrics)


# %%
