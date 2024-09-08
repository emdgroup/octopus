# %%
# Andreas Wurl, 2024.09.08
# Analyse Study

# %%
from pathlib import Path
import pandas as pd
import re
from octopus.experiment import OctoExperiment
from octopus.config.core import OctoConfig
import socket
import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
print("Notebook kernel is running on server:", socket.gethostname())
print("Conda environment on server:", os.environ["CONDA_DEFAULT_ENV"])
# show directory name
print("Working directory: ", os.getcwd())


# %% [markdown]
# ## Settings

# %%
path_study = Path("./studies/MBOS6_mbclhe_5steps_mb")

# %% [markdown]
# ## Read Config

# %%
path_config = path_study / "config"
config = OctoConfig.from_pickle(path_config / "config.pkl")

ml_type = config.study.ml_type
target_metric = config.study.target_metric
n_folds_outer = config.study.n_folds_outer
sequence_items = config.sequence.sequence_items
print("Number of sequence items:", len(sequence_items))
for cnt, item in enumerate(sequence_items):
    print(f"Sequence {cnt}:  {item.module}")
# print(sequence_items)

# %% [markdown]
# ## Read Scores

# %%
# get all experiments
path_experiments = [f for f in path_study.glob("experiment*") if f.is_dir()]
# results df
df = pd.DataFrame(
    columns=["Experiment", "Sequence", "Sequene_name", "Results_key", "Scores_dict"]
)


# iterate through experiments
for path_exp in path_experiments:
    # name of experiment
    exp_name = str(path_exp.name)
    # number of experiment
    match = re.search(r"\d+", exp_name)
    exp_num = int(match.group()) if match else None

    # sequences
    path_sequences = [f for f in path_exp.glob("sequence*") if f.is_dir()]
    print("Processing....:", path_exp)

    # iterate through sequences
    for path_seq in path_sequences:
        # name of sequence item
        seq_name = str(path_seq.name)
        # number of sequence item
        match = re.search(r"\d+", seq_name)
        seq_num = int(match.group()) if match else None

        path_exp_pkl = path_seq.joinpath(f"exp{exp_num}_{seq_num}.pkl")

        if path_exp_pkl.exists():
            # load experiment
            exp = OctoExperiment.from_pickle(path_exp_pkl)
            # iterate through keys
            for key in exp.results.keys():
                df.loc[len(df)] = [
                    exp_num,
                    seq_num,
                    seq_name,
                    str(key),
                    exp.results[key].scores,
                ]


# %%
df


# %% [markdown]
#

# %%
df_seq = df[df["Sequence"] == 5]
display(df_seq)


# Expand the Scores_dict column into separate columns
scores_df = df_seq["Scores_dict"].apply(pd.Series)

# Combine with the original DataFrame, setting 'Experiment' as the index
result_df = df_seq[["Experiment"]].join(scores_df).set_index("Experiment")

# Remove columns that do not contain numeric values
result_df = result_df.select_dtypes(include="number")

mean_values = {}
# Iterate through the columns
for column in result_df.columns:
    if result_df[column].dtype in [
        "float64",
        "int64",
    ]:  # Check if the column contains numeric values
        mean_values[column] = result_df[column].mean()  # Calculate mean
    else:
        mean_values[column] = ""  # Assign an empty string for non-numeric columns

# Append the mean values as a new row
result_df.loc["Mean"] = mean_values

result_df

# %%
