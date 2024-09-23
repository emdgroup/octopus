# %%
# Andreas Wurl, 2024.09.08
# Analyse Study

# %%
import os
import re
import socket
from pathlib import Path

import pandas as pd

from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment

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
# how to evaluate
# reference: MBOS6_mb_OMORO_fixed_xgb_mb_clhe, Octo-MrmR-Octo(ensel)=final result

# %%
# path_study = Path("./studies/MBOS6_mbclhe_5steps_mb")
# path_study = Path("./studies/MBOS6_sfs_xtree_all") #0.79/0.808
# path_study = Path("./studies/MBOS6_sfs_xgb_all")
# path_study = Path("./studies/MBOS6_sfs_rf_all")
# path_study = Path("./studies/MBOS6_sfs_xtree_all")
# path_study = Path("./studies/MBOS6_mbclhe_5steps_ROC08_mb")
# path_study = Path("./studies/MBOS6_mb_5steps_ROC085_MRMR50mb")
# path_study = Path("./studies/MBOS6_mb_5steps_NoROC_MRMR50mb")
# path_study = Path("./studies/MBOS6_radmbclhe_4steps_mb") #small dataset MBOS6_mb(small)_OctoOctoRfeOcto_xgb_mb
# path_study = Path("./studies/MBOS6_mb(small)_OctoOctoRfeOcto_xgb_mb")
# path_study = Path("./studies/MBOS6_mb(small)_OctoMrmrOctoRfeOcto_xtree_new_strat_mb")
# path_study = Path("./studies/MBOS12_mb(small)_OctoMrmrOctoRfeOcto_xgb_mb")
# path_study = Path("./studies/MBOS6_mb_OMORO_xgb_mb_clhe")
# path_study = Path("./studies/MBOS6_mb_oomoro_v2_xgb_mb")
# path_study = Path("./studies/MBOS6_mb_OMORO_fixed_xgb_mb_clhe")
# path_study = Path("./studies/MBOS9_mb_OMORO_fixed_xgb_mb_clhe")
# path_study = Path("./studies/MBOS6_mb_test1_romo_xgb_mb")
# path_study = Path("./studies/MBOS6_mb_test2_romo_all_mb")
# path_study = Path("./studies/MBOS6_mb_romoro_xt_mb")
path_study = Path("./studies/MBOS6_mb_OMORO_fixed_xgb_mb")  # recommended
# path_study = Path("./studies/MBOS6_mb_OMORO_xgb_mb")


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

# get octo sequences
octo_seq_lst = list()
for cnt, item in enumerate(sequence_items):
    print(f"Sequence {cnt}:  {item.module}")
    if item.module == "octo":
        octo_seq_lst.append(cnt)
print()
print("Octo sequence items:", octo_seq_lst)

print("Config sequence:", config.sequence.sequence_items)

# %% [markdown]
# ## Read Scores

# %%
# get all experiments
path_experiments = [f for f in path_study.glob("experiment*") if f.is_dir()]
# results df
df = pd.DataFrame(
    columns=[
        "Experiment",
        "Sequence",
        "Sequene_name",
        "Results_key",
        "Scores_dict",
        "n_features",
        "Selected_features",
    ]
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
                sel_features = exp.results[key].selected_features
                df.loc[len(df)] = [
                    exp_num,
                    seq_num,
                    seq_name,
                    str(key),
                    exp.results[key].scores,
                    len(sel_features),
                    sel_features,
                ]


# %%
sel_feat = df.iloc[1]["Selected_features"]
print("number of selected features: ", len(sel_feat))

# %%
df


# %%
# table with selected features
df_sel_feat = df[~df["Results_key"].isin(["ensel"])][
    ["Experiment", "Sequence", "Results_key", "n_features", "Selected_features"]
]
df_sel_feat

# %% [markdown]
#

# %%
sorted(df_sel_feat.iloc[0]["Selected_features"])

# %%
for num_sequence, item in enumerate(sequence_items):
    print(f"\033[1mSequence item: {num_sequence}({item.module})\033[0m")

    df_seq = df[df["Sequence"] == num_sequence]
    # display(df_seq)

    # available results keys
    res_keys = sorted(set(df_seq["Results_key"].tolist()))
    print("Available results keys:", res_keys)

    for key in res_keys:
        print("Selected results key:", key)
        df_seq_selected = df_seq.copy()
        df_seq_selected = df_seq_selected[df_seq_selected["Results_key"] == key]
        # Expand the Scores_dict column into separate columns
        scores_df = df_seq_selected["Scores_dict"].apply(pd.Series)
        # Combine with the original DataFrame, setting 'Experiment' as the index
        result_df = (
            df_seq_selected[["Experiment"]].join(scores_df).set_index("Experiment")
        )
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
                mean_values[
                    column
                ] = ""  # Assign an empty string for non-numeric columns
        # Append the mean values as a new row
        result_df.loc["Mean"] = mean_values
        print(result_df)

# %%
print("done")

# %%


# %%
