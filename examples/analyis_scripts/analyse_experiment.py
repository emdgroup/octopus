"""Analyse experiment."""

import copy
import os
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
path_study = Path("./studies/MBOS6_FI_TEST")


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

path_experiments = [f for f in path_study.glob("experiment*") if f.is_dir()]

# %% [markdown]
# ## Analyse a specific sequence item

# %%
# get octo
experiment_num = 1
sequence_item = 0

path_experiment = path_study.joinpath(f"experiment{experiment_num}")
path_sequence = path_experiment.joinpath(f"sequence{sequence_item}")
path_exp_pkl = path_sequence.joinpath(f"exp{experiment_num}_{sequence_item}.pkl")

if path_exp_pkl.exists():
    # load experiment
    exp = OctoExperiment.from_pickle(path_exp_pkl)

octo = copy.deepcopy(exp)

# %%
# get mrmr
experiment_num = 1
sequence_item = 1

path_experiment = path_study.joinpath(f"experiment{experiment_num}")
path_sequence = path_experiment.joinpath(f"sequence{sequence_item}")
path_exp_pkl = path_sequence.joinpath(f"exp{experiment_num}_{sequence_item}.pkl")

if path_exp_pkl.exists():
    # load experiment
    exp = OctoExperiment.from_pickle(path_exp_pkl)

mrmr = copy.deepcopy(exp)

# %% [markdown]
# ## Analyse experiments

# %%
# check selected features
sel_feat = octo.selected_features
print("Number of selected features:", len(sel_feat))

# %%
# investigate octo results, best bag
len(octo.results["best"].selected_features)

# %%
len(octo.results["best"].model.get_selected_features(fi_methods="permutation"))


# %%
# len(octo.results["best"].model.get_selected_features(fi_methods = "internal"))

# %%
# analyse PFI, permutation_dev_mean
fis = mrmr.prior_feature_importances
pfi_dev = fis["best"]["permutation_dev_mean"]
print("Number of entries in pfi:", len(pfi_dev))
pfi_dev_pos = pfi_dev[pfi_dev["importance"] != 0]
# pfi_dev_pos = pfi_dev[pfi_dev['importance']>0]
print("Number of features  with positive importance:", len(pfi_dev_pos))
pfi_dev_pos = pfi_dev_pos.sort_values(by="importance", ascending=False).reset_index()

# remove group features

pfi_dev_pos_nog = pfi_dev_pos[~pfi_dev_pos["feature"].str.startswith("group")]
print("Number of pos features, without groups:", len(pfi_dev_pos_nog))

pfi_features = pfi_dev_pos_nog["feature"].tolist()
pfi_features

difference1 = set(pfi_features).difference(set(sel_feat))
difference2 = set(sel_feat).difference(set(pfi_features))
shared = set(pfi_features).intersection(set(sel_feat))
print()
print("Share features: ", len(shared))
print("Features in pfi but missing in selected features: ", len(difference1))
print("Features in selected features but missing in pfi: ", len(difference2))

# %%
pfi_dev_pos

# %%
# Questions:
# - why only 31 shared (43?)?
# - where do the 12 come from that show up in selected features but not in pfi
# - why are 46 more features in pfi?
# - how are selected feature calculated?
# - how are pfi calculated?

# %%
## Investigate octo.bag

bag = octo.results["best"].model

feat_imp = dict()
# save feature importances for every training in bag
for training in bag.trainings:
    feat_imp[training.training_id] = training.feature_importances

print("Feature keys: ", feat_imp.keys())
print("Feature key for 1_0_0:", feat_imp["1_0_0"].keys())

method_str = "permutation_dev"
fi_pool = list()
for training in bag.trainings:
    fi_pool.append(training.feature_importances[method_str])

# %%
feat_imp["1_0_0"].keys()

# %%
# analyse PFI, permutation_dev_mean
fis = mrmr.prior_feature_importances
pfi_dev = fis["best"]["permutation_dev_mean"]
print("Number of entries in pfi:", len(pfi_dev))
pfi_dev_pos = pfi_dev[pfi_dev["importance"] != 0]
# pfi_dev_pos = pfi_dev[pfi_dev['importance']>0]
print("Number of features  with positive importance:", len(pfi_dev_pos))
pfi_dev_pos = pfi_dev_pos.sort_values(by="importance", ascending=False).reset_index()

groups_all = pfi_dev_pos[pfi_dev_pos["feature"].str.startswith("group")]
print("Number of groups:", len(groups_all))
groups_pos = groups_all[groups_all["importance"] > 0]


# pfi_dev_pos_nog = pfi_dev_pos[~pfi_dev_pos["feature"].str.startswith("group")]
# print("Number of pos features, without groups:", len(pfi_dev_pos_nog))

# pfi_features = pfi_dev_pos_nog['feature'].tolist()
# pfi_features

# difference1 = set(pfi_features).difference(set(sel_feat))
# difference2 = set(sel_feat).difference(set(pfi_features))
# shared = set(pfi_features).intersection(set(sel_feat))
# print()
# print("Share features: ",len(shared))
# print("Features in pfi but missing in selected features: ",len(difference1))
# print("Features in selected features but missing in pfi: ",len(difference2))

# %%


# %%

groups = groups_pos["feature"].tolist()

# take first element in each group
gfeatures = [
    octo.feature_groups[key][0] for key in groups if key in octo.feature_groups
]

print("Number of groups:", len(groups))
print("Number of group features: ", len(gfeatures))
print("NUmber of feaures: ", len(pfi_dev_pos))
print("All features", len(set(gfeatures + pfi_dev_pos["feature"].tolist())))

# %%
