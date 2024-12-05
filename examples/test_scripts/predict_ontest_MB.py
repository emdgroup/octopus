# %%
"""Workflow example for prediction on new data."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.predict import OctoPredict

# %% [markdown]
# ## Select Study

# %%
# setup
path_study = Path("./studies/MBOS6_mb_OMORO_fixed_xgb_mb")

# %% [markdown]
# ## Study Information

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
print("Octo sequence items:", octo_seq_lst)
print()


# print('Config sequence:',config.sequence.sequence_items)

# %% [markdown]
# ## Setup Predict

# %%
# Select sequence item
sequence_item_id = 0
print()

# check available results_keys in cell output below
results_key = "best"

# create study-predict object
print(f"Predictions using sequence item: {sequence_item_id} and results: {results_key}")
seq_item = OctoPredict(study_path=path_study, sequence_item_id=sequence_item_id)

print()
path_sequences = [
    f for f in path_study.glob(f"*/sequence{sequence_item_id}") if f.is_dir()
]
path_exp_pkl = path_sequences[0].joinpath(
    f"exp{0}_{sequence_item_id}.pkl"
)  # one example

if path_exp_pkl.exists():
    # load experiment
    exp = OctoExperiment.from_pickle(path_exp_pkl)
    # iterate through keys
    print("Available results keys:", exp.results.keys())

# %% [markdown]
# ## Calculate Test feature importances

# %%
# (C) calculate permutation feature importances using final models (bag)
#     on test data
# - fi tables are saved in the  study.results dictionary
# - pdf plots are saved in the results directory of the sequence item

# calculate pfi for one experiment
# seq_item.calculate_fi_test(fi_type="group_permutation", n_repeat=5, experiment_id=4)


# calculate pfi for all available experiments
# for cnt in range(seq_item.n_experiments):
#    seq_item.calculate_fi_test(
#        fi_type="group_permutation", n_repeat=120, experiment_id=cnt
#    )

print("PFI calculation running.....")
# seq_item.calculate_fi_test_grouppfi_all(n_repeat=10)
seq_item.calculate_fi_test(fi_type="group_permutation", n_repeat=2)

# %% [markdown]
#

# %%
# (D) calculate shap feature importances using final models (bag)
#     on test data
seq_item.calculate_fi_test(fi_type="shap", shap_type="kernel")
# - for highest quality use "exact" or "kernel"
# - shap_type could be ["kernel", "permutation", "exact"]
# - shap_type "exact" does not scale well with number of features
# - shap_type "permutation" scales better than "exact" but
#   takes longer for a small number of features
# - shap_type "kernel" does scales better than "exact" but is slower than "permutation"
# - fi tables are saved in the  study.results dictionary
# - pdf plots are saved in the results directory

# %%
# results dict
print("Results keys:")
print(list(seq_item.results.keys()))

# %% [markdown]
# ## Analyse test feature importance over all outer folds

# %%
pfi0 = seq_item.results["fi_table_grouppermutation_exp0"]
pfi1 = seq_item.results["fi_table_grouppermutation_exp1"]
pfi2 = seq_item.results["fi_table_grouppermutation_exp2"]
pfi3 = seq_item.results["fi_table_grouppermutation_exp3"]
pfi4 = seq_item.results["fi_table_grouppermutation_exp4"]

# %%


def plot_pfi(df):
    """Create plot for permutation fi and save to file."""
    # Calculate error bars
    lower_error = df["importance"] - df["ci_low_95"]
    upper_error = df["ci_high_95"] - df["importance"]
    error = [lower_error.values, upper_error.values]

    plt.figure(figsize=(8.27, 11.69))  # portrait orientation (A4)
    _ = plt.barh(
        df["feature"],
        df["importance"],
        xerr=error,
        capsize=5,
        color="royalblue",
        # edgecolor="black",
    )

    # Adding labels and title
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.title("Feature Importance with Confidence Intervals")
    plt.grid(True, axis="x")

    # Adjust layout to make room for the plot
    plt.tight_layout()
    plt.show()
    plt.close()


# %% [markdown]
#

# %%
plot_pfi(pfi4)

# %%
