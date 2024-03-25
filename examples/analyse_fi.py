# analyse experiment

import pandas as pd
import numpy as np
import shap

from octopus.experiment import OctoExperiment


from pathlib import Path

for k_outer in [1]:
    print("Outer experiment:", k_outer)
    path_exp = Path(
        f"./studies/20240322A_MBOS6_octofull_5x5_ETREE/experiment{k_outer}/sequence0/exp{k_outer}_0.pkl"
    )
    experiment = OctoExperiment.from_pickle(path_exp)
    # print(experiment.feature_importances.keys())

    fi_lst = list()
    for k_inner in range(5):
        fi_dev = experiment.feature_importances[f"{k_outer}_0_{k_inner}"][
            "permutation_test"
        ]

        print("FI_SUM", fi_dev["importance"].sum())

        nonzero_features = fi_dev[fi_dev["importance"] != 0]["feature"].tolist()
        fi_lst.extend(nonzero_features)
        # print('Non-zero features: ',nonzero_features)
        print(
            f"{k_outer}_{k_inner} Number of non-zero features: {len(nonzero_features)}"
        )

    print(
        f"K_outer: {k_outer}, Number of non-zero unique features: {len(list(set(fi_lst)))}"
    )
    print("-------------------------")


print("Investigate experiment 1")

path_exp = Path(
    f"./studies/20240322A_MBOS6_octofull_5x5_ETREE/experiment1/sequence0/exp1_0.pkl"
)
experiment = OctoExperiment.from_pickle(path_exp)

print("Model keys:", experiment.models.keys())

best_bag = experiment.models["best"]

training0 = best_bag.trainings[0]


print("recalc shap fi")

training0.calculate_fi_shap(partition="dev")

fi_dev = training0.feature_importances["shap_dev"]
print("FI_SUM train0 after recalc shap", fi_dev["importance"].sum())

print("shape Explainer")

explainer = shap.Explainer(training0.model, training0.x_train)
shap_values = explainer.shap_values(training0.x_dev)

feature_importances = np.abs(shap_values).mean(axis=0)

print(feature_importances)
