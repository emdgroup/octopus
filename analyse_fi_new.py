# analyse experiment

from pathlib import Path

import numpy as np
import shap

from octopus.experiment import OctoExperiment


print("Investigate experiment 1")

path_exp = Path("./studies/20240110B/experiment1/sequence0/exp1_0.pkl")
experiment = OctoExperiment.from_pickle(path_exp)

print("Model keys:", experiment.models.keys())

best_bag = experiment.models["best"]

print(best_bag.feature_importances)
