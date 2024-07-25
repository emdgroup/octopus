"""Test script for reading saved experiments."""

from pathlib import Path

from octopus.experiment import OctoExperiment

# Setup
path_experiment = Path(
    "./studies/example_multisequence/experiment1/sequence1/exp1_1.pkl"
)

# Read experiment
experiment = OctoExperiment.from_pickle(path_experiment)

# Show content
print("\nConfig")
print(experiment.ml_config)

# print("\nBag")
# print(experiment.models)

# print("\nPredictions")
# print(experiment.predictions)

# print("\nScores")
# print(experiment.scores)


# print("\nResults")
# print(experiment.results)

# print("\nFeature_Importances")
# print(experiment.feature_importances)

# print("\nFeature_Importances['permutation_dev_mean']")
# print(experiment.feature_importances["permutation_dev_mean"])
