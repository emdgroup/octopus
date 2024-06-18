"""Test script for reading saved experiments."""

from pathlib import Path
from octopus.experiment import OctoExperiment


# Setup
path_experiment = Path("./studies/classification_2/experiment4/sequence0/exp4_0.pkl")

# Read experiment
experiment = OctoExperiment.from_pickle(path_experiment)

# Show content
print("\nConfig")
print(experiment.config)

print("\nBag")
print(experiment.models)

print("\nPredictions")
print(experiment.predictions)

print("\nScores")
print(experiment.scores)


print("\nResults")
print(experiment.results)
