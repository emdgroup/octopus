"""Time to event models."""

from sksurv.ensemble import ExtraSurvivalTrees

from octopus.models.machine_learning.config import ModelConfig
from octopus.models.machine_learning.hyperparameter import Hyperparameter


def get_time_to_event_models():
    """Return a list of ModelConfig objects for time to event models.

    Each ModelConfig object contains the configuration for a specific time to event
    model, including the model class, hyperparameters, and other settings.

    Returns:
        List[ModelConfig]: A list of ModelConfig objects for time to event models.
    """
    return [
        ModelConfig(
            name="ExtraTreesSurv",
            model_class=ExtraSurvivalTrees,
            ml_type="timetoevent",
            feature_method="permutation",
            n_repeats=2,
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(
                    type="int", name="n_estimators", low=100, high=500, log=False
                ),
            ],
            translate={
                "n_jobs": "n_jobs",
                "model_seed": "random_state",
            },
        ),
    ]
