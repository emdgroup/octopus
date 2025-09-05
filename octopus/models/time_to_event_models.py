"""Time to event models."""

from .config import ModelConfig
from .hyperparameter import Hyperparameter
from .registry import ModelRegistry


@ModelRegistry.register("ExtraTreesSurv")
class ExtraTreesSurvModel:
    """Extra Tree Survival Model."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        from octopus._optional.survival import ExtraSurvivalTrees

        return ModelConfig(
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
                Hyperparameter(type="int", name="n_estimators", low=100, high=500, log=False),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


__all__ = ["ExtraTreesSurvModel"]
