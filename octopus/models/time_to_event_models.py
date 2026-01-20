"""Time to event models."""

from .config import ModelConfig
from .hyperparameter import FloatHyperparameter, IntHyperparameter
from .registry import ModelRegistry


@ModelRegistry.register("ExtraTreesSurv")
class ExtraTreesSurvModel:
    """Extra Tree Survival Model."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        from octopus._optional.survival import ExtraSurvivalTrees  # noqa: PLC0415

        return ModelConfig(
            name="ExtraTreesSurv",
            model_class=ExtraSurvivalTrees,
            ml_type="timetoevent",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                IntHyperparameter(name="max_depth", low=2, high=32),
                IntHyperparameter(name="min_samples_split", low=2, high=100),
                IntHyperparameter(name="min_samples_leaf", low=1, high=50),
                FloatHyperparameter(name="max_features", low=0.1, high=1),
                IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


__all__ = ["ExtraTreesSurvModel"]
