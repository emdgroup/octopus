"""Classification models."""

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import ModelConfig
from .hyperparameter import Hyperparameter
from .registry import ModelRegistry
from .wrapper_models.GaussianProcessClassifier import GPClassifierWrapper


@ModelRegistry.register("ExtraTreesClassifier")
class ExtraTreesClassifierModel:
    """ExtraTrees classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="ExtraTreesClassifier",
            model_class=ExtraTreesClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(type="int", name="n_estimators", low=100, high=500, log=False),
                Hyperparameter(
                    type="categorical",
                    name="class_weight",
                    choices=[None, "balanced"],
                ),
                Hyperparameter(type="fixed", name="criterion", value="entropy"),
                Hyperparameter(type="fixed", name="bootstrap", value=True),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("HistGradientBoostingClassifier")
class HistGradientBoostingClassifierModel:
    """Histogram-based gradient boosting classification model class (scikit-learn 1.6.1)."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="HistGradientBoostingClassifier",
            model_class=HistGradientBoostingClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=0.01, high=0.3, log=True),
                Hyperparameter(type="int", name="max_iter", low=50, high=1000),
                Hyperparameter(type="int", name="max_leaf_nodes", low=7, high=256),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=200),
                Hyperparameter(type="int", name="max_bins", low=16, high=255),
                Hyperparameter(type="float", name="l2_regularization", low=0.0, high=10.0, log=False),
                Hyperparameter(type="fixed", name="loss", value="log_loss"),
            ],
            # HistGradientBoostingClassifier uses `random_state` for seeding (map model_seed -> "random_state")
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("GradientBoostingClassifier")
class GradientBoostingClassifierModel:
    """Gradient boosting classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="GradientBoostingClassifier",
            model_class=GradientBoostingClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=0.01, high=1, log=True),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=200),
                Hyperparameter(type="int", name="max_leaf_nodes", low=3, high=2047),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="int", name="n_estimators", low=30, high=500),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(type="fixed", name="loss", value="log_loss"),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("RandomForestClassifier")
class RandomForestClassifierModel:
    """Random forrest classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="RandomForestClassifier",
            model_class=RandomForestClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,  # maybe: False - check!
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(type="int", name="n_estimators", low=100, high=500, log=False),
                Hyperparameter(type="categorical", name="class_weight", choices=[None, "balanced"]),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("XGBClassifier")
class XGBClassifierModel:
    """XGBoost classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="XGBClassifier",
            model_class=XGBClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=False,  # Maybe True - check!
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=1e-4, high=0.3, log=True),
                Hyperparameter(type="int", name="min_child_weight", low=2, high=15),
                Hyperparameter(type="float", name="subsample", low=0.15, high=1.0),
                Hyperparameter(type="int", name="n_estimators", low=30, high=200),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="fixed", name="validate_parameters", value=True),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("CatBoostClassifier")
class CatBoostClassifierModel:
    """Catboost classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="CatBoostClassifier",
            model_class=CatBoostClassifier,
            ml_type="classification",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=1e-2, high=1e-1, log=True),
                Hyperparameter(type="int", name="depth", low=3, high=10),
                Hyperparameter(type="float", name="l2_leaf_reg", low=2, high=10),
                Hyperparameter(type="float", name="random_strength", low=2, high=10),
                Hyperparameter(type="float", name="rsm", low=0.1, high=1),
                Hyperparameter(type="fixed", name="iterations", value=1000),
                Hyperparameter(
                    type="categorical",
                    name="auto_class_weights",
                    choices=[None, "Balanced"],
                ),
                Hyperparameter(type="fixed", name="allow_writing_files", value=False),
                Hyperparameter(type="fixed", name="logging_level", value="Silent"),
                Hyperparameter(type="fixed", name="verbose", value=False),
                Hyperparameter(type="fixed", name="thread_count", value=1),
                Hyperparameter(type="fixed", name="task_type", value="CPU"),
            ],
            n_jobs="thread_count",
            model_seed="random_state",
        )


@ModelRegistry.register("LogisticRegressionClassifier")
class LogisticRegressionClassifierModel:
    """LogisticRegression classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="LogisticRegressionClassifier",
            model_class=LogisticRegression,
            ml_type="classification",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=True,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="int", name="max_iter", low=100, high=500),
                Hyperparameter(type="float", name="C", low=1e-2, high=100, log=True),
                Hyperparameter(type="float", name="tol", low=1e-4, high=1e-2, log=True),
                Hyperparameter(type="categorical", name="penalty", choices=["l2", "elasticnet", None]),
                Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
                Hyperparameter(type="categorical", name="class_weight", choices=[None, "balanced"]),
                Hyperparameter(type="fixed", name="solver", value="lbfgs"),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("TabPFNClassifier")
class TabPFNClassifierModel:
    """TabPFN classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        from octopus._optional.tabpfn import TabPFNClassifier  # noqa: PLC0415

        return ModelConfig(
            name="TabPFNClassifier",
            model_class=TabPFNClassifier,
            ml_type="classification",
            feature_method="constant",  # constant FI, disable constraint HPO
            n_repeats=2,
            chpo_compatible=True,
            scaler="StandardScaler",
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                Hyperparameter(type="fixed", name="n_estimators", value=4),
                Hyperparameter(type="fixed", name="softmax_temperature", value=0.9),
                Hyperparameter(type="fixed", name="balance_probabilities", value=True),
                Hyperparameter(type="fixed", name="average_before_softmax", value=False),
                Hyperparameter(type="fixed", name="device", value="cpu"),
                Hyperparameter(type="fixed", name="ignore_pretraining_limits", value=False),
                Hyperparameter(type="fixed", name="fit_mode", value="fit_preprocessors"),
                Hyperparameter(type="fixed", name="memory_saving_mode", value="auto"),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("GaussianProcessClassifier")
class GaussianProcessClassifierModel:
    """Gaussian process classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="GaussianProcessClassifier",
            model_class=GPClassifierWrapper,
            ml_type="classification",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=False,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(
                    type="categorical",
                    name="kernel",
                    choices=["RBF", "Matern", "RationalQuadratic"],
                ),
                Hyperparameter(
                    type="categorical",
                    name="optimizer",
                    choices=["fmin_l_bfgs_b", None],
                ),
                Hyperparameter(
                    type="int",
                    name="n_restarts_optimizer",
                    low=0,
                    high=10,
                    log=False,
                ),
                Hyperparameter(
                    type="int",
                    name="max_iter_predict",
                    low=50,
                    high=200,
                    log=False,
                ),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


__all__ = [
    "CatBoostClassifierModel",
    "ExtraTreesClassifierModel",
    "GradientBoostingClassifierModel",
    "LogisticRegressionClassifierModel",
    "RandomForestClassifierModel",
    "TabPFNClassifierModel",
    "XGBClassifierModel",
    "GaussianProcessClassifierModel",
    "HistGradientBoostingClassifierModel",
]
