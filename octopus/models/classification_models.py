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
from .hyperparameter import CategoricalHyperparameter, FixedHyperparameter, FloatHyperparameter, IntHyperparameter
from .registry import ModelRegistry
from .wrapper_models.GaussianProcessClassifier import GPClassifierWrapper
from .wrapper_models.TabularNNClassifier import TabularNNClassifier


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
                IntHyperparameter(name="max_depth", low=2, high=32),
                IntHyperparameter(name="min_samples_split", low=2, high=100),
                IntHyperparameter(name="min_samples_leaf", low=1, high=50),
                FloatHyperparameter(name="max_features", low=0.1, high=1),
                IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
                CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
                FixedHyperparameter(name="criterion", value="entropy"),
                FixedHyperparameter(name="bootstrap", value=True),
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
                FloatHyperparameter(name="learning_rate", low=0.01, high=0.3, log=True),
                IntHyperparameter(name="max_iter", low=50, high=1000),
                IntHyperparameter(name="max_leaf_nodes", low=7, high=256),
                IntHyperparameter(name="min_samples_leaf", low=1, high=200),
                IntHyperparameter(name="max_bins", low=16, high=255),
                FloatHyperparameter(name="l2_regularization", low=0.0, high=10.0, log=False),
                FixedHyperparameter(name="loss", value="log_loss"),
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
                FloatHyperparameter(name="learning_rate", low=0.01, high=1, log=True),
                IntHyperparameter(name="min_samples_leaf", low=1, high=200),
                IntHyperparameter(name="max_leaf_nodes", low=3, high=2047),
                IntHyperparameter(name="max_depth", low=3, high=9, step=2),
                IntHyperparameter(name="n_estimators", low=30, high=500),
                FloatHyperparameter(name="max_features", low=0.1, high=1),
                FixedHyperparameter(name="loss", value="log_loss"),
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
                IntHyperparameter(name="max_depth", low=2, high=32),
                IntHyperparameter(name="min_samples_split", low=2, high=100),
                IntHyperparameter(name="min_samples_leaf", low=1, high=50),
                FloatHyperparameter(name="max_features", low=0.1, high=1),
                IntHyperparameter(name="n_estimators", low=100, high=500, log=False),
                CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
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
                FloatHyperparameter(name="learning_rate", low=1e-4, high=0.3, log=True),
                IntHyperparameter(name="min_child_weight", low=2, high=15),
                FloatHyperparameter(name="subsample", low=0.15, high=1.0),
                IntHyperparameter(name="n_estimators", low=30, high=200),
                IntHyperparameter(name="max_depth", low=3, high=9, step=2),
                FixedHyperparameter(name="validate_parameters", value=True),
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
                FloatHyperparameter(name="learning_rate", low=1e-2, high=1e-1, log=True),
                IntHyperparameter(name="depth", low=3, high=10),
                FloatHyperparameter(name="l2_leaf_reg", low=2, high=10),
                FloatHyperparameter(name="random_strength", low=2, high=10),
                FloatHyperparameter(name="rsm", low=0.1, high=1),
                FixedHyperparameter(name="iterations", value=1000),
                CategoricalHyperparameter(name="auto_class_weights", choices=[None, "Balanced"]),
                FixedHyperparameter(name="allow_writing_files", value=False),
                FixedHyperparameter(name="logging_level", value="Silent"),
                FixedHyperparameter(name="thread_count", value=1),
                FixedHyperparameter(name="task_type", value="CPU"),
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
                IntHyperparameter(name="max_iter", low=100, high=500),
                FloatHyperparameter(name="C", low=1e-2, high=100, log=True),
                FloatHyperparameter(name="tol", low=1e-4, high=1e-2, log=True),
                CategoricalHyperparameter(name="penalty", choices=["l2", None]),
                CategoricalHyperparameter(name="fit_intercept", choices=[True, False]),
                CategoricalHyperparameter(name="class_weight", choices=[None, "balanced"]),
                FixedHyperparameter(name="solver", value="lbfgs"),
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
        from octopus._optional.tabpfn_utils import get_tabpfn_model_path  # noqa: PLC0415

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
                FixedHyperparameter(name="n_estimators", value=4),
                FixedHyperparameter(name="softmax_temperature", value=0.9),
                FixedHyperparameter(name="balance_probabilities", value=True),
                FixedHyperparameter(name="average_before_softmax", value=False),
                FixedHyperparameter(name="device", value="cpu"),
                FixedHyperparameter(name="ignore_pretraining_limits", value=False),
                FixedHyperparameter(name="fit_mode", value="fit_preprocessors"),
                FixedHyperparameter(name="memory_saving_mode", value="auto"),
                FixedHyperparameter(name="model_path", value=get_tabpfn_model_path("classifier")),
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
                CategoricalHyperparameter(name="kernel", choices=["RBF", "Matern", "RationalQuadratic"]),
                CategoricalHyperparameter(name="optimizer", choices=["fmin_l_bfgs_b", None]),
                IntHyperparameter(name="n_restarts_optimizer", low=0, high=10, log=False),
                IntHyperparameter(name="max_iter_predict", low=50, high=200, log=False),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("TabularNNClassifier")
class TabularNNClassifierModel:
    """Tabular Neural Network classification model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="TabularNNClassifier",
            model_class=TabularNNClassifier,
            ml_type="classification",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=False,
            scaler="StandardScaler",
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                CategoricalHyperparameter(
                    name="hidden_sizes",
                    choices=[
                        [512, 256, 128],
                        [512, 256],
                        [512, 128],
                        [256, 256, 128],
                        [256, 128, 64],
                        [256, 128],
                        [256, 64],
                        [128, 128, 64],
                        [128, 64],
                        [128, 32],
                    ],
                ),
                FloatHyperparameter(name="dropout", low=0.0, high=0.5),
                FloatHyperparameter(name="learning_rate", low=1e-5, high=1e-2, log=True),
                FixedHyperparameter(name="weight_decay", value=1e-5),
                FixedHyperparameter(name="activation", value="elu"),
                FixedHyperparameter(name="optimizer", value="adamw"),
                CategoricalHyperparameter(name="batch_size", choices=[32, 64, 128, 256]),
                FixedHyperparameter(name="epochs", value=200),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


__all__ = [
    "CatBoostClassifierModel",
    "ExtraTreesClassifierModel",
    "GaussianProcessClassifierModel",
    "GradientBoostingClassifierModel",
    "HistGradientBoostingClassifierModel",
    "LogisticRegressionClassifierModel",
    "RandomForestClassifierModel",
    "TabPFNClassifierModel",
    "TabularNNClassifierModel",
    "XGBClassifierModel",
]
