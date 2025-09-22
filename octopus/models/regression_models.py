"""Regression models."""

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ARDRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .config import ModelConfig
from .hyperparameter import Hyperparameter
from .registry import ModelRegistry
from .wrapper_models.GaussianProcessRegressor import GPRegressorWrapper


@ModelRegistry.register("ARDRegressor")
class ARDRegressorModel:
    """ARD regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="ARDRegressor",
            model_class=ARDRegression,
            ml_type="regression",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=False,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="float", name="alpha_1", low=1e-10, high=1e-3, log=True),
                Hyperparameter(type="float", name="alpha_2", low=1e-10, high=1e-3, log=True),
                Hyperparameter(type="float", name="lambda_1", low=1e-10, high=1e-3, log=True),
                Hyperparameter(type="float", name="lambda_2", low=1e-10, high=1e-3, log=True),
                Hyperparameter(type="float", name="threshold_lambda", low=1e3, high=1e5, log=True),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
                Hyperparameter(type="fixed", name="fit_intercept", value=True),
            ],
            n_jobs=None,
            model_seed=None,
        )


@ModelRegistry.register("CatBoostRegressor")
class CatBoostRegressorModel:
    """Cat boost regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="CatBoostRegressor",
            model_class=CatBoostRegressor,
            ml_type="regression",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=1e-3, high=1e-1, log=True),
                Hyperparameter(type="int", name="depth", low=3, high=10),
                Hyperparameter(type="float", name="l2_leaf_reg", low=2, high=10),
                Hyperparameter(type="float", name="random_strength", low=2, high=10),
                Hyperparameter(type="float", name="rsm", low=0.1, high=1),
                Hyperparameter(type="fixed", name="iterations", value=500),
                Hyperparameter(type="fixed", name="allow_writing_files", value=False),
                Hyperparameter(type="fixed", name="logging_level", value="Silent"),
                Hyperparameter(type="fixed", name="verbose", value=False),
                Hyperparameter(type="fixed", name="thread_count", value=1),
                Hyperparameter(type="fixed", name="task_type", value="CPU"),
            ],
            n_jobs="thread_count",
            model_seed="random_state",
        )


@ModelRegistry.register("ElasticNetRegressor")
class ElasticNetRegressorModel:
    """ElasticNet regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="ElasticNetRegressor",
            model_class=ElasticNet,
            ml_type="regression",
            feature_method="shap",
            chpo_compatible=True,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="float", name="alpha", low=1e-10, high=1e2, log=True),
                Hyperparameter(type="float", name="l1_ratio", low=0, high=1, log=False),
                Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
                Hyperparameter(type="fixed", name="max_iter", value=4000),
                Hyperparameter(type="fixed", name="selection", value="random"),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("ExtraTreesRegressor")
class ExtraTreesRegressorModel:
    """ExtraTrees regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="ExtraTreesRegressor",
            model_class=ExtraTreesRegressor,
            ml_type="regression",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="int", name="n_estimators", low=100, high=500, log=False),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("GaussianProcessRegressor")
class GaussianProcessRegressorModel:
    """Gaussian process regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="GaussianProcessRegressor",
            model_class=GPRegressorWrapper,
            ml_type="regression",
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
                Hyperparameter(type="float", name="alpha", low=1e-10, high=1e-1, log=True),
                Hyperparameter(type="float", name="alpha", low=1e-10, high=1e-1, log=True),
                Hyperparameter(type="categorical", name="normalize_y", choices=[True, False]),
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
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("GradientBoostingRegressor")
class GradientBoostingRegressorModel:
    """Gradient boost regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="GradientBoostingRegressor",
            model_class=GradientBoostingRegressor,
            ml_type="regression",
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
                Hyperparameter(type="fixed", name="loss", value="squared_error"),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("RandomForestRegressor")
class RandomForestRegressorModel:
    """Random forrest regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="RandomForestRegressor",
            model_class=RandomForestRegressor,
            ml_type="regression",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=True,  # maybe: False -- check!
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="int", name="n_estimators", low=100, high=500),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("RidgeRegressor")
class RidgeRegressorModel:
    """Ridge regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="RidgeRegressor",
            model_class=Ridge,
            ml_type="regression",
            feature_method="shap",
            chpo_compatible=False,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="float", name="alpha", low=1e-5, high=1e5, log=True),
                Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
                Hyperparameter(type="fixed", name="solver", value="svd"),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


@ModelRegistry.register("SvrRegressor")
class SvrRegressorModel:
    """Svr regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="SvrRegressor",
            model_class=SVR,
            ml_type="regression",
            feature_method="permutation",
            n_repeats=2,
            chpo_compatible=False,
            scaler="StandardScaler",
            imputation_required=True,
            categorical_enabled=False,
            hyperparameters=[
                Hyperparameter(type="float", name="C", low=0.03125, high=32768, log=True),
                Hyperparameter(type="float", name="epsilon", low=0.001, high=1, log=True),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
            ],
            n_jobs=None,
            model_seed=None,
        )


@ModelRegistry.register("TabPFNRegressor")
class TabPFNRegressorModel:
    """TabPFN regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        from octopus._optional.tabpfn import TabPFNRegressor  # noqa: PLC0415

        return ModelConfig(
            name="TabPFNRegressor",
            model_class=TabPFNRegressor,
            ml_type="regression",
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


@ModelRegistry.register("XGBRegressor")
class XGBRegressorModel:
    """XGBoost regression model class."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="XGBRegressor",
            model_class=XGBRegressor,
            ml_type="regression",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=False,  # maybe:True -- check!
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=1e-4, high=0.3, log=True),
                Hyperparameter(type="int", name="min_child_weight", low=2, high=15),
                Hyperparameter(type="float", name="subsample", low=0.15, high=1.0),
                Hyperparameter(type="int", name="n_estimators", low=30, high=500),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="fixed", name="validate_parameters", value=True),
                Hyperparameter(type="float", name="lambda", low=1e-8, high=1, log=True),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        )


@ModelRegistry.register("HistGradientBoostingRegressor")
class HistGradientBoostingRegressorModel:
    """Histogram-based gradient boosting regression model class (scikit-learn 1.6.1)."""

    @staticmethod
    def get_model_config():
        """Get model config."""
        return ModelConfig(
            name="HistGradientBoostingRegressor",
            model_class=HistGradientBoostingRegressor,
            ml_type="regression",
            feature_method="internal",
            chpo_compatible=True,
            scaler=None,
            imputation_required=False,
            categorical_enabled=True,
            hyperparameters=[
                Hyperparameter(type="float", name="learning_rate", low=0.01, high=0.3, log=True),
                Hyperparameter(type="int", name="max_iter", low=50, high=1000),
                Hyperparameter(type="int", name="max_leaf_nodes", low=7, high=256),
                Hyperparameter(type="float", name="l2_regularization", low=1e-6, high=10.0, log=True),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=200),
                Hyperparameter(type="int", name="max_bins", low=16, high=255),
                Hyperparameter(type="fixed", name="loss", value="squared_error"),
            ],
            n_jobs=None,
            model_seed="random_state",
        )


__all__ = [
    "ARDRegressorModel",
    "CatBoostRegressorModel",
    "ElasticNetRegressorModel",
    "ExtraTreesRegressorModel",
    "GaussianProcessRegressorModel",
    "GradientBoostingRegressorModel",
    "HistGradientBoostingRegressorModel",
    "RandomForestRegressorModel",
    "RidgeRegressorModel",
    "SvrRegressorModel",
    "TabPFNRegressorModel",
    "XGBRegressorModel",
]
