"""Regression models."""

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ARDRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from octopus.models.machine_learning.config import ModelConfig
from octopus.models.machine_learning.hyperparameter import Hyperparameter


def get_regression_models():
    """Return a list of ModelConfig objects for regression models.

    Each ModelConfig object contains the configuration for a specific regression model,
    including the model class, hyperparameters, and other settings.

    Returns:
        List[ModelConfig]: A list of ModelConfig objects for regression models.
    """
    return [
        ModelConfig(
            name="RandomForestRegressor",
            model_class=RandomForestRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="int", name="n_estimators", low=100, high=500),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
            ],
            translate={
                "n_jobs": "n_jobs",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="XGBRegressor",
            model_class=XGBRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="learning_rate", low=1e-4, high=0.3, log=True
                ),
                Hyperparameter(type="int", name="min_child_weight", low=2, high=15),
                Hyperparameter(type="float", name="subsample", low=0.15, high=1.0),
                Hyperparameter(type="int", name="n_estimators", low=30, high=500),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="fixed", name="validate_parameters", value=True),
                Hyperparameter(type="float", name="lambda", low=1e-8, high=1, log=True),
            ],
            translate={
                "n_jobs": "n_jobs",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="ExtraTreesRegressor",
            model_class=ExtraTreesRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="learning_rate", low=1e-4, high=0.3, log=True
                ),
                Hyperparameter(type="int", name="min_child_weight", low=2, high=15),
                Hyperparameter(type="float", name="subsample", low=0.15, high=1.0),
                Hyperparameter(type="int", name="n_estimators", low=30, high=500),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="fixed", name="validate_parameters", value=True),
                Hyperparameter(type="float", name="lambda", low=1e-8, high=1, log=True),
            ],
            translate={
                "n_jobs": "n_jobs",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="RidgeRegressor",
            model_class=Ridge,
            ml_type="regression",
            feature_method="shap",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="alpha", low=1e-5, high=1e5, log=True
                ),
                Hyperparameter(
                    type="categorical", name="fit_intercept", choices=[True, False]
                ),
                Hyperparameter(type="fixed", name="solver", value="svd"),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="ElasticNetRegressor",
            model_class=ElasticNet,
            ml_type="regression",
            feature_method="shap",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="alpha", low=1e-10, high=1e2, log=True
                ),
                Hyperparameter(type="float", name="l1_ratio", low=0, high=1, log=False),
                Hyperparameter(
                    type="categorical", name="fit_intercept", choices=[True, False]
                ),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
                Hyperparameter(type="fixed", name="max_iter", value=4000),
                Hyperparameter(type="fixed", name="selection", value="random"),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="ARDRegressor",
            model_class=ARDRegression,
            ml_type="regression",
            feature_method="permutation",
            n_repeats=2,
            hyperparameters=[
                Hyperparameter(
                    type="float", name="alpha_1", low=1e-10, high=1e-3, log=True
                ),
                Hyperparameter(
                    type="float", name="alpha_2", low=1e-10, high=1e-3, log=True
                ),
                Hyperparameter(
                    type="float", name="lambda_1", low=1e-10, high=1e-3, log=True
                ),
                Hyperparameter(
                    type="float", name="lambda_2", low=1e-10, high=1e-3, log=True
                ),
                Hyperparameter(
                    type="float", name="threshold_lambda", low=1e3, high=1e5, log=True
                ),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
                Hyperparameter(type="fixed", name="fit_intercept", value=True),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "NA",
            },
        ),
        ModelConfig(
            name="GradientBoostingRegressor",
            model_class=GradientBoostingRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="learning_rate", low=0.01, high=1, log=True
                ),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=200),
                Hyperparameter(type="int", name="max_leaf_nodes", low=3, high=2047),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="int", name="n_estimators", low=30, high=500),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(type="fixed", name="loss", value="squared_error"),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="SvrRegressor",
            model_class=SVR,
            ml_type="regression",
            feature_method="permutation",
            n_repeats=2,
            hyperparameters=[
                Hyperparameter(
                    type="float", name="C", low=0.03125, high=32768, log=True
                ),
                Hyperparameter(
                    type="float", name="epsilon", low=0.001, high=1, log=True
                ),
                Hyperparameter(type="float", name="tol", low=1e-5, high=1e-1, log=True),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "NA",
            },
        ),
        ModelConfig(
            name="CatBoostRegressor",
            model_class=CatBoostRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="learning_rate", low=1e-3, high=1e-1, log=True
                ),
                Hyperparameter(type="int", name="depth", low=3, high=10),
                Hyperparameter(type="float", name="l2_leaf_reg", low=2, high=10),
                Hyperparameter(type="float", name="random_strength", low=2, high=10),
                Hyperparameter(type="float", name="rsm", low=0.1, high=1),
                Hyperparameter(type="fixed", name="iterations", value=500),
                Hyperparameter(type="fixed", name="allow_writing_files", value=False),
                Hyperparameter(type="fixed", name="verbose", value=250),
            ],
            translate={
                "n_jobs": "thread_count",
                "model_seed": "random_state",
            },
        ),
        ModelConfig(
            name="GaussianProcessRegressor",
            model_class=GaussianProcessRegressor,
            ml_type="regression",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="alpha", low=1e-10, high=1e-1, log=True
                ),
                Hyperparameter(
                    type="categorical", name="normalize_y", choices=[True, False]
                ),
                Hyperparameter(
                    type="categorical",
                    name="optimizer",
                    choices=["fmin_l_bfgs_b", None],
                ),
                Hyperparameter(
                    type="int", name="n_restarts_optimizer", low=0, high=10, log=False
                ),
            ],
            translate={
                "n_jobs": "NA",
                "model_seed": "random_state",
            },
        ),
    ]
