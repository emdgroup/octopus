"""Classification models."""

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from octopus.models.config import ModelConfig
from octopus.models.hyperparameter import Hyperparameter


def get_classification_models():
    """Return a list of ModelConfig objects for classification models.

    Each ModelConfig object contains the configuration for a specific classification
    model, including the model class, hyperparameters, and other settings.

    Returns:
        List[ModelConfig]: A list of ModelConfig objects for classification models.
    """
    return [
        ModelConfig(
            name="ExtraTreesClassifier",
            model_class=ExtraTreesClassifier,
            ml_type="classification",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(
                    type="int", name="n_estimators", low=100, high=500, log=False
                ),
                Hyperparameter(
                    type="categorical", name="class_weight", choices=[None, "balanced"]
                ),
                Hyperparameter(type="fixed", name="criterion", value="entropy"),
                Hyperparameter(type="fixed", name="bootstrap", value=True),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        ),
        ModelConfig(
            name="RandomForestClassifier",
            model_class=RandomForestClassifier,
            ml_type="classification",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(type="int", name="max_depth", low=2, high=32),
                Hyperparameter(type="int", name="min_samples_split", low=2, high=100),
                Hyperparameter(type="int", name="min_samples_leaf", low=1, high=50),
                Hyperparameter(type="float", name="max_features", low=0.1, high=1),
                Hyperparameter(
                    type="int", name="n_estimators", low=100, high=500, log=False
                ),
                Hyperparameter(
                    type="categorical", name="class_weight", choices=[None, "balanced"]
                ),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        ),
        ModelConfig(
            name="XGBClassifier",
            model_class=XGBClassifier,
            ml_type="classification",
            feature_method="internal",
            hyperparameters=[
                Hyperparameter(
                    type="float", name="learning_rate", low=1e-4, high=0.3, log=True
                ),
                Hyperparameter(type="int", name="min_child_weight", low=2, high=15),
                Hyperparameter(type="float", name="subsample", low=0.15, high=1.0),
                Hyperparameter(type="int", name="n_estimators", low=30, high=200),
                Hyperparameter(type="int", name="max_depth", low=3, high=9, step=2),
                Hyperparameter(type="fixed", name="validate_parameters", value=True),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        ),
        ModelConfig(
            name="GradientBoostingClassifier",
            model_class=GradientBoostingClassifier,
            ml_type="classification",
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
            n_jobs="n_jobs",
            model_seed="random_state",
        ),
        ModelConfig(
            name="CatBoostClassifier",
            model_class=CatBoostClassifier,
            ml_type="classification",
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
                Hyperparameter(
                    type="categorical",
                    name="auto_class_weights",
                    choices=[None, "Balanced"],
                ),
                Hyperparameter(type="fixed", name="allow_writing_files", value=False),
                Hyperparameter(type="fixed", name="verbose", value=250),
            ],
            n_jobs="thread_count",
            model_seed="random_state",
        ),
        ModelConfig(
            name="TabPFNClassifier",
            model_class=TabPFNClassifier,
            ml_type="classification",
            feature_method="permutation",
            n_repeats=2,
            hyperparameters=[
                Hyperparameter(
                    type="fixed", name="N_ensemble_configurations", value=16
                ),
                Hyperparameter(type="fixed", name="subsample_features", value=False),
            ],
            n_jobs=None,
            model_seed="seed",
        ),
        ModelConfig(
            name="LogisticRegressionClassifier",
            model_class=LogisticRegression,
            ml_type="classification",
            feature_method="permutation",
            n_repeats=2,
            hyperparameters=[
                Hyperparameter(type="int", name="max_iter", low=100, high=500),
                Hyperparameter(type="float", name="C", low=1e-2, high=100, log=True),
                Hyperparameter(type="float", name="tol", low=1e-4, high=1e-2, log=True),
                Hyperparameter(
                    type="categorical", name="penalty", choices=["l2", "none"]
                ),
                Hyperparameter(
                    type="categorical", name="fit_intercept", choices=[True, False]
                ),
                Hyperparameter(
                    type="categorical", name="class_weight", choices=[None, "balanced"]
                ),
                Hyperparameter(type="fixed", name="solver", value="lbfgs"),
            ],
            n_jobs="n_jobs",
            model_seed="random_state",
        ),
    ]
