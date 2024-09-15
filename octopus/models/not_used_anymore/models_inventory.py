"""Model inventory and optuna parameters.

Models should follow the sklearn convention:
https://scikit-learn.org/stable/developers/develop.html
Template:
https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
"""

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ARDRegression, ElasticNet, LogisticRegression, Ridge
from sklearn.svm import SVR
from sksurv.ensemble import ExtraSurvivalTrees
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier, XGBRegressor

# model_inventory:
# "model": model,
# "feature_method": method to calculated number of used features
#                  ["internal", "shap", "permutation"]
# "parameters for feature method": 2
model_inventory = {
    "ExtraTreesClassifier": {
        "model": ExtraTreesClassifier,
        "feature_method": "internal",
        "ml_type": "classification",
    },
    "ExtraTreesRegressor": {
        "model": ExtraTreesRegressor,
        "feature_method": "internal",
        "ml_type": "regression",
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier,
        "feature_method": "internal",
        "ml_type": "classification",
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor,
        "feature_method": "internal",
        "ml_type": "regression",
    },
    "XGBRegressor": {
        "model": XGBRegressor,
        "feature_method": "internal",
        "ml_type": "regression",
    },
    "XGBClassifier": {
        "model": XGBClassifier,
        "feature_method": "internal",
        "ml_type": "classification",
    },
    "RidgeRegressor": {
        "model": Ridge,
        "feature_method": "shap",
        "ml_type": "regression",
    },
    "ElasticNetRegressor": {
        "model": ElasticNet,
        "feature_method": "shap",
        "ml_type": "regression",
    },
    "ARDRegressor": {
        "model": ARDRegression,
        "feature_method": "permutation",
        "n_repeats": 2,
        "ml_type": "regression",
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor,
        "feature_method": "internal",
        "ml_type": "regression",
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier,
        "feature_method": "internal",
        "ml_type": "classification",
    },
    "SvrRegressor": {
        "model": SVR,
        "feature_method": "permutation",
        "n_repeats": 2,
        "ml_type": "regression",
    },
    "ExtraTreesSurv": {
        "model": ExtraSurvivalTrees,
        "feature_method": "permutation",
        "n_repeats": 2,
        "ml_type": "timetoevent",
    },
    "CatBoostRegressor": {
        "model": CatBoostRegressor,
        "feature_method": "internal",
        "ml_type": "regression",
    },
    "CatBoostClassifier": {
        "model": CatBoostClassifier,
        "feature_method": "internal",
        "ml_type": "classification",
    },
    "TabPFNClassifier": {
        "model": TabPFNClassifier,
        "feature_method": "permutation",
        "n_repeats": 2,
        "ml_type": "classification",
    },
    "LogisticRegressionClassifier": {
        "model": LogisticRegression,
        "feature_method": "permutation",
        "n_repeats": 2,
        "ml_type": "classification",
    },
}
