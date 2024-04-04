"""Model inventory and optuna parameters.

Models should follow the sklearn convention:
https://scikit-learn.org/stable/developers/develop.html
Template:
https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ARDRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from sksurv.ensemble import ExtraSurvivalTrees
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
    },
    "ExtraTreesRegressor": {"model": ExtraTreesRegressor, "feature_method": "internal"},
    "RandomForestClassifier": {
        "model": RandomForestClassifier,
        "feature_method": "internal",
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor,
        "feature_method": "internal",
    },
    "XGBRegressor": {"model": XGBRegressor, "feature_method": "internal"},
    "XGBClassifier": {"model": XGBClassifier, "feature_method": "internal"},
    "RidgeRegressor": {"model": Ridge, "feature_method": "shap"},
    "ElasticNetRegressor": {"model": ElasticNet, "feature_method": "shap"},
    "ARDRegressor": {
        "model": ARDRegression,
        "feature_method": "permutation",
        "n_repeats": 2,
    },
    "ExtraTreesSurv": {
        "model": ExtraSurvivalTrees,
        "feature_method": "permutation",
        "n_repeats": 2,
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor,
        "feature_method": "internal",
    },
    "SvrRegressor": {
        "model": SVR,
        "feature_method": "permutation",
        "n_repeats": 2,
    },
}
