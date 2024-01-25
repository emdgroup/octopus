"""Model inventory and optuna parameters.

Models should follow the sklearn convention:
https://scikit-learn.org/stable/developers/develop.html
Template:
https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
"""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

from octopus.models.extratree import (
    extratree_class_parameters,
    extratree_reg_parameters,
)
from octopus.models.rf import rf_class_parameters, rf_reg_parameters
from octopus.models.xgb import xgb_class_parameters, xgb_reg_parameters

model_inventory = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "XGBClassifier": XGBClassifier,
}

parameters_inventory = {
    "ExtraTreesClassifier": extratree_class_parameters,
    "ExtraTreesRegressor": extratree_reg_parameters,
    "RandomForestClassifier": rf_class_parameters,
    "RandomForestRegressor": rf_reg_parameters,
    "XGBRegressor": xgb_class_parameters,
    "XGBClassifier": xgb_reg_parameters,
}
