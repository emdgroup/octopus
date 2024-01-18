"""Model inventory and optuna parameters.

Models should follow the sklearn convention:
https://scikit-learn.org/stable/developers/develop.html
Template:
https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py
"""
from sklearn.ensemble import ExtraTreesClassifier

model_inventory = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
}


def extratreeclass_suggest(trial):
    """Suggest function for ExtraTreeClassifier."""
    max_depth = (trial.suggest_int("max_depth", 2, 32),)
    min_samples_split = (trial.suggest_int("min_samples_split", 2, 100),)
    min_samples_leaf = (trial.suggest_int("min_samples_leaf", 1, 50),)
    max_features = (trial.suggest_float("max_features", 0.1, 1),)
    nestimators = (trial.suggest_int("n_estimators", 100, 500, log=False),)
    params = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "nestimators": nestimators,
    }
    return params


optuna_inventory = {
    "ExtraTreesClassifier": extratreeclass_suggest,
}
