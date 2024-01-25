"""Model: ExtraTrees, parameters."""
from octopus.models.utils import create_trialparams_from_config


def extratree_class_parameters(trial, fixed_params=None, optuna_settings=None):
    """Suggest function for ExtraTreeClassifier.

    optuna_params: dict describing parameters to be optimized
    by Optuna, see: optuna_default. Dict needs to contain all
    parameters to be optimized.

    fixed_params: dict describing general parameters not to be optimized
    by Optuna. Those settings are translated into the parameters
    compatible with the specific model.
    """
    fixed_default = {
        "ml_jobs": 1,
        "ml_seed": 0,
    }

    optuna_default = [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
        ("categorical", {"name": "class_weight", "choices": [None, "balanced"]}),
    ]
    if fixed_params is None:
        fixed_params = fixed_default

    if optuna_settings is None:
        optuna_settings = optuna_default

    # model parameters set by optuna
    model_params = create_trialparams_from_config(trial, optuna_settings)

    # add fixed model parameters
    # model specific settings
    model_params["n_jobs"] = fixed_params["ml_jobs"]
    model_params["random_state"] = fixed_params["ml_seed"]

    return model_params


def extratree_reg_parameters(trial, fixed_params=None, optuna_settings=None):
    """Suggest function for ExtraTreeRegressor.

    optuna_params: dict describing parameters to be optimized
    by Optuna, see: optuna_default. Dict needs to contain all
    parameters to be optimized.

    fixed_params: dict describing general parameters not to be optimized
    by Optuna. Those settings are translated into the parameters
    compatible with the specific model.
    """
    fixed_default = {
        "ml_jobs": 1,
        "ml_seed": 0,
    }

    optuna_default = [
        ("int", {"name": "max_depth", "low": 2, "high": 32}),
        ("int", {"name": "min_samples_split", "low": 2, "high": 100}),
        ("int", {"name": "min_samples_leaf", "low": 1, "high": 50}),
        ("float", {"name": "max_features", "low": 0.1, "high": 1}),
        ("int", {"name": "n_estimators", "low": 100, "high": 500, "log": False}),
    ]
    if fixed_params is None:
        fixed_params = fixed_default

    if optuna_settings is None:
        optuna_settings = optuna_default

    # model parameters set by optuna
    model_params = create_trialparams_from_config(trial, optuna_settings)

    # add fixed model parameters
    # model specific settings
    model_params["n_jobs"] = fixed_params["ml_jobs"]
    model_params["random_state"] = fixed_params["ml_seed"]

    return model_params
