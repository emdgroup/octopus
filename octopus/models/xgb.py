"""Model: xgboost, parameters."""

from octopus.models.utils import create_trialparams_from_config


def xgb_class_parameters(trial, fixed_params=None, optuna_settings=None):
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
        "class_weights": False,
    }

    optuna_default = [
        ("float", {"name": "learning_rate", "low": 1e-4, "high": 0.3, "log": True}),
        ("int", {"name": "min_child_weight", "low": 2, "high": 15}),
        ("float", {"name": "subsample", "low": 0.15, "high": 1.0}),
        ("int", {"name": "n_estimators", "low": 30, "high": 200}),
        ("int", {"name": "max_depth", "low": 3, "high": 9, "step": 2}),
        # ("float", {"name": "lambda", "low": 1e-8, "high": 1, "log": True}),
        # missing: pos_class_weight
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


def xgb_reg_parameters(trial, fixed_params=None, optuna_settings=None):
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
        ("float", {"name": "learning_rate", "low": 1e-4, "high": 0.3, "log": True}),
        ("int", {"name": "min_child_weight", "low": 2, "high": 15}),
        ("float", {"name": "subsample", "low": 0.15, "high": 1.0}),
        ("int", {"name": "n_estimators", "low": 30, "high": 200}),
        ("int", {"name": "max_depth", "low": 3, "high": 9, "step": 2}),
        # ("float", {"name": "lambda", "low": 1e-8, "high": 1, "log": True}),
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
