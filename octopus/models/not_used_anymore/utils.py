"""Model utility functions."""

import copy


def create_trialparams_from_config(trial, hp_settings, ml_model_type):
    """Create trial parameters from optuna config.

    We support int/float/categorical/fixed.
    Fixed is used to overwrite default model values
    """
    params = dict()
    for dtype, config in hp_settings:
        # check if log and step are specified
        if all(key in config for key in ["log", "step"]):
            raise ValueError("Optuna config must not contain log and step parameter")
        # make trial suggest name unique for each model type
        # add prefix to cfg["name"]
        cfg = copy.deepcopy(config)
        parameter_name = cfg["name"]
        cfg["name"] = cfg["name"] + "_" + ml_model_type

        if dtype == "int":
            params[parameter_name] = trial.suggest_int(**cfg)
        elif dtype == "float":
            params[parameter_name] = trial.suggest_float(**cfg)
        elif dtype == "categorical":
            params[parameter_name] = trial.suggest_categorical(
                name=cfg["name"], choices=cfg["choices"]
            )
        elif dtype == "fixed":  # overwrite default model values
            params[parameter_name] = cfg["value"]
        else:
            raise ValueError("HP type not supported")

    return params
