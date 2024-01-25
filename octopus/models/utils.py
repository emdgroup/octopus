"""Model utility functions."""


def create_trialparams_from_config(trial, hps):
    """Create trial parameters from optuna config.

    We support int/float/categorical/fixed.
    Fixed is used to overwrite default model values
    """
    params = dict()
    for dtype, config in hps:
        # check if log and step are specified
        if all(key in config for key in ["log", "step"]):
            raise ValueError("Optuna config must not contain log and step parameter")
        if dtype == "int":
            params[config["name"]] = trial.suggest_int(**config)
        elif dtype == "float":
            params[config["name"]] = trial.suggest_float(**config)
        elif dtype == "categorical":
            params[config["name"]] = trial.suggest_categorical(
                name=config["name"], choices=config["choices"]
            )
        elif dtype == "fixed":
            params[config["name"]] = config["value"]
        else:
            raise ValueError("HP type not supported")
    return params
