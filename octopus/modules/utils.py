"""Helper functions."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_score(metric: str, y_true: np.array, y_pred: np.array) -> float:
    """Calculate the specified metric for the given true and predicted values.

    Args:
        metric: The name of the metric to compute.
            Valid options are 'MAE', 'R2', and 'MSE'.
        y_true: An array of true values for the model's predictions
            to be evaluated against.
        y_pred: An array of predicted values to evaluate.

    Returns:
        The computed score of the specified metric.

    Raises:
        ValueError: If the metric provided is not recognized or supported.
    """
    if metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "R2":
        return r2_score(y_true, y_pred)
    elif metric == "MSE":
        return mean_squared_error(y_true, y_pred)
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics are 'MAE', 'R2', 'MSE'."
        )


def optuna_direction(metric: str) -> str:
    """Determine the direction (minimize or maximize) for the given metric.

    Args:
        metric: The name of the performance metric, which should be one of
                        'MAE', 'MSE', 'R2', 'ACC', 'ACCBAL', 'LOGLOSS', 'AUCROC', 'CI'.

    Returns:
        The optimization direction, either 'minimize' or 'maximize'.

    Raises:
        ValueError: If the metric provided is not recognized.
    """
    direction = {
        "MAE": "minimize",
        "MSE": "minimize",
        "R2": "maximize",
        "ACC": "maximize",
        "ACCBAL": "maximize",
        "LOGLOSS": "minimize",
        "AUCROC": "maximize",
        "CI": "maximize",
    }

    if metric not in direction:
        raise ValueError(
            f"""Unknown metric '{metric}'.
            Available metrics are: {list(direction.keys())}."""
        )

    return direction[metric]
