"""Helper functions for analytics."""

import pandas as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def get_score(metric: str, y_true: np.array, y_pred: np.array) -> float:
    """Calculate selected metric."""
    if metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "R2":
        return r2_score(y_true, y_pred)
    elif metric == "MSE":
        return mean_squared_error(y_true, y_pred)
