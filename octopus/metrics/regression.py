"""Regression metrics."""

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MetricConfig
from .core import Metrics


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        float: RMSE value
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


@Metrics.register("R2")
def r2_metric() -> MetricConfig:
    """R2 metric configuration."""
    return MetricConfig(
        name="R2",
        metric_function=r2_score,
        ml_type="regression",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="r2",
    )


@Metrics.register("MAE")
def mae_metric() -> MetricConfig:
    """MAE metric configuration."""
    return MetricConfig(
        name="MAE",
        metric_function=mean_absolute_error,
        ml_type="regression",
        higher_is_better=False,
        prediction_type="predict",
        scorer_string="neg_mean_absolute_error",
    )


@Metrics.register("MSE")
def mse_metric() -> MetricConfig:
    """MSE metric configuration."""
    return MetricConfig(
        name="MSE",
        metric_function=mean_squared_error,
        ml_type="regression",
        higher_is_better=False,
        prediction_type="predict",
        scorer_string="neg_mean_squared_error",
    )


@Metrics.register("RMSE")
def rmse_metric() -> MetricConfig:
    """RMSE metric configuration."""
    return MetricConfig(
        name="RMSE",
        metric_function=root_mean_squared_error,
        ml_type="regression",
        higher_is_better=False,
        prediction_type="predict",
        scorer_string="neg_root_mean_squared_error",
    )
