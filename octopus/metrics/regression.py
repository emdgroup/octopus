"""Regression metrics."""

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MetricConfig
from .registry import MetricRegistry


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        float: RMSE value
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


@MetricRegistry.register("R2")
class R2Metric:
    """R2 metric class."""

    @staticmethod
    def get_metric_config():
        """Get metric config."""
        return MetricConfig(
            name="R2",
            metric_function=r2_score,
            ml_type="regression",
            higher_is_better=True,
            prediction_type="predict",
            scorer_string="r2",
        )


@MetricRegistry.register("MAE")
class MAEMetric:
    """MAE metric class."""

    @staticmethod
    def get_metric_config():
        """Get metric config."""
        return MetricConfig(
            name="MAE",
            metric_function=mean_absolute_error,
            ml_type="regression",
            higher_is_better=False,
            prediction_type="predict",
            scorer_string="neg_mean_absolute_error",
        )


@MetricRegistry.register("MSE")
class MSEMetric:
    """MSE metric class."""

    @staticmethod
    def get_metric_config():
        """Get metric config."""
        return MetricConfig(
            name="MSE",
            metric_function=mean_squared_error,
            ml_type="regression",
            higher_is_better=False,
            prediction_type="predict",
            scorer_string="neg_mean_squared_error",
        )


@MetricRegistry.register("RMSE")
class RMSEMetric:
    """RMSE metric class."""

    @staticmethod
    def get_metric_config():
        """Get metric config."""
        return MetricConfig(
            name="RMSE",
            metric_function=root_mean_squared_error,
            ml_type="regression",
            higher_is_better=False,
            prediction_type="predict",
            scorer_string="neg_root_mean_squared_error",
        )


__all__ = ["MAEMetric", "MSEMetric", "R2Metric", "RMSEMetric"]
