"""Regression metrics."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MetricConfig
from .registry import MetricRegistry


@MetricRegistry.register("R2")
class R2Metric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="R2",
            metric_function=r2_score,
            ml_type="regression",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("MAE")
class MAEMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="MAE",
            metric_function=mean_absolute_error,
            ml_type="regression",
            higher_is_better=False,
            prediction_type="predict",
        )


@MetricRegistry.register("MSE")
class MSEMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="MSE",
            metric_function=mean_squared_error,
            ml_type="regression",
            higher_is_better=False,
            prediction_type="predict",
        )


__all__ = ["R2Metric", "MAEMetric", "MSEMetric"]


# # Constants for metric names
# MAE = "MAE"
# MSE = "MSE"
# R2 = "R2"

# regression_metrics = {
#     MAE: mean_absolute_error,
#     MSE: mean_squared_error,
#     R2: r2_score,
# }
