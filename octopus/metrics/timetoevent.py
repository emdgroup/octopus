"""Time to event metrics."""

from .config import MetricConfig
from .registry import MetricRegistry


@MetricRegistry.register("CI")
class CIMetric:
    """CI metric class."""

    @staticmethod
    def get_metric_config():
        """Get metric config."""
        from sksurv.metrics import concordance_index_censored

        return MetricConfig(
            name="CI",
            metric_function=concordance_index_censored,
            ml_type="timetoevent",
            higher_is_better=True,
            prediction_type="predict",
        )


__all__ = ["CIMetric"]
