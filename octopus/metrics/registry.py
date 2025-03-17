"""Metrics Registry."""


class MetricRegistry:
    """Metric Registry."""

    _metrics = {}

    @classmethod
    def register(cls, name: str):
        """Register metric."""

        def decorator(metric_class):
            cls._metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get_metric(cls, name: str):
        """Get metric."""
        return cls._metrics.get(name)

    @classmethod
    def get_all_metrics(cls):
        """Get all metrics."""
        return cls._metrics
