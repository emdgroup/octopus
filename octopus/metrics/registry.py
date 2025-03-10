"""Metrics Registry."""


class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_class):
            cls._metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get_metric(cls, name: str):
        return cls._metrics.get(name)

    @classmethod
    def get_all_metrics(cls):
        return cls._metrics
