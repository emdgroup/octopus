"""Metric inventory."""

from typing import Any

from attrs import define, field

from octopus.exceptions import UnknownMetricError

from .config import MetricConfig
from .registry import MetricRegistry


@define
class MetricsInventory:
    """Metrics inventory."""

    metrics: dict[str, Any] = field(factory=dict)
    _metric_configs: dict[str, MetricConfig] = field(factory=dict)

    def __attrs_post_init__(self):
        self.metrics = MetricRegistry.get_all_metrics()

    def get_metric_config(self, name: str) -> MetricConfig | None:
        """Get metric config."""
        if name not in self._metric_configs:
            metric_class = self.metrics.get(name)
            if metric_class:
                config = metric_class.get_metric_config()
                config.name = name
                self._metric_configs[name] = config
            else:
                raise UnknownMetricError(
                    f"Unknown metric '{name}'. "
                    f"Available metrics are: {', '.join(list(self.metrics.keys()))}. "
                    "Please check the metrics name and try again."
                )
        return self._metric_configs[name]

    def get_inventory_item(self, name: str) -> MetricConfig:
        """Get the metric configuration for a given metric name."""
        for item in self.metrics:
            if item.name == name:
                return item
        raise ValueError(f"Metric item with name '{name}' not found")

    def get_metric_by_name(self, name: str) -> type:
        """Get metric class by name."""
        return self.get_metric_config(name)

    def get_metric_function(self, name: str) -> str:
        """Get metric function by name."""
        return self.get_metric_config(name).metric_function

    def get_direction(self, name: str) -> str:
        """Get the optuna direction by name."""
        if self.get_metric_config(name).higher_is_better:
            return "maximize"
        else:
            return "minimize"
