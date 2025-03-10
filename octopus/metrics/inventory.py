"""Metric inventory."""

from typing import Any, Dict, List, Type

import pandas as pd
from attrs import define, field

from octopus.exceptions import UnknownMetricError

from .config import MetricConfig
from .registry import MetricRegistry


@define
class MetricsInventory:
    """Metrics inventory."""

    metrics: Dict[str, Any] = field(factory=dict)
    _metric_configs: Dict[str, MetricConfig] = field(factory=dict)

    def __attrs_post_init__(self):
        self.metrics = MetricRegistry.get_all_metrics()

    def get_metric_config(self, name: str) -> MetricConfig | None:
        if name not in self._metric_configs:
            metric_class = self.metrics.get(name)
            if metric_class:
                config = metric_class.get_metric_config()
                config.name = name
                self._metric_configs[name] = config
            else:
                raise UnknownMetricError(
                    f"Unknown metric '{name}'. "
                    f"Available models are: {', '.join(list(self.metrics.keys()))}. "
                    "Please check the model name and try again."
                )
        return self._metric_configs[name]

    def get_inventory_item(self, name: str) -> MetricConfig:
        """Get the metric configuration for a given metric name.
        Args:
            name: The name of the metric to search for.
        Returns:
            The MetricConfig instance of the model if found.
        Raises:
            ValueError: If the metric with the given name is not found.
        """
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

    def get_performance_score(
        self,
        metric: str,
        target_assignments: dict,
        model: Type,
        data: pd.DataFrame,
        feature_columns: List,
    ) -> float:
        """Get performance score."""
        metric_config = self.get_metric_config(metric)
        metric_function = metric_config.metric_function
        prediction_type = metric_config.prediction_type
        ml_type = metric_config.ml_type

        if ml_type == "timetoevent":
            estimate = model.predict(data[feature_columns])
            event_time = data[target_assignments["duration"]].astype(float)
            event_indicator = data[target_assignments["event"]].astype(bool)
            performance = metric_function(event_indicator, event_time, estimate)[0]
        else:
            if prediction_type == "predict_proba":
                # binary only!!
                probabilities = model.predict_proba(data[feature_columns])[:, 1]
            elif prediction_type == "predict":
                probabilities = model.predict(data[feature_columns])
            else:
                return ValueError(f"Unknown prediction type {prediction_type}")
            target = data[target_assignments["default"]]
            performance = metric_function(target, probabilities)
        if metric_function.higher_is_better is True:
            return performance
        else:
            return -performance
