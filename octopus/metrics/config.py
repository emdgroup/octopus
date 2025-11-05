"""Metric Config."""

from collections.abc import Callable
from typing import Any

from attrs import define, field, validators

from octopus.models.config import ML_TYPES, PRED_TYPES, MLType, OctoArrayLike, PredType

# Type alias for metric functions
# Metric functions should accept (y_true, y_pred, **kwargs) and return a numeric value.
# We use Callable[..., Any] because sklearn functions have varying signatures and
# return types (float, np.float64, floating[_16Bit], etc.).
# Parameter structure and return type conversion are handled by runtime validation
# and the compute() method which explicitly converts to float.
MetricFunction = Callable[..., Any]


@define
class MetricConfig:
    """Metric config.

    Stores configuration for a metric function including the function itself
    and any additional parameters needed to call it.
    """

    name: str
    metric_function: MetricFunction = field(validator=validators.is_callable())
    ml_type: MLType = field(validator=validators.in_(ML_TYPES))
    higher_is_better: bool = field(validator=validators.instance_of(bool))
    prediction_type: PredType = field(validator=validators.in_(PRED_TYPES))
    scorer_string: str = field(validator=validators.instance_of(str))  # needed for some sklearn functionalities
    metric_params: dict[str, Any] = field(factory=dict)

    def compute(self, y_true: OctoArrayLike, y_pred: OctoArrayLike) -> float:
        """Compute the metric with stored parameters.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            float: The computed metric value
        """
        return float(self.metric_function(y_true, y_pred, **self.metric_params))
