"""Metric Config."""

from collections.abc import Callable

from attrs import define, field, validators
from numpy.typing import ArrayLike

type MetricFunction = Callable[[ArrayLike, ArrayLike], float]


@define
class MetricConfig:
    """Metric config."""

    name: str
    metric_function: MetricFunction = field(validator=validators.is_callable())
    ml_type: str = field(validator=validators.in_(["regression", "classification", "multiclass", "timetoevent"]))
    higher_is_better: bool = field(validator=validators.instance_of(bool))
    prediction_type: str = field(validator=validators.in_(["predict", "predict_proba"]))
    scorer_string: str = field(validator=validators.instance_of(str))  # needed for some sklearn functionalities
