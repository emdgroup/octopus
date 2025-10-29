"""Config study."""

from typing import Any

from attrs import Factory, define, field, validators

from octopus.metrics import metrics_inventory


def validate_metric(instance: "ConfigStudy", attribute: Any, value: str) -> None:
    """Validate the target_metric based on the ml_type.

    Args:
        instance: The ConfigStudy instance being validated.
        attribute: The name of the attribute being validated.
        value: The value of the target_metric being validated.

    Raises:
        ValueError: If the target_metric is not valid for the given ml_type.
    """
    if isinstance(value, str):
        value = [value]
    for metric in value:
        metric_ml_type = metrics_inventory.get_metric_config(metric).ml_type

        if metric_ml_type != instance.ml_type:
            raise ValueError(f"Invalid target metric '{metric}' for ml_type '{instance.ml_type}'.")


@define
class ConfigStudy:
    """Configuration for study parameters."""

    name: str = field(validator=[validators.instance_of(str)])
    """The name of the study."""

    ml_type: str = field(
        validator=[
            validators.in_(["classification", "multiclass", "regression", "timetoevent"]),
        ]
    )
    """The type of machine learning model.
    Choose from "classification", "regression" or "timetoevent"."""

    target_metric: str = field(validator=[validate_metric])
    """The primary metric used for model evaluation."""

    positive_class: int = field(default=1, validator=validators.instance_of(int))
    """The positive class label for binary classification. Defaults to 1. Not relevant for other ml_types."""

    path: str = field(default="./studies/")
    """The path where study outputs are saved. Defaults to "./studies/"."""

    start_with_empty_study: bool = field(default=True, validator=[validators.instance_of(bool)])

    n_folds_outer: int = field(default=5, validator=[validators.instance_of(int)])
    """The number of outer folds for cross-validation. Defaults to 5."""

    datasplit_seed_outer: int = field(default=0, validator=[validators.instance_of(int)])
    """The seed used for data splitting in outer cross-validation. Defaults to 0."""

    imputation_method: str = field(
        default="median",
        validator=[
            validators.in_(["median", "halfmin", "mice"]),
        ],
    )

    silently_overwrite_study: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """Indicates whether the study can be overwritten. Defaults to False."""

    # is this really useful?
    metrics: list = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[validators.instance_of(list), validate_metric],
    )
    """A list of metrics to be calculated.
    Defaults to target_metric value."""

    ignore_data_health_warning: bool = field(default=Factory(lambda: False), validator=[validators.instance_of(bool)])
    """Ignore data health checks warning and run machine learning workflow."""
