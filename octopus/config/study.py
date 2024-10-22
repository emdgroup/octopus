"""Config study."""

from typing import Any, List

from attrs import Factory, define, field, validators

from octopus.metrics import metrics_inventory


def validate_target_metric(instance: "ConfigStudy", attribute: Any, value: str) -> None:
    """Validate the target_metric based on the ml_type.

    Args:
        instance: The ConfigStudy instance being validated.
        attribute: The name of the attribute being validated.
        value: The value of the target_metric being validated.

    Raises:
        ValueError: If the target_metric is not valid for the given ml_type.
    """
    ml_type = instance.ml_type
    valid_metrics = [
        metric
        for metric, details in metrics_inventory.items()
        if details["ml_type"] == ml_type
    ]

    if value not in valid_metrics:
        raise ValueError(
            f"Invalid target metric '{value}' for ml_type '{ml_type}'. "
            f"Valid options are: {valid_metrics}"
        )


# Custom validator for target_metric
def validate_metrics(instance: "ConfigStudy", attribute: Any, value: str) -> None:
    """Validate metrics based on the ml_type.

    Args:
        instance: The ConfigStudy instance being validated.
        attribute: The name of the attribute being validated.
        value: The value of metrics being validated.

    Raises:
        ValueError: If any metric is not valid for the given ml_type.
    """
    ml_type = instance.ml_type
    valid_metrics = [
        metric
        for metric, details in metrics_inventory.items()
        if details["ml_type"] == ml_type
    ]

    invalid_metrics = [metric for metric in value if metric not in valid_metrics]
    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics {','.join(invalid_metrics)} for ml_type '{ml_type}'. "
            f"Valid options are: {valid_metrics}"
        )


@define
class ConfigStudy:
    """Configuration for study parameters."""

    name: str = field(validator=[validators.instance_of(str)])
    """The name of the study."""

    ml_type: str = field(
        validator=[
            validators.in_(["classification", "regression", "timetoevent"]),
        ]
    )
    """The type of machine learning model.
    Choose from "classification", "regression" or "timetoevent"."""

    target_metric: str = field(validator=[validate_target_metric])
    """The primary metric used for model evaluation."""

    path: str = field(default="./studies/")
    """The path where study outputs are saved. Defaults to "./studies/"."""

    start_with_empty_study: bool = field(
        default=True, validator=[validators.instance_of(bool)]
    )

    n_folds_outer: int = field(default=5, validator=[validators.instance_of(int)])
    """The number of outer folds for cross-validation. Defaults to 5."""

    datasplit_seed_outer: int = field(
        default=0, validator=[validators.instance_of(int)]
    )
    """The seed used for data splitting in outer cross-validation. Defaults to 0."""

    silently_overwrite_study: bool = field(
        default=Factory(lambda: False), validator=[validators.instance_of(bool)]
    )
    """Indicates whether the study can be overwritten. Defaults to False."""

    # is this really useful?
    metrics: List = field(
        default=Factory(lambda self: [self.target_metric], takes_self=True),
        validator=[validators.instance_of(list), validate_metrics],
    )
    """A list of metrics to be calculated.
    Defaults to target_metric value."""

    ignore_data_health_warning: bool = field(
        default=Factory(lambda: False), validator=[validators.instance_of(bool)]
    )
    """Ignore data health checks warning and run machine learning workflow."""
