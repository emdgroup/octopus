"""Model hyperparameter class."""

from typing import Any, List, Union

from attrs import define, field, validators


def validate_hyperparameters(instance):
    """Validate hyperparameter inputs."""

    # Helper function to raise type error
    def raise_type_error(attr, expected_type):
        raise TypeError(
            f"""{attr} must be {expected_type},
            got {type(getattr(instance, attr)).__name__}."""
        )

    # Validate int and float types
    if instance.type in {"int", "float"}:
        if instance.low is None or instance.high is None:
            raise ValueError("low and high must be provided for int and float types.")
        if not isinstance(instance.low, (int, float)):
            raise_type_error("low", "int or float")
        if not isinstance(instance.high, (int, float)):
            raise_type_error("high", "int or float")
        if instance.low >= instance.high:
            raise ValueError("low must be less than high.")
        if instance.step is not None:
            if not isinstance(instance.step, (int, float)):
                raise_type_error("step", "int or float")
            if instance.step <= 0:
                raise ValueError("step must be greater than 0.")
        if instance.choices:
            raise ValueError("choices should not be provided for int or float types.")

    # Validate categorical type
    elif instance.type == "categorical":
        if not instance.choices:
            raise ValueError("choices must be provided for categorical type.")

    # Validate fixed type
    elif instance.type == "fixed":
        if instance.value is None:
            raise ValueError("value must be provided for fixed type.")

    # Invalid type
    else:
        raise ValueError(f"Invalid type: {instance.type}.")

    # Validate value attribute
    if instance.value is not None:
        if instance.type == "int" and not isinstance(instance.value, int):
            raise_type_error("value", "int")
        if instance.type == "float" and not isinstance(instance.value, float):
            raise_type_error("value", "float")
        if instance.type == "categorical" and instance.value not in instance.choices:
            raise ValueError(
                f"value must be one of {instance.choices}, got {instance.value}."
            )


@define
class Hyperparameter:
    """Class to create hyperparameter space."""

    type: str = field(
        validator=validators.in_(["int", "float", "categorical", "fixed"])
    )
    name: str
    low: Union[int, float, None] = None
    high: Union[int, float, None] = None
    step: Union[int, float, None] = None
    choices: List[Any] = field(factory=list)
    log: bool = False
    value: Any = None

    def __attrs_post_init__(self):
        validate_hyperparameters(self)
