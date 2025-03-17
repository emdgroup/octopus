"""Hyperparameter class."""

from typing import Any, List

from attrs import define, field, validators


@define
class Hyperparameter:
    """Class to create hyperparameter space."""

    type: str = field(
        validator=validators.in_(["int", "float", "categorical", "fixed"])
    )
    name: str
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    choices: List[Any] = field(factory=list)
    log: bool = False
    value: Any = None

    def __attrs_post_init__(self):
        self.validate_hyperparameters()

    def validate_hyperparameters(self):
        """Validate hyperparameter inputs."""

        # Helper function to raise type error
        def raise_type_error(attr, expected_type):
            raise TypeError(
                f"""{attr} must be {expected_type},
                got {type(getattr(self, attr)).__name__}."""
            )

        # Validate int and float types
        if self.type in {"int", "float"}:
            if self.low is None or self.high is None:
                raise ValueError(
                    "low and high must be provided for int and float types."
                )
            if not isinstance(self.low, (int, float)):
                raise_type_error("low", "int or float")
            if not isinstance(self.high, (int, float)):
                raise_type_error("high", "int or float")
            if self.low >= self.high:
                raise ValueError("low must be less than high.")
            if self.step is not None:
                if not isinstance(self.step, (int, float)):
                    raise_type_error("step", "int or float")
                if self.step <= 0:
                    raise ValueError("step must be greater than 0.")
            if self.choices:
                raise ValueError(
                    "choices should not be provided for int or float types."
                )
            if self.step is not None and self.log is True:
                raise ValueError(
                    "Both step and log cannot be selected at the same time."
                )

        # Validate categorical type
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("choices must be provided for categorical type.")

        # Validate fixed type
        elif self.type == "fixed":
            if self.value is None:
                raise ValueError("value must be provided for fixed type.")

        # Invalid type
        else:
            raise ValueError(f"Invalid type: {self.type}.")

        # Validate value attribute
        if self.value is not None:
            if self.type == "int" and not isinstance(self.value, int):
                raise_type_error("value", "int")
            if self.type == "float" and not isinstance(self.value, float):
                raise_type_error("value", "float")
            if self.type == "categorical" and self.value not in self.choices:
                raise ValueError(
                    f"value must be one of {self.choices}, got {self.value}."
                )
