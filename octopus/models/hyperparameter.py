"""Hyperparameter class."""

from typing import Any

from attrs import define, field, validators


@define
class Hyperparameter:
    """Class to create hyperparameter space."""

    name: str


@define
class FloatHyperparameter(Hyperparameter):
    """Float Hyperparameter class."""

    low: float = field(validator=validators.instance_of((float, int)))
    high: float = field(validator=validators.instance_of((float, int)))
    step: float | None = field(default=None, validator=validators.optional(validators.instance_of((float, int))))
    log: bool = False
    value: float | None = field(default=None, validator=validators.optional(validators.instance_of((float, int))))

    def __attrs_post_init__(self):
        if self.low > self.high:
            raise ValueError("Low limit must be <= high limit.")

        if self.step is not None:
            if self.step <= 0:
                raise ValueError("step must be greater than 0.")
            if self.log:
                raise ValueError("Both step and log cannot be selected at the same time.")

        if self.value is not None and not (self.low <= self.value <= self.high):
            raise ValueError(f"value must be between low ({self.low}) and high ({self.high}), got {self.value}.")


@define
class IntHyperparameter(Hyperparameter):
    """Integer Hyperparameter class."""

    low: int = field(validator=validators.instance_of(int))
    high: int = field(validator=validators.instance_of(int))
    step: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    log: bool = False
    value: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))

    def __attrs_post_init__(self):
        if self.low > self.high:
            raise ValueError("Low limit must be <= high limit.")

        if self.step is not None:
            if self.step <= 0:
                raise ValueError("step must be greater than 0.")
            if self.log:
                raise ValueError("Both step and log cannot be selected at the same time.")

        if self.value is not None and not (self.low <= self.value <= self.high):
            raise ValueError(f"value must be between low ({self.low}) and high ({self.high}), got {self.value}.")


@define
class CategoricalHyperparameter(Hyperparameter):
    """Categorical Hyperparameter class."""

    choices: list[Any] = field(factory=list)
    value: Any | None = field(default=None)

    def __attrs_post_init__(self):
        if len(self.choices) == 0:
            raise ValueError("choices must be a non-empty list.")

        if self.value is not None and self.value not in self.choices:
            raise ValueError(f"value must be one of {self.choices}, got {self.value}.")


@define
class FixedHyperparameter(Hyperparameter):
    """Fixed Hyperparameter class."""

    value: Any = field()

    def __attrs_post_init__(self):
        if self.value is None:
            raise ValueError("value must be provided for FixedHyperparameter.")
