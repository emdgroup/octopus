"""Hyperparameter class."""

from typing import Any

import optuna
from attrs import define, field, validators


@define
class Hyperparameter:
    """Class to create hyperparameter space."""

    name: str

    def suggest(self, trial: optuna.trial.Trial, unique_name: str) -> Any:
        """Suggest a value for this hyperparameter using Optuna trial."""
        raise NotImplementedError("Subclasses must implement suggest()")


@define
class FloatHyperparameter(Hyperparameter):
    """Float Hyperparameter class."""

    low: float = field(validator=validators.instance_of((float, int)))
    high: float = field(validator=validators.instance_of((float, int)))
    step: float | None = field(default=None, validator=validators.optional(validators.instance_of((float, int))))
    log: bool = False

    def __attrs_post_init__(self):
        if self.low > self.high:
            raise ValueError("Low limit must be <= high limit.")

        if self.step is not None:
            if self.step <= 0:
                raise ValueError("step must be greater than 0.")
            if self.log:
                raise ValueError("Both step and log cannot be selected at the same time.")

    def suggest(self, trial: optuna.trial.Trial, unique_name: str) -> float:
        """Suggest a float value using Optuna trial."""
        if self.step is not None:
            return trial.suggest_float(name=unique_name, low=self.low, high=self.high, step=self.step)
        else:
            return trial.suggest_float(name=unique_name, low=self.low, high=self.high, log=self.log)


@define
class IntHyperparameter(Hyperparameter):
    """Integer Hyperparameter class."""

    low: int = field(validator=validators.instance_of(int))
    high: int = field(validator=validators.instance_of(int))
    step: int | None = field(default=None, validator=validators.optional(validators.instance_of(int)))
    log: bool = False

    def __attrs_post_init__(self):
        if self.low > self.high:
            raise ValueError("Low limit must be <= high limit.")

        if self.step is not None:
            if self.step <= 0:
                raise ValueError("step must be greater than 0.")
            if self.log:
                raise ValueError("Both step and log cannot be selected at the same time.")

    def suggest(self, trial: optuna.trial.Trial, unique_name: str) -> int:
        """Suggest an int value using Optuna trial."""
        if self.step is not None:
            return trial.suggest_int(name=unique_name, low=self.low, high=self.high, step=self.step)
        else:
            return trial.suggest_int(name=unique_name, low=self.low, high=self.high, log=self.log)


@define
class CategoricalHyperparameter(Hyperparameter):
    """Categorical Hyperparameter class."""

    choices: list[Any] = field(factory=list)

    def __attrs_post_init__(self):
        if len(self.choices) == 0:
            raise ValueError("choices must be a non-empty list.")

    def suggest(self, trial: optuna.trial.Trial, unique_name: str) -> Any:
        """Suggest a categorical value using Optuna trial."""
        return trial.suggest_categorical(name=unique_name, choices=self.choices)


@define
class FixedHyperparameter(Hyperparameter):
    """Fixed Hyperparameter class."""

    value: Any = field()

    def __attrs_post_init__(self):
        if self.value is None:
            raise ValueError("value must be provided for FixedHyperparameter.")

    def suggest(self, trial: optuna.trial.Trial, unique_name: str) -> Any:
        """Return the fixed value (no trial suggestion needed)."""
        return self.value
