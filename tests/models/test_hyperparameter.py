"""Test hyperparameter."""

import pytest

from octopus.models.hyperparameter import (
    CategoricalHyperparameter,
    FixedHyperparameter,
    FloatHyperparameter,
    IntHyperparameter,
)


@pytest.mark.parametrize(
    "hyperparameter_type, name, kwargs, expected_exception",
    [
        # Valid int hyperparameter
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": False, "value": None}, None),
        # Invalid int hyperparameter: low > high
        (IntHyperparameter, "para1", {"low": 10, "high": 1, "step": None, "log": False, "value": None}, ValueError),
        # Invalid int hyperparameter with step
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": -1, "log": False, "value": None}, ValueError),
        # Valid int hyperparameter step
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": False, "value": None}, None),
        # Valid int hyperparameter log
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": True, "value": None}, None),
        # Invalid int hyperparameter step and log selected
        (IntHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": True, "value": None}, ValueError),
        # Valid float hyperparameter
        (FloatHyperparameter, "para1", {"low": 0.1, "high": 1.0, "step": None, "log": False, "value": None}, None),
        # Invalid float hyperparameter with high less than low
        (
            FloatHyperparameter,
            "param1",
            {"low": 1.0, "high": 0.1, "step": None, "log": False, "value": None},
            ValueError,
        ),
        # Valid float hyperparameter step
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": False, "value": None}, None),
        # Valid float hyperparameter log
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": None, "log": True, "value": None}, None),
        # Invalid float hyperparameter step and log selected
        (FloatHyperparameter, "para1", {"low": 1, "high": 10, "step": 1, "log": True, "value": None}, ValueError),
        # Valid categorical hyperparameter
        (CategoricalHyperparameter, "para1", {"choices": ["a", "b"], "value": None}, None),
        # Invalid categorical hyperparameter without choices
        (CategoricalHyperparameter, "para1", {"choices": [], "value": None}, ValueError),
        # Invalid categorical hyperparameter with value not in choices
        (CategoricalHyperparameter, "para1", {"choices": ["a", "b"], "value": "c"}, ValueError),
        # Valid fixed hyperparameter
        (FixedHyperparameter, "para1", {"value": 5}, None),
        # Invalid fixed hyperparameter without value
        (FixedHyperparameter, "para1", {"value": None}, ValueError),
    ],
)
def test_validate_hyperparameters(hyperparameter_type, name, kwargs, expected_exception):
    """Test validate hyperparameters."""
    if expected_exception:
        with pytest.raises(expected_exception):
            hyperparameter_type(name=name, **kwargs)
    else:
        hyperparameter_type(name=name, **kwargs)
