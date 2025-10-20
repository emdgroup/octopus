"""Test hyperparameter."""

import pytest

from octopus.models.hyperparameter import Hyperparameter


@pytest.mark.parametrize(
    "hyperparameter_type, name, low, high, step, choices, log, value, expected_exception",
    [
        # Valid int hyperparameter
        ("int", "para1", 1, 10, None, [], False, None, None),
        # Invalid int hyperparameter with step
        ("int", "para1", 1, 10, -1, [], False, None, ValueError),
        # Invalid int hyperparameter with choices
        ("int", "para1", 1, 10, None, [1, 2, 3], False, None, ValueError),
        # Valid int hyperparameter step
        ("int", "para1", 1, 10, 1, [], False, None, None),
        # Vvalid int hyperparameter log
        ("int", "para1", 1, 10, None, [], True, None, None),
        # Invalid int hyperparameter step and log selected
        ("int", "para1", 1, 10, 1, [], True, None, ValueError),
        # Valid float hyperparameter
        ("float", "para1", 0.1, 1.0, None, [], False, None, None),
        # Invalid float hyperparameter with high less than low
        ("float", "param1", 1.0, 0.1, None, [], False, None, ValueError),
        # Invalid float hyperparameter with choices
        ("float", "para1", 1, 10, None, [1, 2, 3], False, None, ValueError),
        # Valid float hyperparameter step
        ("float", "para1", 1, 10, 1, [], False, None, None),
        # Vvalid float hyperparameter log
        ("float", "para1", 1, 10, None, [], True, None, None),
        # Invalid float hyperparameter step and log selected
        ("float", "para1", 1, 10, 1, [], True, None, ValueError),
        # Valid categorical hyperparameter
        ("categorical", "para1", None, None, None, ["a", "b"], False, None, None),
        # Invalid categorical hyperparameter without choices
        ("categorical", "para1", None, None, None, [], False, None, ValueError),
        # Valid fixed hyperparameter
        ("fixed", "para1", None, None, None, [], False, 5, None),
        # Invalid fixed hyperparameter without value
        ("fixed", "para1", None, None, None, [], False, None, ValueError),
        # Invalid type
        ("unknown", "para1", None, None, None, [], False, None, ValueError),
    ],
)
def test_validate_hyperparameters(hyperparameter_type, name, low, high, step, choices, log, value, expected_exception):
    """Test validate hyperparameters."""
    if expected_exception:
        with pytest.raises(expected_exception):
            Hyperparameter(
                type=hyperparameter_type,
                name=name,
                low=low,
                high=high,
                step=step,
                choices=choices,
                log=log,
                value=value,
            )
    else:
        Hyperparameter(
            type=hyperparameter_type,
            name=name,
            low=low,
            high=high,
            step=step,
            choices=choices,
            log=log,
            value=value,
        )
