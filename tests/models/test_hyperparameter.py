"""Test hyperparameter."""

import pytest

from octopus.models.machine_learning.hyperparameter import Hyperparameter


@pytest.mark.parametrize(
    "type, name, low, high, step, choices, log, value, expected_exception",
    [
        # Valid cases
        ("int", "learning_rate", 0, 10, 1, [], False, None, None),
        ("float", "learning_rate", 0.0, 10.0, 0.1, [], False, None, None),
        (
            "categorical",
            "optimizer",
            None,
            None,
            None,
            ["adam", "sgd"],
            False,
            None,
            None,
        ),
        ("fixed", "dropout", None, None, None, [], False, 0.5, None),
        # Invalid cases
        ("invalid", "param", None, None, None, [], False, None, ValueError),
        ("int", "param", 10, 5, None, [], False, None, ValueError),
        ("float", "param", 0.0, 10.0, -0.1, [], False, None, ValueError),
        ("categorical", "param", None, None, None, [], False, None, ValueError),
        ("int", "param", 0, 10, None, [1, 2, 3], False, None, ValueError),
        ("categorical", "param", None, None, None, ["a", "b"], False, "c", ValueError),
        ("int", "param", 0, 10, None, [], False, 0.5, TypeError),
        ("float", "param", 0.0, 10.0, None, [], False, 5, TypeError),
        ("fixed", "param", None, None, None, [], False, None, ValueError),
    ],
)
def test_hyperparameters(
    type, name, low, high, step, choices, log, value, expected_exception
):
    """Test hyperparameters."""
    if expected_exception:
        with pytest.raises(expected_exception):
            Hyperparameter(
                type=type,
                name=name,
                low=low,
                high=high,
                step=step,
                choices=choices,
                log=log,
                value=value,
            )
    else:
        hp = Hyperparameter(
            type=type,
            name=name,
            low=low,
            high=high,
            step=step,
            choices=choices,
            log=log,
            value=value,
        )
        assert hp.type == type
        assert hp.name == name
        assert hp.low == low
        assert hp.high == high
        assert hp.step == step
        assert hp.choices == choices
        assert hp.log == log
        assert hp.value == value
