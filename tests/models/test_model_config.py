"""Test hyperparameter."""

import pytest

from octopus.models.machine_learning.config import ModelConfig
from octopus.models.machine_learning.hyperparameter import Hyperparameter


def test_model_config_initialization():
    """Test initialization of ModelConfig."""
    hyperparameters = [
        Hyperparameter(type="float", name="alpha", low=1e-5, high=1e5, log=True),
        Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
    ]

    config = ModelConfig(
        name="test_model",
        model_class=object,
        feature_method="some_method",
        ml_type="regression",
        hyperparameters=hyperparameters,
        translate={"param1": "translated_param1"},
    )

    assert config.name == "test_model"
    assert config.model_class == object
    assert config.feature_method == "some_method"
    assert config.ml_type == "regression"
    assert config.hyperparameters == hyperparameters
    assert config.translate == {"param1": "translated_param1"}
    assert config.n_repeats is None


def test_model_config_with_conflict():
    """Test ModelConfig initialization with hyperparameter name conflict."""
    hyperparameters = [
        Hyperparameter(type="float", name="alpha", low=1e-5, high=1e5, log=True),
        Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
    ]

    with pytest.raises(
        ValueError,
        match="Hyperparameter name 'alpha' conflicts with a translated name.",
    ):
        ModelConfig(
            name="test_model",
            model_class=object,
            feature_method="some_method",
            ml_type="regression",
            hyperparameters=hyperparameters,
            translate={"param1": "alpha"},
        )
