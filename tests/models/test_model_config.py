"""Test hyperparameter."""

import pytest

from octopus.models.config import ModelConfig
from octopus.models.hyperparameter import Hyperparameter


def test_model_config_initialization():
    """Test initialization of ModelConfig."""
    hyperparameters = [
        Hyperparameter(type="float", name="alpha", low=1e-5, high=1e5, log=True),
        Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
    ]

    model = ModelConfig(
        name="test_model",
        model_class=object,
        feature_method="some_method",
        ml_type="regression",
        hyperparameters=hyperparameters,
        n_jobs="n_jobs",
        model_seed="random_seed",
    )

    assert model.name == "test_model"
    assert model.model_class == object
    assert model.feature_method == "some_method"
    assert model.ml_type == "regression"
    assert model.hyperparameters == hyperparameters
    assert model.n_jobs == "n_jobs"
    assert model.model_seed == "random_seed"
    assert model.n_repeats is None


def test_model_config_with_conflict():
    """Test ModelConfig initialization with hyperparameter name conflict."""
    hyperparameters = [
        Hyperparameter(type="float", name="n_jobs", low=1e-5, high=1e5, log=True),
        Hyperparameter(type="categorical", name="fit_intercept", choices=[True, False]),
    ]

    with pytest.raises(
        ValueError,
        match="Hyperparameter 'n_jobs' is not allowed in 'hyperparameters'.",
    ):
        ModelConfig(
            name="test_model",
            model_class=object,
            feature_method="some_method",
            ml_type="regression",
            hyperparameters=hyperparameters,
            n_jobs="n_jobs",
            model_seed="random_seed",
        )
