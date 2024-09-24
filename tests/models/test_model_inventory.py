"""Test model inventory."""

# missing: test_create_optuna_parameters

from typing import Any, Dict, Optional

import pytest

from octopus.models.config import ModelConfig
from octopus.models.hyperparameter import Hyperparameter
from octopus.models.inventory import ModelInventory


# Mock ModelConfig class
class MockModel:
    """Random model."""

    def __init__(self, **params):
        self.params = params


@pytest.fixture
def model_config() -> ModelConfig:
    """Fixture to create a mock ModelConfig instance."""
    return ModelConfig(
        name="mock_model",
        model_class=MockModel,
        ml_type="regression",
        feature_method="internal",
        hyperparameters=[
            Hyperparameter(type="int", name="param1", low=0, high=10),
            Hyperparameter(type="float", name="param2", low=0.0, high=1.0),
            Hyperparameter(type="categorical", name="param3", choices=["a", "b", "c"]),
            Hyperparameter(type="fixed", name="param4", value=0.5),
        ],
        n_jobs="n_jobs",
        model_seed="random_seed",
    )


@pytest.fixture
def model_inventory(model_config: ModelConfig) -> ModelInventory:
    """Fixture to create a ModelInventory instance and add the mock ModelConfig."""
    inventory = ModelInventory()
    inventory.add_model(model_config)
    return inventory


@pytest.mark.parametrize(
    "name, expected_model",
    [
        ("mock_model", True),
        ("nonexistent_model", False),
    ],
)
def test_get_model_by_name(
    model_inventory: ModelInventory, name: str, expected_model: bool
) -> None:
    """Test retrieving a model by name."""
    model = model_inventory.get_model_by_name(name)
    if expected_model:
        assert model is not None
        assert model.name == name
    else:
        assert model is None


@pytest.mark.parametrize(
    "name, params, expected_exception",
    [
        (
            "mock_model",
            {"param1": 5, "param2": 0.5, "param3": "a", "param4": 0.5},
            None,
        ),
        ("nonexistent_model", {}, ValueError),
    ],
)
def test_get_model_instance(
    model_inventory: ModelInventory,
    name: str,
    params: Dict[str, Any],
    expected_exception: Optional[type],
) -> None:
    """Test creating a model instance with specified parameters."""
    if expected_exception:
        with pytest.raises(expected_exception):
            model_inventory.get_model_instance(name, params)
    else:
        model_instance = model_inventory.get_model_instance(name, params)
        assert model_instance.params == params


@pytest.mark.parametrize(
    "ml_type",
    [
        ("regression"),
        ("classification"),
        ("timetoevent"),
    ],
)
def test_inventory_initialization(model_inventory: ModelInventory, ml_type) -> None:
    """Test filtering models by type."""
    models = model_inventory.get_models_by_type(ml_type)
    assert len(models) > 0
    for model in models:
        assert model.ml_type == ml_type
