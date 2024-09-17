"""Model inventory."""

from octopus.models.machine_learning.classification_models import (
    get_classification_models,
)
from octopus.models.machine_learning.model_inventory import ModelInventory
from octopus.models.machine_learning.regression_models import get_regression_models
from octopus.models.machine_learning.time_to_event_models import (
    get_time_to_event_models,
)

# Initialize the model inventory
model_inventory = ModelInventory()

for model_config in get_regression_models():
    model_inventory.add_model(model_config)

for model_config in get_classification_models():
    model_inventory.add_model(model_config)

for model_config in get_time_to_event_models():
    model_inventory.add_model(model_config)
