"""Init."""

from .classification_models import *  # noqa F403
from .inventory import ModelInventory
from .regression_models import *  # noqa F403
from .time_to_event_models import *  # noqa F403

model_inventory = ModelInventory()
