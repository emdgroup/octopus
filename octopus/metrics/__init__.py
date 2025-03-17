"""Init metrics."""

from .classification import *  # noqa F403
from .inventory import MetricsInventory
from .regression import *  # noqa F403
from .timetoevent import *  # noqa F403

metrics_inventory = MetricsInventory()
