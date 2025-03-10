"""Init metrics."""

from .classification import *
from .iventory import MetricsInventory
from .regression import *
from .timetoevent import *

metrics_inventory = MetricsInventory()
