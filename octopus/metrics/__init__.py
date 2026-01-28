"""Init metrics."""

from .classification import *  # noqa: F403
from .core import Metrics
from .multiclass import *  # noqa: F403
from .regression import *  # noqa: F403
from .timetoevent import *  # noqa: F403

__all__ = ["Metrics"]
