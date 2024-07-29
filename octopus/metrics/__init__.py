"""Init metrics."""

from typing import Callable, Dict

from .classification import classification_metrics
from .regression import regression_metrics
from .timetoevent import timetoevent_metrics

MetricsInventoryType = Dict[str, Dict[str, Callable]]


# Helper function to add ml_type
def add_ml_type(metrics, ml_type):
    """Create dict with ml type."""
    return {
        name: {"method": method, "ml_type": ml_type} for name, method in metrics.items()
    }


metrics_inventory: MetricsInventoryType = {
    **add_ml_type(classification_metrics, "classification"),
    **add_ml_type(regression_metrics, "regression"),
    **add_ml_type(timetoevent_metrics, "timetoevent"),
}
