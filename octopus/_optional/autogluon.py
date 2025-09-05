"""Optional autogluon imports."""

from octopus.exceptions import OptionalImportError

try:
    from autogluon.core.metrics import (
        accuracy,
        balanced_accuracy,
        log_loss,
        mean_absolute_error,
        r2,
        roc_auc,
        root_mean_squared_error,
    )
    from autogluon.tabular import TabularPredictor


except ModuleNotFoundError as ex:
    raise OptionalImportError(
        "Autogluon is unavailable because the necessary optional "
        "dependencies are not installed. "
        'Consider installing Octopus with "autogluon" dependency, '
        'e.g. via `pip install -e ".[autogluon]"`.'
    ) from ex

__all__ = [
    "TabularPredictor",
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "mean_absolute_error",
    "r2",
    "roc_auc",
    "root_mean_squared_error",
]
