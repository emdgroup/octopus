"""Metrics."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sksurv.metrics import concordance_index_censored

metrics_inventory = {
    "AUCROC": roc_auc_score,
    "ACC": accuracy_score,
    "ACCBAL": balanced_accuracy_score,
    "LOGLOSS": log_loss,
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "R2": r2_score,
    "CI": concordance_index_censored,
}
