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
    "AUCROC": {"method": roc_auc_score, "ml_type": "classification"},
    "ACC": {"method": accuracy_score, "ml_type": "classification"},
    "ACCBAL": {"method": balanced_accuracy_score, "ml_type": "classification"},
    "LOGLOSS": {"method": log_loss, "ml_type": "classification"},
    "MAE": {"method": mean_absolute_error, "ml_type": "regression"},
    "MSE": {"method": mean_squared_error, "ml_type": "regression"},
    "R2": {"method": r2_score, "ml_type": "regression"},
    "CI": {"method": concordance_index_censored, "ml_type": "timetoevent"},
}
