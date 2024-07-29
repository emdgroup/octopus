"""Classification metrics."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

# Constants for metric names
AUCROC = "AUCROC"
ACC = "ACC"
ACCBAL = "ACCBAL"
LOGLOSS = "LOGLOSS"

classification_metrics = {
    AUCROC: roc_auc_score,
    ACC: accuracy_score,
    ACCBAL: balanced_accuracy_score,
    LOGLOSS: log_loss,
}
