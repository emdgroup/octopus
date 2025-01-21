"""Classification metrics."""

from typing import Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def auc_pr(y_true: Union[np.ndarray, list], y_score: Union[np.ndarray, list]) -> float:
    """Calculate the Area Under the Precision-Recall Curve (AUC-PR).

    Parameters:
    ----------
    y_true : Union[np.ndarray, list]
        True binary labels. Ground truth (correct) target values.

    y_score : Union[np.ndarray, list]
        Estimated probabilities or decision function output.
        Target scores can either be probability estimates of the positive class,
        confidence values, or non-thresholded measures of decisions.

    Returns:
    -------
    auc_score : float
        Area Under the Precision-Recall Curve (AUC-PR).

    """
    # Compute precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Compute the area under the precision-recall curve using the trapezoidal rule
    auc_score = auc(recall, precision)
    return auc_score


# Constants for metric names
AUCROC = "AUCROC"
ACC = "ACC"
ACCBAL = "ACCBAL"
LOGLOSS = "LOGLOSS"
F1 = "F1"
NEGBRIERSCORE = "NEGBRIERSCORE"
AUCPR = "AUCPR"


classification_metrics = {
    AUCROC: roc_auc_score,
    ACC: accuracy_score,
    ACCBAL: balanced_accuracy_score,
    LOGLOSS: log_loss,
    F1: f1_score,
    NEGBRIERSCORE: brier_score_loss,
    AUCPR: auc_pr,
}
