"""Classification metrics."""

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

from .config import MetricConfig
from .registry import MetricRegistry


@MetricRegistry.register("AUCROC")
class AUCROCMetric:
    """AUCROC metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="AUCROC",
            metric_function=roc_auc_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("ACC")
class ACCMetric:
    """Accuracy metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="ACC",
            metric_function=accuracy_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("ACCBAL")
class ACCBALMetric:
    """Balanced accuracy metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="ACCBAL",
            metric_function=balanced_accuracy_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("LOGLOSS")
class LOGLOSSMetric:
    """Log loss metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="LOGLOSS",
            metric_function=log_loss,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("F1")
class F1Metric:
    """F1 metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="F1",
            metric_function=f1_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("NEGBRIERSCORE")
class NEGBRIERSCOREMetric:
    """Brier metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="NEGBRIERSCORE",
            metric_function=brier_score_loss,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("AUCPR")
class AUCPRMetric:
    """AUCPR metric class."""

    def _auc_pr(y_true: np.ndarray | list, y_score: np.ndarray | list) -> float:
        """Calculate the Area Under the Precision-Recall Curve (AUC-PR)."""
        # Compute precision-recall pairs for different probability thresholds
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # Compute the area under the precision-recall curve using the trapezoidal rule
        auc_score = auc(recall, precision)
        return auc_score

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="AUCPR",
            metric_function=AUCPRMetric._auc_pr,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


__all__ = [
    "ACCBALMetric",
    "ACCMetric",
    "AUCPRMetric",
    "AUCROCMetric",
    "F1Metric",
    "LOGLOSSMetric",
    "NEGBRIERSCOREMetric",
]
