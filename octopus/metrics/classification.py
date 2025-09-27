"""Classification metrics."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
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
            scorer_string="roc_auc",
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
            scorer_string="accuracy",
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
            scorer_string="balanced_accuracy",
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
            scorer_string="neg_log_loss",
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
            scorer_string="f1",
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
            scorer_string="neg_brier_score",
        )


@MetricRegistry.register("AUCPR")
class AUCPRMetric:
    """AUCPR metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="AUCPR",
            metric_function=average_precision_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
            scorer_string="average_precision",
        )


@MetricRegistry.register("MCC")
class MCCMetric:
    """Matthews Correlation Coefficient metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="MCC",
            metric_function=matthews_corrcoef,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
            scorer_string="matthews_corrcoef",
        )


@MetricRegistry.register("PRECISION")
class PRECISIONMetric:
    """Precision metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="PRECISION",
            metric_function=precision_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
            scorer_string="precision",
        )


@MetricRegistry.register("RECALL")
class RECALLMetric:
    """Recall metric class."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="RECALL",
            metric_function=recall_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
            scorer_string="recall",
        )


__all__ = [
    "ACCBALMetric",
    "ACCMetric",
    "AUCPRMetric",
    "AUCROCMetric",
    "F1Metric",
    "LOGLOSSMetric",
    "MCCMetric",
    "NEGBRIERSCOREMetric",
    "PRECISIONMetric",
    "RECALLMetric",
]
