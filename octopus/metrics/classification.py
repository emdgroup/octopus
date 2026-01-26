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
from .core import Metrics


@Metrics.register("AUCROC")
def aucroc_metric() -> MetricConfig:
    """AUCROC metric configuration."""
    return MetricConfig(
        name="AUCROC",
        metric_function=roc_auc_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="roc_auc",
    )


@Metrics.register("ACC")
def acc_metric() -> MetricConfig:
    """Accuracy metric configuration."""
    return MetricConfig(
        name="ACC",
        metric_function=accuracy_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="accuracy",
    )


@Metrics.register("ACCBAL")
def accbal_metric() -> MetricConfig:
    """Balanced accuracy metric configuration."""
    return MetricConfig(
        name="ACCBAL",
        metric_function=balanced_accuracy_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="balanced_accuracy",
    )


@Metrics.register("LOGLOSS")
def logloss_metric() -> MetricConfig:
    """Log loss metric configuration."""
    return MetricConfig(
        name="LOGLOSS",
        metric_function=log_loss,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="neg_log_loss",
    )


@Metrics.register("F1")
def f1_metric() -> MetricConfig:
    """F1 metric configuration."""
    return MetricConfig(
        name="F1",
        metric_function=f1_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="f1",
    )


@Metrics.register("NEGBRIERSCORE")
def negbrierscore_metric() -> MetricConfig:
    """Brier score metric configuration."""
    return MetricConfig(
        name="NEGBRIERSCORE",
        metric_function=brier_score_loss,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="neg_brier_score",
    )


@Metrics.register("AUCPR")
def aucpr_metric() -> MetricConfig:
    """AUCPR metric configuration."""
    return MetricConfig(
        name="AUCPR",
        metric_function=average_precision_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="average_precision",
    )


@Metrics.register("MCC")
def mcc_metric() -> MetricConfig:
    """Matthews Correlation Coefficient metric configuration."""
    return MetricConfig(
        name="MCC",
        metric_function=matthews_corrcoef,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="matthews_corrcoef",
    )


@Metrics.register("PRECISION")
def precision_metric() -> MetricConfig:
    """Precision metric configuration."""
    return MetricConfig(
        name="PRECISION",
        metric_function=precision_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="precision",
    )


@Metrics.register("RECALL")
def recall_metric() -> MetricConfig:
    """Recall metric configuration."""
    return MetricConfig(
        name="RECALL",
        metric_function=recall_score,
        ml_type="classification",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="recall",
    )
