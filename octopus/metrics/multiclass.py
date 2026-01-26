"""Multiclass metrics."""

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .config import MetricConfig
from .core import Metrics


@Metrics.register("ACCBAL_MC")
def accbal_multiclass_metric() -> MetricConfig:
    """Balanced accuracy metric configuration for multiclass problems."""
    return MetricConfig(
        name="ACCBAL_MC",
        metric_function=balanced_accuracy_score,
        ml_type="multiclass",
        higher_is_better=True,
        prediction_type="predict",
        scorer_string="balanced_accuracy",
    )


@Metrics.register("AUCROC_MACRO")
def aucroc_macro_multiclass_metric() -> MetricConfig:
    """AUCROC metric configuration for multiclass problems (macro-average)."""
    return MetricConfig(
        name="AUCROC_MACRO",
        metric_function=roc_auc_score,
        metric_params={"multi_class": "ovr", "average": "macro"},
        ml_type="multiclass",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="roc_auc_ovr",
    )


@Metrics.register("AUCROC_WEIGHTED")
def aucroc_weighted_multiclass_metric() -> MetricConfig:
    """AUCROC metric configuration for multiclass problems (weighted-average)."""
    return MetricConfig(
        name="AUCROC_WEIGHTED",
        metric_function=roc_auc_score,
        metric_params={"multi_class": "ovr", "average": "weighted"},
        ml_type="multiclass",
        higher_is_better=True,
        prediction_type="predict_proba",
        scorer_string="roc_auc_ovr_weighted",
    )
