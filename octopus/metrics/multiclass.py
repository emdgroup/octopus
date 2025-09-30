"""Multiclass metrics."""

from functools import partial

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .config import MetricConfig
from .registry import MetricRegistry


@MetricRegistry.register("ACCBAL_MC")
class ACCBALMulticlassMetric:
    """Balanced accuracy metric class for multiclass problems."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="ACCBAL_MC",
            metric_function=balanced_accuracy_score,
            ml_type="multiclass",
            higher_is_better=True,
            prediction_type="predict",
            scorer_string="balanced_accuracy",
        )


@MetricRegistry.register("AUCROC_MACRO")
class AUCROCMacroMulticlassMetric:
    """AUCROC metric class for multiclass problems (macro-average)."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="AUCROC_MACRO",
            metric_function=partial(roc_auc_score, multi_class="ovr", average="macro"),
            ml_type="multiclass",
            higher_is_better=True,
            prediction_type="predict_proba",
            scorer_string="roc_auc_ovr",
        )


@MetricRegistry.register("AUCROC_WEIGHTED")
class AUCROCWeightedMulticlassMetric:
    """AUCROC metric class for multiclass problems (weighted-average)."""

    @staticmethod
    def get_metric_config():
        """Create metric config."""
        return MetricConfig(
            name="AUCROC_WEIGHTED",
            metric_function=partial(roc_auc_score, multi_class="ovr", average="weighted"),
            ml_type="multiclass",
            higher_is_better=True,
            prediction_type="predict_proba",
            scorer_string="roc_auc_ovr_weighted",
        )


__all__ = [
    "ACCBALMulticlassMetric",
    "AUCROCMacroMulticlassMetric",
    "AUCROCWeightedMulticlassMetric",
]
