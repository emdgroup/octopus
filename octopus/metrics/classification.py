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

from .config import MetricConfig
from .registry import MetricRegistry


@MetricRegistry.register("AUCROC")
class AUCROCMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="AUCROC",
            metric_function=roc_auc_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("ACC")
class ACCMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="ACC",
            metric_function=accuracy_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("ACCBAL")
class ACCBALMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="ACCBAL",
            metric_function=balanced_accuracy_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("LOGLOSS")
class LOGLOSSMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="LOGLOSS",
            metric_function=log_loss,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("F1")
class F1Metric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="F1",
            metric_function=f1_score,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict",
        )


@MetricRegistry.register("NEGBRIERSCORE")
class NEGBRIERSCOREMetric:
    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="NEGBRIERSCORE",
            metric_function=brier_score_loss,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


@MetricRegistry.register("AUCPR")
class AUCPRMetric:
    def _auc_pr(
        y_true: Union[np.ndarray, list], y_score: Union[np.ndarray, list]
    ) -> float:
        """Calculate the Area Under the Precision-Recall Curve (AUC-PR).

        Args:
            y_true: True binary labels. Ground truth (correct) target values.
            y_score: Estimated probabilities or decision function output.
                Target scores can either be probability estimates of the positive class,
                confidence values, or non-thresholded measures of decisions.

        Returns:
            Area Under the Precision-Recall Curve (AUC-PR).
        """
        # Compute precision-recall pairs for different probability thresholds
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # Compute the area under the precision-recall curve using the trapezoidal rule
        auc_score = auc(recall, precision)
        return auc_score

    @staticmethod
    def get_metric_config():
        return MetricConfig(
            name="AUCPR",
            metric_function=AUCPRMetric._auc_pr,
            ml_type="classification",
            higher_is_better=True,
            prediction_type="predict_proba",
        )


__all__ = [
    "AUCROCMetric",
    "ACCMetric",
    "ACCBALMetric",
    "LOGLOSSMetric",
    "F1Metric",
    "NEGBRIERSCOREMetric",
    "AUCPRMetric",
]

# def get_classification_metrics():
#     """Return a list of MetricConfig objects for classification metrics.
#     Each MetricConfig object contains the configuration for a specific classification
#     metrics.
#     Returns:
#         List[MetricConfig]: A list of ModelConfig objects for classification models.
#     """
#     return [
#         MetricConfig(
#             name="AUCROC",
#             metric_class=roc_auc_score,
#             ml_type="classification",
#             minimize=False,
#             prediction_type="predict_proba",
#         ),
#         MetricConfig(
#             name="ACC",
#             metric_class=accuracy_score,
#             ml_type="classification",
#             minimize=False,
#             prediction_type="predict",
#         ),
#         MetricConfig(
#             name="ACCBAL",
#             metric_class=balanced_accuracy_score,
#             ml_type="classification",
#             minimize=False,
#             prediction_type="predict",
#         ),
#         MetricConfig(
#             name="LOGLOSS",
#             metric_class=log_loss,
#             ml_type="classification",
#             minimize=True,
#             prediction_type="predict_proba",
#         ),
#     ]


# """Classification metrics."""

# from typing import Union

# import numpy as np
# from sklearn.metrics import (
#     accuracy_score,
#     auc,
#     balanced_accuracy_score,
#     brier_score_loss,
#     f1_score,
#     log_loss,
#     precision_recall_curve,
#     roc_auc_score,
# )


# def auc_pr(y_true: Union[np.ndarray, list], y_score: Union[np.ndarray, list]) -> float:
#     """Calculate the Area Under the Precision-Recall Curve (AUC-PR).

#     Args:
#         y_true: True binary labels. Ground truth (correct) target values.
#         y_score: Estimated probabilities or decision function output.
#             Target scores can either be probability estimates of the positive class,
#             confidence values, or non-thresholded measures of decisions.

#     Returns:
#         Area Under the Precision-Recall Curve (AUC-PR).
#     """
#     # Compute precision-recall pairs for different probability thresholds
#     precision, recall, _ = precision_recall_curve(y_true, y_score)
#     # Compute the area under the precision-recall curve using the trapezoidal rule
#     auc_score = auc(recall, precision)
#     return auc_score


# # Constants for metric names
# AUCROC = "AUCROC"
# ACC = "ACC"
# ACCBAL = "ACCBAL"
# LOGLOSS = "LOGLOSS"
# F1 = "F1"
# NEGBRIERSCORE = "NEGBRIERSCORE"
# AUCPR = "AUCPR"


# classification_metrics = {
#     AUCROC: roc_auc_score,
#     ACC: accuracy_score,
#     ACCBAL: balanced_accuracy_score,
#     LOGLOSS: log_loss,
#     F1: f1_score,
#     NEGBRIERSCORE: brier_score_loss,
#     AUCPR: auc_pr,
# }
