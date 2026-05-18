"""Test metrics utility functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from octopus.metrics import Metrics
from octopus.metrics.utils import (
    get_performance_from_model,
    get_performance_from_predictions,
    get_score_from_model,
    get_score_from_prediction,
)
from octopus.types import PredictionType


class TestGetPerformanceFromModel:
    """Test get_performance_from_model function."""

    def test_binary_classification(self):
        """Test binary classification."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 2, 100), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=1,
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 3, 150), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="ACCBAL_MC",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    def test_regression(self):
        """Test regression."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randn(100) * 10 + 50, columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = Ridge(random_state=42)
        model.fit(X, y.values.ravel())

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="R2",
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)


class TestGetPerformanceFromPredictions:
    """Test get_performance_from_predictions function.

    Tests use prediction format from training.py (lines 408-432):
    - Binary/multiclass: row_id_col, "prediction", target, probability columns (as int)
    - Regression: row_id_col, "prediction", target
    """

    def test_binary_classification(self):
        """Test binary classification predictions."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 1, 0, 1],
                        "target": [0, 1, 1, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        1: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="AUCROC", target_assignments={"default": "target"}, positive_class=1
        )

        assert "training_0" in performance
        assert "dev" in performance["training_0"]
        assert isinstance(performance["training_0"]["dev"], float)

    def test_binary_raises_when_positive_class_missing_from_columns(self):
        """positive_class absent from prediction columns raises typed ValueError, not KeyError."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2],
                        "prediction": [0, 1, 0],
                        "target": [0, 1, 0],
                        0: [0.8, 0.3, 0.9],
                        1: [0.2, 0.7, 0.1],
                    }
                )
            }
        }
        with pytest.raises(ValueError, match=r"positive_class=2 not found in prediction columns"):
            get_performance_from_predictions(
                predictions=predictions,
                target_metric="AUCROC",
                target_assignments={"default": "target"},
                positive_class=2,
            )

    def test_multiclass_standard_row_id_col(self):
        """Test multiclass with standard 'row' column."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_row_id_col(self):
        """Test multiclass with numeric 'row_id' column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row_id_col": [10, 20, 30, 40, 50],  # numeric row identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_sample_id_col_column(self):
        """Test multiclass with numeric 'sample_id_col' column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "sample_id_col": [100, 101, 102, 103, 104],  # another numeric identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_multiclass_with_string_row_id_col(self):
        """Test multiclass with string row column (should be excluded from probabilities)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "patient_id": ["P001", "P002", "P003", "P004", "P005"],  # string identifier
                        "prediction": [0, 1, 2, 0, 1],
                        "target": [0, 1, 2, 0, 2],
                        0: [0.7, 0.2, 0.1, 0.8, 0.3],
                        1: [0.2, 0.6, 0.2, 0.1, 0.5],
                        2: [0.1, 0.2, 0.7, 0.1, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="ACCBAL_MC", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)

    def test_regression(self):
        """Test regression predictions."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [50.2, 48.7, 52.1, 49.3, 51.8],
                        "target": [50.0, 49.0, 52.0, 49.5, 51.5],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions, target_metric="R2", target_assignments={"default": "target"}
        )

        assert "training_0" in performance
        assert isinstance(performance["training_0"]["dev"], float)


class TestGetScoreFromPrediction:
    """Test get_score_from_prediction function."""

    def test_maximize_metric(self):
        """Test score calculation for maximize metric (AUROC)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 1, 1, 0, 1],
                        "target": [0, 1, 1, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        1: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        scores = get_score_from_prediction(
            predictions=predictions, target_metric="AUCROC", target_assignments={"default": "target"}, positive_class=1
        )

        assert "training_0" in scores
        assert scores["training_0"]["dev"] > 0  # maximize: score = performance

    def test_minimize_metric(self):
        """Test score calculation for minimize metric (MSE)."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [50.2, 48.7, 52.1, 49.3, 51.8],
                        "target": [50.0, 49.0, 52.0, 49.5, 51.5],
                    }
                )
            }
        }

        scores = get_score_from_prediction(
            predictions=predictions, target_metric="MSE", target_assignments={"default": "target"}
        )

        assert "training_0" in scores
        assert scores["training_0"]["dev"] < 0  # minimize: score = -performance


class TestGetScoreFromModel:
    """Test get_score_from_model function."""

    def test_maximize_metric(self):
        """Test score calculation for maximize metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randint(0, 2, 100), columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y.values.ravel())

        score = get_score_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="AUCROC",
            target_assignments={"default": "target"},
            positive_class=1,
        )

        assert isinstance(score, float)
        assert score > 0  # maximize: score = performance

    def test_minimize_metric(self):
        """Test score calculation for minimize metric."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(np.random.randn(100) * 10 + 50, columns=["target"])
        data = pd.concat([X, y], axis=1)

        model = Ridge(random_state=42)
        model.fit(X, y.values.ravel())

        score = get_score_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric="MSE",
            target_assignments={"default": "target"},
        )

        assert isinstance(score, float)
        assert score < 0  # minimize: score = -performance


class TestMulticlassNonConsecutiveLabels:
    """Test multiclass with non-consecutive integer class labels."""

    @pytest.mark.parametrize(
        "labels, metric",
        [
            ([1, 3, 5], "ACCBAL_MC"),
            ([1, 2, 3], "AUCROC_MACRO"),
        ],
        ids=["non-consecutive-accbal", "non-zero-based-aucroc"],
    )
    def test_model_based(self, labels, metric):
        """get_performance_from_model handles non-zero-based and non-consecutive labels."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
        y = np.tile(labels, 150 // len(labels))
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric=metric,
            target_assignments={"default": "target"},
        )

        assert isinstance(performance, float)
        assert 0 <= performance <= 1

    @pytest.mark.parametrize(
        "labels, metric",
        [
            ([1, 3, 5], "ACCBAL_MC"),
            ([1, 2, 3], "AUCROC_MACRO"),
        ],
        ids=["non-consecutive-accbal", "non-zero-based-aucroc"],
    )
    def test_predictions_based(self, labels, metric):
        """get_performance_from_predictions handles non-zero-based and non-consecutive labels."""
        c0, c1, c2 = labels
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4, 5],
                        "prediction": [c0, c1, c2, c0, c1, c2],
                        "target": [c0, c1, c2, c0, c2, c1],
                        c0: [0.7, 0.1, 0.1, 0.8, 0.1, 0.2],
                        c1: [0.2, 0.8, 0.1, 0.1, 0.2, 0.6],
                        c2: [0.1, 0.1, 0.8, 0.1, 0.7, 0.2],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric=metric,
            target_assignments={"default": "target"},
        )

        assert isinstance(performance["training_0"]["dev"], float)
        assert 0 <= performance["training_0"]["dev"] <= 1

    def test_predictions_missing_probability_columns_raises(self):
        """Test ValueError when no integer-named probability columns exist."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2],
                        "prediction": [1, 2, 3],
                        "target": [1, 2, 3],
                        "prob_1": [0.7, 0.1, 0.2],
                        "prob_2": [0.2, 0.8, 0.1],
                        "prob_3": [0.1, 0.1, 0.7],
                    }
                )
            }
        }

        with pytest.raises(ValueError, match="at least 2 integer-named probability columns"):
            get_performance_from_predictions(
                predictions=predictions,
                target_metric="AUCROC_MACRO",
                target_assignments={"default": "target"},
            )

    def test_predictions_missing_class_in_prob_columns_raises(self):
        """Test ValueError when target has a class not in probability columns."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2],
                        "prediction": [1, 3, 5],
                        "target": [1, 3, 5],
                        1: [0.7, 0.2, 0.1],
                        3: [0.3, 0.8, 0.2],
                    }
                )
            }
        }

        with pytest.raises(ValueError, match="missing probability columns"):
            get_performance_from_predictions(
                predictions=predictions,
                target_metric="AUCROC_MACRO",
                target_assignments={"default": "target"},
            )


class TestBinaryBinarization:
    """Test binary classification with non-{0,1} labels."""

    @pytest.mark.parametrize(
        "labels, positive_class, metric",
        [
            ([0, 2], 2, "AUCROC"),
            ([0, 2], 0, "AUCROC"),
            ([0, 2], 2, "ACCBAL"),
            ([0, 2], 2, "F1"),
            ([-1, 1], 1, "AUCROC"),
            ([-1, 1], 1, "ACCBAL"),
            ([-1, 1], 1, "F1"),
        ],
        ids=[
            "0_2-pos2-aucroc",
            "0_2-pos0-aucroc",
            "0_2-pos2-accbal",
            "0_2-pos2-f1",
            "neg1_1-pos1-aucroc",
            "neg1_1-pos1-accbal",
            "neg1_1-pos1-f1",
        ],
    )
    def test_model_based(self, labels, positive_class, metric):
        """get_performance_from_model handles non-{0,1} binary labels."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = np.tile(labels, 50)
        data = pd.concat([X, pd.DataFrame({"target": y})], axis=1)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        performance = get_performance_from_model(
            model=model,
            data=data,
            feature_cols=X.columns.tolist(),
            target_metric=metric,
            target_assignments={"default": "target"},
            positive_class=positive_class,
        )

        pos_idx = list(model.classes_).index(positive_class)
        proba_pos = model.predict_proba(X)[:, pos_idx]
        y_binary = (y == positive_class).astype(int)
        metric_instance = Metrics.get_instance(metric)
        if metric_instance.prediction_type == PredictionType.PROBABILITIES:
            baseline = metric_instance.calculate(y_binary, proba_pos)
        else:
            preds_binary = (proba_pos >= 0.5).astype(int)
            baseline = metric_instance.calculate(y_binary, preds_binary)
        assert performance == pytest.approx(baseline, rel=1e-9)

    @pytest.mark.parametrize(
        "metric, positive_class, lower_bound, upper_bound",
        [
            ("AUCROC", 2, 0.0, 1.0),
            ("AUCPR", 2, 0.0, 1.0),
            ("LOGLOSS", 0, 0.0, None),
            ("LOGLOSS", 2, 0.0, None),
        ],
        ids=["aucroc-pos2", "aucpr-pos2", "logloss-pos0", "logloss-pos2"],
    )
    def test_predictions_based_binary_labels_0_2(self, metric, positive_class, lower_bound, upper_bound):
        """get_performance_from_predictions handles binary labels {0, 2} across metrics."""
        predictions = {
            "training_0": {
                "dev": pd.DataFrame(
                    {
                        "row": [0, 1, 2, 3, 4],
                        "prediction": [0, 2, 2, 0, 2],
                        "target": [0, 2, 2, 0, 0],
                        0: [0.8, 0.3, 0.2, 0.9, 0.4],
                        2: [0.2, 0.7, 0.8, 0.1, 0.6],
                    }
                )
            }
        }

        performance = get_performance_from_predictions(
            predictions=predictions,
            target_metric=metric,
            target_assignments={"default": "target"},
            positive_class=positive_class,
        )

        value = performance["training_0"]["dev"]
        assert isinstance(value, float)
        assert value > lower_bound
        if upper_bound is not None:
            assert value <= upper_bound

        pred_df = predictions["training_0"]["dev"]
        y_binary = (pred_df["target"] == positive_class).astype(int).to_numpy()
        proba_pos = pred_df[positive_class].to_numpy()
        metric_instance = Metrics.get_instance(metric)
        if metric_instance.prediction_type == PredictionType.PROBABILITIES:
            baseline = metric_instance.calculate(y_binary, proba_pos)
        else:
            preds_binary = (proba_pos >= 0.5).astype(int)
            baseline = metric_instance.calculate(y_binary, preds_binary)
        assert value == pytest.approx(baseline, rel=1e-9)
