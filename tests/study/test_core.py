"""Test OctoStudy core class."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from octopus.modules import Tako
from octopus.study import OctoClassification, OctoRegression
from octopus.study.core import _RUNNING_IN_TESTSUITE
from octopus.types import MLType


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(100)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 10, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.randint(0, 2, 100),
        }
    ).astype({"feature3": "category"})


@pytest.fixture
def basic_study():
    """Create a basic OctoClassification instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield OctoClassification(
            study_name="test_study",
            target_metric="AUCROC",
            feature_cols=["feature1", "feature2", "feature3"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )


def test_initialization(basic_study):
    """Test OctoStudy initialization."""
    assert basic_study.study_name == "test_study"
    assert basic_study.ml_type is None  # ml_type is determined during data validation
    assert basic_study.target_metric == "AUCROC"
    assert basic_study.feature_cols == ["feature1", "feature2", "feature3"]
    assert basic_study.target_col == "target"
    assert basic_study.sample_id_col == "sample_id_col"


def test_regression_ml_type():
    """Test that OctoRegression sets ml_type to regression."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoRegression(
            study_name="test",
            target_metric="R2",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert study.ml_type == MLType.REGRESSION


def test_default_workflow():
    """Test that default workflow is a single Tako task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert len(study.workflow) == 1
        assert isinstance(study.workflow[0], Tako)
        assert study.workflow[0].task_id == 0


def test_default_values():
    """Test default values are set correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
        )
        assert study.row_id_col is None
        assert study.stratification_col is None
        assert study.positive_class is None  # positive_class is determined during data validation
        assert study.n_outer_splits == 5 if not _RUNNING_IN_TESTSUITE else 2
        assert study.outer_split_seed == 0
        assert study.single_outer_split is None


def test_ml_type_values():
    """Test all valid ml_type values with appropriate classes."""
    test_cases = [
        (None, OctoClassification, "AUCROC", {"target_col": "target"}),  # ml_type determined during data validation
        (MLType.REGRESSION, OctoRegression, "R2", {"target_col": "target"}),
    ]
    for expected_ml_type, study_class, metric, extra_kwargs in test_cases:
        with tempfile.TemporaryDirectory() as temp_dir:
            study = study_class(
                study_name="test",
                target_metric=metric,
                feature_cols=["f1"],
                sample_id_col="id",
                studies_directory=temp_dir,
                **extra_kwargs,
            )
            assert study.ml_type == expected_ml_type


@pytest.mark.parametrize(
    "target_values, ml_type, positive_class, metric, match",
    [
        (np.tile([0, 1, 2], 30), None, None, "AUCROC", "does not support"),
        (np.tile([0, 1, 2], 30), MLType.MULTICLASS, 1, "ACCBAL_MC", "positive_class is not used for multiclass"),
        (np.tile([0, 1, 2], 30), None, 1, "ACCBAL_MC", "positive_class is not used for multiclass"),
        (np.tile([1, 5], 45), None, None, "AUCROC", "non-.0, 1. labels"),
        (np.tile([-1, 1], 45), None, None, "AUCROC", "non-.0, 1. labels"),
    ],
    ids=[
        "auto-multiclass-binary-metric",
        "explicit-multiclass-with-positive_class",
        "auto-multiclass-with-positive_class",
        "binary-15-no-positive_class",
        "binary-neg1_1-no-positive_class",
    ],
)
def test_resolve_ml_config_raises(target_values, ml_type, positive_class, metric, match):
    """_resolve_ml_config rejects incompatible (ml_type, positive_class, metric) combinations."""
    data = pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(len(target_values))],
            "feature1": np.linspace(0, 1, len(target_values)),
            "target": target_values,
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs: dict = {
            "study_name": "test",
            "target_metric": metric,
            "feature_cols": ["feature1"],
            "target_col": "target",
            "sample_id_col": "sample_id_col",
            "studies_directory": temp_dir,
        }
        if ml_type is not None:
            kwargs["ml_type"] = ml_type
        if positive_class is not None:
            kwargs["positive_class"] = positive_class
        study = OctoClassification(**kwargs)
        with pytest.raises(ValueError, match=match):
            study.fit(data)


def test_binary_01_auto_infers_positive_class_1():
    """Binary classification with {0, 1} labels auto-infers positive_class=1 when omitted."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(90)],
            "feature1": np.random.rand(90),
            "target": np.tile([0, 1], 45),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["feature1"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )
        resolved_ml_type, resolved_positive_class = study._resolve_ml_config(data)
        assert resolved_ml_type == MLType.BINARY
        assert resolved_positive_class == 1


def test_positive_class_bool_normalized_to_int():
    """OctoClassification(positive_class=True) is normalized to int(1) at construction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["f1"],
            target_col="target",
            sample_id_col="id",
            studies_directory=temp_dir,
            positive_class=True,
        )
        assert study.positive_class == 1
        assert not isinstance(study.positive_class, bool)


def test_resolve_ml_config_bool_target_auto_infers_positive_class():
    """Bool target auto-infers positive_class=1 without explicit value."""
    data = pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(60)],
            "feature1": np.random.rand(60),
            "target": np.tile([True, False], 30),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            target_metric="AUCROC",
            feature_cols=["feature1"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )
        resolved_ml_type, resolved_positive_class = study._resolve_ml_config(data)
        assert resolved_ml_type == MLType.BINARY
        assert resolved_positive_class == 1


def test_default_target_metric_auto_promotes_to_aucroc_macro_for_multiclass():
    """target_metric=None auto-promotes to AUCROC_MACRO when multiclass is detected."""
    data = pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(90)],
            "feature1": np.random.rand(90),
            "target": np.tile([0, 1, 2], 30),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            feature_cols=["feature1"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )
        assert study.target_metric is None
        ml_type, _ = study._resolve_ml_config(data)
        assert ml_type == MLType.MULTICLASS
        assert study.target_metric == "AUCROC_MACRO"


def test_default_target_metric_auto_promotes_to_aucroc_for_binary():
    """target_metric=None auto-resolves to AUCROC when binary is detected."""
    data = pd.DataFrame(
        {
            "sample_id_col": [f"S{i}" for i in range(60)],
            "feature1": np.random.rand(60),
            "target": np.tile([0, 1], 30),
        }
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoClassification(
            study_name="test",
            feature_cols=["feature1"],
            target_col="target",
            sample_id_col="sample_id_col",
            studies_directory=temp_dir,
        )
        ml_type, _ = study._resolve_ml_config(data)
        assert ml_type == MLType.BINARY
        assert study.target_metric == "AUCROC"
