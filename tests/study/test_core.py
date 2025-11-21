"""Test OctoStudy core class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from octopus import OctoStudy
from octopus.modules import Octo
from octopus.study.types import DatasplitType, ImputationMethod, MLType


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(100)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 10, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.randint(0, 2, 100),
        }
    ).astype({"feature3": "category"})


@pytest.fixture
def basic_study():
    """Create a basic OctoStudy instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield OctoStudy(
            name="test_study",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["feature1", "feature2", "feature3"],
            target_columns=["target"],
            sample_id="sample_id",
            path=temp_dir,
            ignore_data_health_warning=True,
        )


def test_initialization(basic_study):
    """Test OctoStudy initialization."""
    assert basic_study.name == "test_study"
    assert basic_study.ml_type == MLType.CLASSIFICATION
    assert basic_study.target_metric == "AUCROC"
    assert basic_study.feature_columns == ["feature1", "feature2", "feature3"]
    assert basic_study.target_columns == ["target"]
    assert basic_study.sample_id == "sample_id"


def test_ml_type_string_conversion():
    """Test that ml_type accepts strings and converts to MLType enum."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="regression",
            target_metric="R2",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert study.ml_type == MLType.REGRESSION
        assert study.ml_type.value == "regression"


def test_datasplit_type_string_conversion():
    """Test that datasplit_type accepts strings and converts to DatasplitType enum."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            datasplit_type="group_features",
            path=temp_dir,
        )
        assert study.datasplit_type == DatasplitType.GROUP_FEATURES


def test_imputation_method_string_conversion():
    """Test that imputation_method accepts strings and converts to ImputationMethod enum."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            imputation_method="halfmin",
            path=temp_dir,
        )
        assert study.imputation_method == ImputationMethod.HALFMIN


def test_output_path_property():
    """Test that output_path is correctly computed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="my_study",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert study.output_path == Path(temp_dir) / "my_study"


def test_default_tasks():
    """Test that default tasks is a single Octo task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert len(study.tasks) == 1
        assert isinstance(study.tasks[0], Octo)
        assert study.tasks[0].task_id == 0


def test_default_metrics():
    """Test that default metrics list contains target_metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert study.metrics == ["AUCROC"]


def test_custom_metrics():
    """Test custom metrics list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            metrics=["AUCROC", "ACCBAL", "F1"],
            path=temp_dir,
        )
        assert study.metrics == ["AUCROC", "ACCBAL", "F1"]


def test_default_values():
    """Test default values are set correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            ml_type="classification",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert study.datasplit_type == DatasplitType.SAMPLE
        assert study.row_id is None
        assert study.stratification_column is None
        assert study.positive_class == 1
        assert study.n_folds_outer == 5
        assert study.datasplit_seed_outer == 0
        assert study.imputation_method == ImputationMethod.MEDIAN
        assert study.ignore_data_health_warning is False
        assert study.outer_parallelization is True
        assert study.run_single_experiment_num == -1


def test_ml_type_values():
    """Test all valid ml_type values."""
    ml_type_metrics = {
        "classification": "AUCROC",
        "regression": "R2",
        "timetoevent": "CI",
        "multiclass": "AUCROC_MACRO",
    }
    for ml_type, metric in ml_type_metrics.items():
        with tempfile.TemporaryDirectory() as temp_dir:
            study = OctoStudy(
                name="test",
                ml_type=ml_type,
                target_metric=metric,
                feature_columns=["f1"],
                target_columns=["target"],
                sample_id="id",
                path=temp_dir,
            )
            assert study.ml_type.value == ml_type


def test_invalid_ml_type():
    """Test that invalid ml_type raises error."""
    with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(ValueError):
        OctoStudy(
            name="test",
            ml_type="invalid_type",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
