"""Test OctoStudy core class."""

import tempfile

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from octopus import OctoStudy
from octopus.modules import Octo
from octopus.study.core import _RUNNING_IN_TESTSUITE
from octopus.study.types import DatasplitType, ImputationMethod, MLType
from octopus.task import Task


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
def basic_study(sample_data):
    """Create a basic OctoStudy instance with fitted data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test_study",
            target_metric="AUCROC",
            feature_columns=["feature1", "feature2"],
            target_columns=["target"],
            sample_id="sample_id",
            path=temp_dir,
            ignore_data_health_warning=True,
            silently_overwrite_study=True,
        )
        study.fit(sample_data)
        yield study


def test_initialization(basic_study):
    """Test OctoStudy initialization and inference."""
    assert basic_study.name == "test_study"
    assert basic_study.ml_type == MLType.CLASSIFICATION  # Auto-inferred from binary target
    assert basic_study.target_metric == "AUCROC"
    assert "feature1" in basic_study.feature_columns
    assert "feature2" in basic_study.feature_columns
    assert basic_study.target_columns == ["target"]
    assert basic_study.sample_id == "sample_id"


@pytest.mark.parametrize(
    "param_name,param_value,expected_enum,kwargs",
    [
        (
            "datasplit_type",
            "group_features",
            DatasplitType.GROUP_FEATURES,
            {"target_metric": "AUCROC", "datasplit_type": "group_features"},
        ),
        (
            "imputation_method",
            "halfmin",
            ImputationMethod.HALFMIN,
            {"target_metric": "AUCROC", "imputation_method": "halfmin"},
        ),
    ],
)
def test_string_to_enum_conversion(param_name, param_value, expected_enum, kwargs):
    """Test that parameters accept strings and convert to enum types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            **kwargs,
        )
        assert getattr(study, param_name) == expected_enum


def test_output_path_property():
    """Test that output_path is correctly computed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="my_study",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert study.output_path == UPath(temp_dir) / "my_study"


def test_default_workflow():
    """Test that default workflow is a single Octo task."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
        )
        assert len(study.workflow) == 1
        assert isinstance(study.workflow[0], Octo)
        assert study.workflow[0].task_id == 0


@pytest.mark.parametrize(
    "metrics_input,expected_metrics",
    [
        (None, ["AUCROC"]),  # default metrics
        (["AUCROC", "ACCBAL", "F1"], ["AUCROC", "ACCBAL", "F1"]),  # custom metrics
    ],
)
def test_metrics(metrics_input, expected_metrics):
    """Test metrics list with default and custom values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs = {
            "name": "test",
            "target_metric": "AUCROC",
            "feature_columns": ["f1"],
            "target_columns": ["target"],
            "sample_id": "id",
            "path": temp_dir,
        }
        if metrics_input is not None:
            kwargs["metrics"] = metrics_input

        study = OctoStudy(**kwargs)
        assert study.metrics == expected_metrics


def test_default_values():
    """Test default values are set correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
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
        assert study.n_folds_outer == 5 if not _RUNNING_IN_TESTSUITE else 2
        assert study.datasplit_seed_outer == 0
        assert study.imputation_method == ImputationMethod.MEDIAN
        assert study.ignore_data_health_warning is False
        assert study.outer_parallelization is True
        assert study.run_single_experiment_num == -1


def test_ml_type_inference():
    """Test that ml_type is correctly inferred during fit()."""
    np.random.seed(42)

    # Test classification (binary target)
    data_classification = pd.DataFrame(
        {
            "id": [f"S{i}" for i in range(50)],
            "f1": np.random.rand(50),
            "target": np.random.randint(0, 2, 50),
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            ignore_data_health_warning=True,
            silently_overwrite_study=True,
        )
        study.fit(data_classification)
        assert study.ml_type == MLType.CLASSIFICATION

    # Test regression (numeric target)
    data_regression = pd.DataFrame(
        {
            "id": [f"S{i}" for i in range(50)],
            "f1": np.random.rand(50),
            "target": np.random.rand(50) * 100,  # Continuous values
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="MAE",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            ignore_data_health_warning=True,
            silently_overwrite_study=True,
        )
        study.fit(data_regression)
        assert study.ml_type == MLType.REGRESSION

    # Test multiclass (categorical target)
    data_multiclass = pd.DataFrame(
        {
            "id": [f"S{i}" for i in range(50)],
            "f1": np.random.rand(50),
            "target": pd.Categorical(np.random.choice(["A", "B", "C"], 50)),
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="AUCROC_MACRO",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            ignore_data_health_warning=True,
            silently_overwrite_study=True,
        )
        study.fit(data_multiclass)
        assert study.ml_type == MLType.MULTICLASS


def test_start_with_empty_study_valid():
    """Test that start_with_empty_study=True works with tasks that don't have load_task=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            start_with_empty_study=True,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=False)],
        )
        assert study.start_with_empty_study is True


def test_start_with_empty_study_invalid():
    """Test that start_with_empty_study=True raises error when workflow has tasks with load_task=True."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        pytest.raises(
            ValueError, match="Cannot set start_with_empty_study=True when workflow contains tasks with load_task=True"
        ),
    ):
        OctoStudy(
            name="test",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            start_with_empty_study=True,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=True)],
        )


def test_start_with_empty_study_false_with_load_task():
    """Test that start_with_empty_study=False allows tasks with load_task=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        study = OctoStudy(
            name="test",
            target_metric="AUCROC",
            feature_columns=["f1"],
            target_columns=["target"],
            sample_id="id",
            path=temp_dir,
            start_with_empty_study=False,
            workflow=[Octo(task_id=0), Task(task_id=1, depends_on_task=0, load_task=True)],
        )
        assert study.start_with_empty_study is False
