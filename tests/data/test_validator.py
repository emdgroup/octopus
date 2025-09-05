"""Test OctoData validator."""

import numpy as np
import pandas as pd
import pytest

from octopus.data.validator import OctoDataValidator


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "sample_id": [f"S{i}" for i in range(1, 101)],
            "feature1": np.random.rand(100),
            "feature2": np.random.randint(0, 5, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.randint(0, 2, 100),
            "time": np.random.rand(100) * 10,
            "strat": ["X", "Y"] * 50,
        }
    ).astype({"strat": "category", "feature3": "category"})


@pytest.fixture
def valid_validator(sample_data):
    """Create valid validator."""
    return OctoDataValidator(
        data=sample_data,
        feature_columns=["feature1", "feature2", "feature3"],
        target_columns=["target"],
        sample_id="sample_id",
        row_id="id",
        stratification_column="strat",
        target_assignments={},
        relevant_columns=[
            "id",
            "sample_id",
            "feature1",
            "feature2",
            "feature3",
            "target",
            "strat",
        ],
    )


def test_initialization(valid_validator):
    """Test initialization."""
    assert isinstance(valid_validator, OctoDataValidator)


def test_validate(valid_validator):
    """Test validate function."""
    valid_validator.validate()


def test_validate_columns_exist(valid_validator):
    """Test column exists validation."""
    valid_validator._validate_columns_exist()

    invalid_validator = valid_validator
    invalid_validator.relevant_columns.append("non_existent_column")
    with pytest.raises(ValueError):
        invalid_validator._validate_columns_exist()


def test_check_for_duplicated_columns(valid_validator):
    """Test duplicated columns validation."""
    valid_validator._check_for_duplicated_columns()

    invalid_validator = valid_validator
    invalid_validator.feature_columns.append("target")
    with pytest.raises(ValueError):
        invalid_validator._check_for_duplicated_columns()


def test_validate_stratification_column(valid_validator):
    """Test stratification column validation."""
    valid_validator._validate_stratification_column()

    invalid_validator = valid_validator
    invalid_validator.stratification_column = "sample_id"
    with pytest.raises(ValueError):
        invalid_validator._validate_stratification_column()


def test_validate_target_assignments(valid_validator):
    """Test target assignment validation."""
    # Test with single target column and no assignments
    valid_validator._validate_target_assignments()  # Should not raise any exceptions

    # Test with single target column and assignments (should raise an error)
    invalid_validator = valid_validator
    invalid_validator.target_assignments = {"event": "target"}
    with pytest.raises(ValueError):
        invalid_validator._validate_target_assignments()

    # Test with multiple target columns and no assignments
    invalid_validator = valid_validator
    invalid_validator.target_columns = ["target", "time"]
    invalid_validator.target_assignments = {}
    with pytest.raises(ValueError):
        invalid_validator._validate_target_assignments()

    # Test with multiple target columns and missing assignments
    invalid_validator.target_assignments = {"event": "target"}
    with pytest.raises(ValueError):
        invalid_validator._validate_target_assignments()

    # Test with multiple target columns and invalid assignments
    invalid_validator.target_assignments = {
        "event": "target",
        "duration": "non_existent_column",
    }
    with pytest.raises(ValueError):
        invalid_validator._validate_target_assignments()

    # Test with multiple target columns and duplicate assignments
    invalid_validator.target_assignments = {"event": "target", "duration": "target"}
    with pytest.raises(ValueError):
        invalid_validator._validate_target_assignments()


def test_validate_number_of_targets(valid_validator):
    """Test number of targets validation."""
    valid_validator._validate_number_of_targets()

    # Test with too many targets
    invalid_validator = valid_validator
    invalid_validator.target_columns = ["target1", "target2", "target3"]
    with pytest.raises(ValueError):
        invalid_validator._validate_number_of_targets()


def test_validate_column_dtypes(valid_validator):
    """Test column dtype validation."""
    valid_validator._validate_column_dtypes()

    invalid_validator = valid_validator
    invalid_validator.data["feature1"] = invalid_validator.data["feature1"].astype("object")
    with pytest.raises(ValueError):
        invalid_validator._validate_column_dtypes()


def test_validate_column_names_characters(valid_validator):
    """Test column names characters validation."""
    valid_validator._validate_column_names_characters()

    invalid_validator = valid_validator
    invalid_validator.relevant_columns.append("invalid:column")
    with pytest.raises(ValueError):
        invalid_validator._validate_column_names_characters()


def test_validate_column_names(valid_validator):
    """Test column name validation."""
    valid_validator._validate_column_names()

    invalid_validator = valid_validator
    invalid_validator.data["group_features"] = 0
    with pytest.raises(ValueError):
        invalid_validator._validate_column_names()


def test_validate_row_id_unique(valid_validator):
    """Test row_id is unique validation."""
    valid_validator._validate_row_id_unique()

    invalid_validator = valid_validator
    invalid_validator.data.loc[0, "id"] = 2
    with pytest.raises(ValueError):
        invalid_validator._validate_row_id_unique()


def test_validate_with_two_targets(sample_data):
    """Test two targets."""
    two_target_validator = OctoDataValidator(
        data=sample_data,
        feature_columns=["feature1", "feature2", "feature3"],
        target_columns=["target", "time"],
        sample_id="sample_id",
        row_id="id",
        stratification_column="strat",
        target_assignments={"event": "target", "time": "time"},
        relevant_columns=[
            "id",
            "sample_id",
            "feature1",
            "feature2",
            "feature3",
            "target",
            "time",
            "strat",
        ],
    )
    two_target_validator.validate()

    invalid_validator = two_target_validator
    invalid_validator.target_assignments = {}
    with pytest.raises(ValueError):
        invalid_validator._validate_number_of_targets()
