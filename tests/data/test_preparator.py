"""Test OctoData preparator."""

import numpy as np
import pandas as pd
import pytest

from octopus.data.preparator import OctoDataPreparator


@pytest.fixture
def sample_data():
    """Create sample data."""
    return pd.DataFrame(
        {
            "feature1": [1, 2, 1, 2, 3, 3, 4, 4],
            "feature2": ["a", "b", "a", "b", "c", "c", "d", "e"],
            "target": [0, 1, 0, 1, 1, 0, 1, 1],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s5", "s6", "s6"],
            "bool_col": [True, False, True, False, True, True, False, True],
            "null_col": ["none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"],
            "inf_col": ["inf", "-infinity", "inf", "-inf", "∞", "-infinity", 5, 6],
        }
    )


@pytest.fixture
def octo_preparator(sample_data):
    """Create OctoDataPreparator instance from sample data."""
    return OctoDataPreparator(
        data=sample_data,
        feature_columns=["feature1", "feature2", "bool_col", "null_col", "inf_col"],
        target_columns=["target"],
        sample_id="sample_id",
        row_id=None,
        target_assignments={},
    )


def test_prepare(octo_preparator):
    """Test prepare function."""
    data, feature_columns, row_id, target_assignments = octo_preparator.prepare()

    assert isinstance(data, pd.DataFrame)
    assert isinstance(feature_columns, list)
    assert isinstance(row_id, str)
    assert isinstance(target_assignments, dict)


def test_sort_features(octo_preparator):
    """Test sort features function."""
    octo_preparator.feature_columns = [
        "feature1",
        "feature2",
        "bool_col",
        "null_col",
        "inf_col",
        "a",
        "aa",
        "aaa",
        "b",
    ]
    octo_preparator._sort_features()
    assert octo_preparator.feature_columns == [
        "a",
        "b",
        "aa",
        "aaa",
        "inf_col",
        "bool_col",
        "feature1",
        "feature2",
        "null_col",
    ]


def test_set_target_assignments(octo_preparator):
    """Test set target assignments function."""
    octo_preparator._set_target_assignments()
    assert octo_preparator.target_assignments == {"default": "target"}


def test_remove_singlevalue_features(octo_preparator):
    """Test remove single value features function."""
    octo_preparator.data["single_value"] = [1, 1, 1, 1, 1, 1, 1, 1]
    octo_preparator.feature_columns.append("single_value")
    octo_preparator._remove_singlevalue_features()
    assert "single_value" not in octo_preparator.feature_columns


def test_transform_bool_to_int(octo_preparator):
    """Test transform bool to int function."""
    octo_preparator._transform_bool_to_int()
    assert octo_preparator.data["bool_col"].dtype == int
    assert octo_preparator.data["bool_col"].tolist() == [1, 0, 1, 0, 1, 1, 0, 1]


def test_create_row_id(octo_preparator):
    """Test create row id function."""
    octo_preparator._create_row_id()
    assert "row_id" in octo_preparator.data.columns
    assert octo_preparator.row_id == "row_id"
    assert octo_preparator.data["row_id"].tolist() == list(range(8))


def test_add_group_features(octo_preparator):
    """Test add group features function."""
    octo_preparator._standardize_null_values()
    octo_preparator._standardize_inf_values()
    octo_preparator._transform_bool_to_int()
    octo_preparator._add_group_features()

    # Check if new columns are added
    assert "group_features" in octo_preparator.data.columns
    assert "group_sample_and_features" in octo_preparator.data.columns

    # Check group_features
    # Rows with the same feature combinations should have the same group_features value
    assert octo_preparator.data.loc[0, "group_features"] == octo_preparator.data.loc[2, "group_features"]
    assert octo_preparator.data.loc[1, "group_features"] == octo_preparator.data.loc[3, "group_features"]
    assert octo_preparator.data.loc[4, "group_features"] != octo_preparator.data.loc[5, "group_features"]
    assert octo_preparator.data.loc[6, "group_features"] != octo_preparator.data.loc[7, "group_features"]

    # There should be 6 unique group_features (4 unique combinations)
    assert octo_preparator.data["group_features"].nunique() == 6

    # Check group_sample_and_features
    # Rows with the same sample_id OR the same features should have the
    # same group_sample_and_features value
    assert (
        octo_preparator.data.loc[0, "group_sample_and_features"]
        == octo_preparator.data.loc[2, "group_sample_and_features"]
    )
    assert (
        octo_preparator.data.loc[4, "group_sample_and_features"]
        == octo_preparator.data.loc[5, "group_sample_and_features"]
    )
    assert (
        octo_preparator.data.loc[6, "group_sample_and_features"]
        == octo_preparator.data.loc[7, "group_sample_and_features"]
    )

    # There should be 6 unique group_sample_and_features
    # (s1/s3, s2/s4, s5, s6 - where s1/s3 and s2/s4 are grouped due to
    # identical features)
    assert octo_preparator.data["group_sample_and_features"].nunique() == 4

    # Verify that the index has been reset
    assert octo_preparator.data.index.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    # Additional checks
    # Verify that rows with different features but same sample_id
    # have the same group_sample_and_features but different group_features
    assert (
        octo_preparator.data.loc[6, "group_sample_and_features"]
        == octo_preparator.data.loc[7, "group_sample_and_features"]
    )
    assert octo_preparator.data.loc[6, "group_features"] != octo_preparator.data.loc[7, "group_features"]

    # Verify that rows with the same features but different sample_id
    # have the same group_features and group_sample_and_features
    assert octo_preparator.data.loc[0, "group_features"] == octo_preparator.data.loc[2, "group_features"]
    assert (
        octo_preparator.data.loc[0, "group_sample_and_features"]
        == octo_preparator.data.loc[2, "group_sample_and_features"]
    )


def test_standardize_null_values(octo_preparator):
    """Test standardize null values function."""
    octo_preparator._standardize_null_values()
    assert octo_preparator.data["null_col"].isna().all()


def test_standardize_inf_values(octo_preparator):
    """Test standardize inf values function."""
    octo_preparator._standardize_inf_values()
    assert np.isinf(octo_preparator.data["inf_col"].iloc[0])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[1])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[4])
    assert np.isinf(octo_preparator.data["inf_col"].iloc[5])


def test_prepare_full_process(octo_preparator):
    """Test preparation function."""
    data, feature_columns, _, target_assignments = octo_preparator.prepare()

    assert "row_id" in data.columns
    assert "group_features" in data.columns
    assert "group_sample_and_features" in data.columns
    assert data["bool_col"].dtype == int
    assert data["null_col"].isna().all()
    assert np.isinf(data["inf_col"].iloc[0])
    assert "single_value" not in feature_columns
    assert target_assignments == {"default": "target"}


def test_row_id_already_exists():
    """Test if row_id already exists."""
    with pytest.raises(Exception):
        prep = OctoDataPreparator(
            data=pd.DataFrame({"row_id": [1, 2], "a": [3, 4]}),
            feature_columns=["a"],
            target_columns=["row_id"],
            sample_id="a",
            row_id=None,
            target_assignments={},
        )
        prep._create_row_id()
