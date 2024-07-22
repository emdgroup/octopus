"""Test Octo Data."""

import pandas as pd
import pytest

from octopus.data import OctoData


def test_initialization():
    data = pd.DataFrame(
        {
            "target": [1, 2, 1],
            "feature2": [4, 5, 4],
            "feature3": [7, 8, 9],
            "sample_id": [0, 1, 2],
        }
    )
    target_columns = ["target"]
    feature_columns = ["feature2", "feature3"]
    sample_id = "sample_id"
    datasplit_type = "sample"

    obj = OctoData(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sample_id=sample_id,
        datasplit_type=datasplit_type,
        data_quality_check=False,
    )

    assert isinstance(obj, OctoData)
    assert obj.target_columns == target_columns
    assert obj.sample_id == sample_id
    assert obj.datasplit_type == datasplit_type
    assert "group_features" in obj.data.columns
    assert "group_sample_and_features" in obj.data.columns
    assert "row_id" in obj.data.columns


if __name__ == "__main__":
    pytest.main()
