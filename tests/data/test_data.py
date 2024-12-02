"""Test Octo Data."""

# import pandas as pd
# import pytest

# from octopus import OctoData


# @pytest.fixture
# def sample_data():
#     """Sample data."""
#     df = pd.DataFrame(
#         {
#             "target_1": [0, 1, 0, 1],
#             "target_2": [5, 1, 5, 1],
#             "target_3": [5, 1, 5, 1],
#             "feature_1": [1, 2, 3, 4],
#             "feature_2": [5, 6, 7, 8],
#             "feature_3": [5, 5, 5, 5],
#             "sample_id": [0, 1, 2, 2],
#             "id": [10, 11, 12, 13],
#         }
#     )
#     return df


# @pytest.mark.parametrize(
#     "target_columns, target_assignments, expected",
#     [
#         (["target_1"], {}, ["target_1"]),
#         (
#             ["target_1", "target_2"],
#             {"Duration": "target_1", "Event": "target_2"},
#             ["target_1", "target_2"],
#         ),
#         (
#             ["target_1", "target_2", "target_3"],
#             {"A": "target_1", "B": "target_2", "C": "target_3"},
#             ["target_1", "target_2", "target_3"],
#         ),
#     ],
# )
# def test_targets(sample_data, target_columns, target_assignments, expected):
#     """Test that targets are initialized correctly."""
#     obj = OctoData(
#         data=sample_data,
#         feature_columns=["feature_1", "feature_2"],
#         target_columns=target_columns,
#         target_assignments=target_assignments,
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert obj.target_columns == expected


# @pytest.mark.parametrize(
#     "feature_columns, expected",
#     [
#         (["feature_1", "feature_2"], ["feature_1", "feature_2"]),
#         (
#             ["feature_1", "feature_2", "feature_3"],
#             ["feature_1", "feature_2"],
#         ),
#     ],
# )
# def test_features(sample_data, feature_columns, expected):
#     """Test that features are initialized correctly."""
#     obj = OctoData(
#         data=sample_data,
#         feature_columns=feature_columns,
#         target_columns=["target_1"],
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert obj.feature_columns == expected


# @pytest.mark.parametrize(
#     "row_id, expected",
#     [
#         ("", "row_id"),
#         ("id", "id"),
#     ],
# )
# def test_row_id(sample_data, row_id, expected):
#     """Test that the row_id is initialized correctly."""
#     obj = OctoData(
#         data=sample_data,
#         feature_columns=["feature_1", "feature_2"],
#         target_columns=["target_1"],
#         sample_id="sample_id",
#         datasplit_type="sample",
#         row_id=row_id,
#         data_quality_check=False,
#     )
#     assert obj.row_id == expected


# @pytest.mark.parametrize(
#     "target_columns, target_assignments, expected",
#     [
#         (["target_1"], {}, {"default": "target_1"}),
#         (
#             ["target_1", "target_2"],
#             {"Duration": "target_1", "Event": "target_2"},
#             {"Duration": "target_1", "Event": "target_2"},
#         ),
#         (
#             ["target_1", "target_2", "target_3"],
#             {"A": "target_1", "B": "target_2", "C": "target_3"},
#             {"A": "target_1", "B": "target_2", "C": "target_3"},
#         ),
#     ],
# )
# def test_target_assignments(
# sample_data, target_columns, target_assignments, expected):
#     """Test that the target assignment is initialized correctly."""
#     obj = OctoData(
#         data=sample_data,
#         feature_columns=["feature_1", "feature_2"],
#         target_columns=target_columns,
#         target_assignments=target_assignments,
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert obj.target_assignments == expected


# def test_initialization():
#     data = pd.DataFrame(
#         {
#             "target": [1, 2, 1],
#             "feature2": [4, 5, 4],
#             "feature3": [7, 8, 9],
#             "sample_id": [0, 1, 2],
#         }
#     )
#     target_columns = ["target"]
#     feature_columns = ["feature2", "feature3"]
#     sample_id = "sample_id"
#     datasplit_type = "sample"

#     obj = OctoData(
#         data=data,
#         feature_columns=feature_columns,
#         target_columns=target_columns,
#         sample_id=sample_id,
#         datasplit_type=datasplit_type,
#         data_quality_check=False,
#     )

#     assert isinstance(obj, OctoData)
#     assert obj.target_columns == target_columns
#     assert obj.sample_id == sample_id
#     assert obj.datasplit_type == datasplit_type
#     assert "group_features" in obj.data.columns
#     assert "group_sample_and_features" in obj.data.columns
#     assert "row_id" in obj.data.columns


# if __name__ == "__main__":
#     pytest.main()
