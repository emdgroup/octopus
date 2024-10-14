"""Test Octo Data."""

# import numpy as np
# import pandas as pd
# import pytest

# from octopus.data import OctoData, QualityChecker


# @pytest.fixture
# def df_data():
#     """Sample data."""
#     df_data = pd.DataFrame(
#         {
#             "target_0": [0, 1, 0, 1],
#             "target_1": [5, 1, 5, 1],
#             "target_2": [5, 1, 5, 1],
#             "target_str": [0, 1, 0, "1"],
#             "feature_0": [1, 2, 3, 4],
#             "feature_1": [5, 2, 1, 4],
#             "feature_2": [5, 3, 3, 3],
#             "feature_3": [1, 2, 3, 3],
#             "feature_nan": [5, 6, 7, np.nan],
#             "feature_inf": [5, 5, 5, np.inf],
#             "feature_str": [5, 6, 7, "1"],
#             "feature_bool": [5, 6, 7, True],
#             "sample_id": [0, 1, 2, 2],
#             "sample_id_unique": [0, 1, 2, 3],
#             "id": [10, 11, 12, 13],
#             "stratification_column_str": [0, 1, 0, "1"],
#             "stratification_column_float": [0, 1, 0, 1.09],
#         }
#     )

#     return df_data


# def test_perform_checks(df_data):
#     """Check perform check for invalid inputs."""
#     with pytest.raises(ValueError):
#         OctoData(
#             data=df_data,
#             target_columns=["target_0"],
#             feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
#             sample_id="sample_id",
#             datasplit_type="sample",
#             row_id="sample_id",
#         )


# def test_perform_checks_warning(df_data):
#     """Check warnings creation."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["target_0"],
#         feature_columns=["feature_2", "feature_3"],
#         sample_id="sample_id_unique",
#         datasplit_type="group_sample_and_features",
#     )

#     warning = QualityChecker(octo_data).perform_checks()[1]
#     assert len(warning) > 0


# def test_unique_row_id(df_data):
#     """Check for unique row id."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["target_0"],
#         feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
#         sample_id="sample_id",
#         datasplit_type="sample",
#         row_id="sample_id",
#         data_quality_check=False,
#     )
#     assert QualityChecker(octo_data).unique_rowid_values() == "Row_ID is not unique"


# def test_unique_columns(df_data):
#     """Check for unique columns."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["target_0"],
#         feature_columns=["feature_0", "feature_0"],
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert (
#         QualityChecker(octo_data).unique_column(octo_data.feature_columns, "features")
#         == "The following features are not unique: feature_0"
#     )


# def test_overlap_columns(df_data):
#     """Check for overlap in columns."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["feature_0"],
#         feature_columns=["feature_0", "feature_1"],
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert (
#         QualityChecker(octo_data).overlap_columns(
#             octo_data.feature_columns,
#             octo_data.target_columns,
#             "features",
#             "targets",
#         )
#         == "Columns shared between features and targets: feature_0"
#     )


# @pytest.mark.parametrize(
#     "sample_id, datasplit_type, expected",
#     [
#         (
#             "sample_id_unique",
#             "sample",
#             (
#                 "Duplicates in features require datasplit type "
#                 "`group_features` or `group_sample_and_features`.",
#                 "Duplicates (rows) in features",
#             ),
#         ),
#         (
#             "sample_id",
#             "sample",
#             (
#                 "Duplicates in features and sample require datasplit "
#                 "type `group_sample_and_features`.",
#                 "Duplicates (rows) in features and sample",
#             ),
#         ),
#         (
#             "sample_id",
#             "group_features",
#             (
#                 "Duplicates in features and sample require datasplit "
#                 "type `group_sample_and_features`.",
#                 "Duplicates (rows) in features and sample",
#             ),
#         ),
#         (
#             "sample_id",
#             "group_sample_and_features",
#             (
#                 None,
#                 "Duplicates (rows) in features and sample",
#             ),
#         ),
#         (
#             "sample_id_unique",
#             "group_sample_and_features",
#             (None, "Duplicates (rows) in features"),
#         ),
#     ],
# )
# def test_duplicated_feature_columns(df_data, sample_id, datasplit_type, expected):
#     """Check for duplicated features and datasplit type."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["target_0"],
#         feature_columns=["feature_2", "feature_3"],
#         sample_id=sample_id,
#         datasplit_type=datasplit_type,
#         data_quality_check=False,
#     )
#     assert QualityChecker(octo_data).duplicates_features_samples() == expected


# @pytest.mark.parametrize(
#     "feature_columns, values, name, expected",
#     [
#         (["feature_nan"], [np.nan], "NaN", "NaN in columns: feature_nan"),
#         (["feature_inf"], [np.inf], "Inf", "Inf in columns: feature_inf"),
#     ],
# )
# def test_not_allowed_values(df_data, feature_columns, values, name, expected):
#     """Check for not allowed values."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=["target_0"],
#         feature_columns=feature_columns,
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#     )
#     assert QualityChecker(octo_data).not_allowed_values(values, name) == expected


# @pytest.mark.parametrize(
#     (
#         "target_columns, feature_columns, stratification_column,"
#         " name, allowed_dtypes, expected"
#     ),
#     [
#         (
#             ["target_0"],
#             ["feature_bool", "feature_str"],
#             [],
#             "Feature",
#             "iuf",
#             "Feature columns are not in types 'iuf': feature_bool, feature_str",
#         ),
#         (
#             ["target_str"],
#             ["feature_0"],
#             [],
#             "Target",
#             "iufb",
#             "Target columns are not in types 'iufb': target_str",
#         ),
#         (
#             ["target_0"],
#             ["feature_0"],
#             ["stratification_column_str", "stratification_column_float"],
#             "Stratification",
#             "iub",
#             (
#                 "Stratification columns are not in types 'iub': "
#                 "stratification_column_str, stratification_column_float"
#             ),
#         ),
#     ],
# )
# def test_check_nonnumeric(
#     df_data,
#     target_columns,
#     feature_columns,
#     stratification_column,
#     name,
#     allowed_dtypes,
#     expected,
# ):
#     """Check for non numeric values."""
#     octo_data = OctoData(
#         data=df_data,
#         target_columns=target_columns,
#         feature_columns=feature_columns,
#         sample_id="sample_id",
#         datasplit_type="sample",
#         data_quality_check=False,
#         stratification_column=stratification_column,
#     )

#     if name == "Feature":
#         columns = octo_data.feature_columns
#     if name == "Target":
#         columns = octo_data.target_columns
#     if name == "Stratification":
#         columns = octo_data.stratification_column
#     assert (
#         QualityChecker(octo_data).check_nonnumeric(name, columns, allowed_dtypes)
#         == expected
#     )
