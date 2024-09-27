from typing import List

import numpy as pd
import pandas as pd
from attrs import define, field, validators

from .checks import (
    check_column_dtypes,
    check_conflicting_labels,
    check_correlation,
    check_missing_values,
    check_mixed_data_types,
    check_single_value,
    check_string_mismatch,
    check_string_out_of_bounds,
)
from .report import Report


@define
class DataHealthChecker:
    """Quality checker class for OctoData."""

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing the dataset."""

    feature_columns: list = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    target_columns: list = field(validator=[validators.instance_of(list)])
    """List of target columns in the dataset."""

    def generate_report(
        self,
        column_dtypes=True,
        conflicting_labels=True,
        mixed_data_types=True,
        constant_value=True,
        missing_values=True,
        str_missmatch=True,
        str_out_of_bounds=True,
        correlation=True,
    ):
        report = Report()

        if column_dtypes:
            dytpe_results = check_column_dtypes(self.data)
            for col, dtype in dytpe_results.items():
                report.add_column_info(col, {"dtype": dtype})

        if constant_value:
            single_value_results = check_single_value(self.data)
            for col, is_constant in single_value_results.items():
                report.add_column_info(col, {"constant_values": is_constant})

        if conflicting_labels:
            conflicting_labels_results = check_conflicting_labels(self.data)
            for col, is_constant in conflicting_labels_results.items():
                report.add_column_info(col, {"conflicting_labels": is_constant})

        if mixed_data_types:
            mixed_data_types_results = check_mixed_data_types(self.data)
            for col, is_constant in mixed_data_types_results.items():
                report.add_column_info(col, {"mixed data types": is_constant})

        if missing_values:
            missing_values_share_result = check_missing_values(self.data)
            for col, missing_share in missing_values_share_result.items():
                report.add_column_info(col, {"missing_values_share": missing_share})

        if str_missmatch:
            str_missmatch_result = check_string_mismatch(self.data)
            for col, missing_share in str_missmatch_result.items():
                report.add_column_info(col, {"string_missmatch": missing_share})

        if str_out_of_bounds:
            str_of_of_bounds_result = check_string_out_of_bounds(self.data)
            for col, missing_share in str_of_of_bounds_result.items():
                report.add_column_info(col, {"string_out_of_bounds": missing_share})
        if correlation:
            correlation_results = check_correlation(self.data)
            for col, correlation in correlation_results.items():
                report.add_column_info(col, {"high_correlation": correlation})
        return report

    # def perform_checks(self):
    #     """Perform all quality checks."""
    #     report = []
    #     warning = []

    #     # # Check unique row id
    #     if unique_row_id := self.unique_rowid_values():
    #         report.append(unique_row_id)

    #     # Check unique features
    #     if unique_features := self.unique_column(
    #         self.octo_data.feature_columns, "features"
    #     ):
    #         report.append(unique_features)

    #     # Check unique targets
    #     if unique_targets := self.unique_column(
    #         self.octo_data.target_columns, "features"
    #     ):
    #         report.append(unique_targets)

    #     # check if features or targets overlap
    #     if overlap_feat_targ := self.overlap_columns(
    #         self.octo_data.feature_columns,
    #         self.octo_data.target_columns,
    #         "features",
    #         "targets",
    #     ):
    #         report.append(overlap_feat_targ)

    #     # check if features or sample_id overlap
    #     if overlap_feat_sample := self.overlap_columns(
    #         self.octo_data.feature_columns,
    #         [self.octo_data.sample_id],
    #         "features",
    #         "sample_id",
    #     ):
    #         report.append(overlap_feat_sample)

    #     # # check if targets or sample_id overlap
    #     if overlap_targ_sample := self.overlap_columns(
    #         self.octo_data.target_columns,
    #         [self.octo_data.sample_id],
    #         "targets",
    #         "sample_id",
    #     ):
    #         report.append(overlap_targ_sample)

    #     # missing columns in dataframe
    #     # is is not working anyway, because octodata can not be created
    #     if missing_columns := self.missing_columns():
    #         report.append(missing_columns)

    #     # check for duplicates in features and sample
    #     if datasplit_required := self.duplicates_features_samples():
    #         if datasplit_required[0]:
    #             report.append(datasplit_required[0])
    #         if datasplit_required[1]:
    #             warning.append(datasplit_required[1])

    #     # check values for Infs and NaN
    #     if nan := self.not_allowed_values([np.nan], "NaN"):
    #         report.append(nan)

    #     if infs := self.not_allowed_values([np.inf, -np.inf], "Inf"):
    #         report.append(infs)

    #     # check dtypes
    #     if dtype_feature := self.check_nonnumeric(
    #         "Feature", self.octo_data.feature_columns, "iuf"
    #     ):
    #         report.append(dtype_feature)
    #     if dtype_target := self.check_nonnumeric(
    #         "Target", self.octo_data.target_columns, "iufb"
    #     ):
    #         report.append(dtype_target)
    #     if dtype_stratifictaion := self.check_nonnumeric(
    #         "Stratification ", self.octo_data.stratification_column, "iub"
    #     ):
    #         report.append(dtype_stratifictaion)

    #     return report, warning

    # def _relevant_columns(self) -> set:
    #     """Get relevant columns for checks."""
    #     return (
    #         set(self.octo_data.feature_columns)
    #         .union(set(self.octo_data.target_columns))
    #         .union({self.octo_data.sample_id, self.octo_data.row_id})
    #         .union(set(self.octo_data.stratification_column))
    #     )

    # def unique_rowid_values(self) -> str | None:
    #     """Check if values of row_id are unique."""
    #     if not self.octo_data.data[self.octo_data.row_id].is_unique:
    #         return "Row_ID is not unique"
    #     return None

    # def unique_column(self, columns: List[str], column_type: str | None) -> None:
    #     """Add non-unique columns check to the report."""
    #     non_unique_columns = list({item for item in columns if columns.count(item) > 1})
    #     if non_unique_columns:
    #         return (
    #             f"The following {column_type} are not unique: "
    #             f"{', '.join(non_unique_columns)}"
    #         )
    #     return None

    # def overlap_columns(
    #     self,
    #     list1: List[str],
    #     list2: List[str],
    #     name1: str,
    #     name2: str,
    # ) -> None:
    #     """Add overlap check between feature and target columns to the report."""
    #     overlapping_columns = set(list1).intersection(list2)
    #     if overlapping_columns:
    #         return (
    #             f"Columns shared between {name1} and {name2}: "
    #             f"{', '.join(overlapping_columns)}"
    #         )
    #     return None

    # def missing_columns(self) -> str | None:
    #     """Adding missing columns in dataframe to the report."""
    #     missing_columns = list(
    #         self._relevant_columns() - set(self.octo_data.data.columns)
    #     )
    #     if missing_columns:
    #         return f"Missing columns in dataset: {', '.join(missing_columns)}"
    #     return None

    # def duplicates_features_samples(self):
    #     """Check for duplicates (rows) in all features."""
    #     duplicated_features = (
    #         self.octo_data.data[self.octo_data.feature_columns].duplicated().any()
    #     )
    #     duplicated_features_and_sample = (
    #         self.octo_data.data[
    #             list(self.octo_data.feature_columns) + [self.octo_data.sample_id]
    #         ]
    #         .duplicated()
    #         .any()
    #     )
    #     if duplicated_features and not duplicated_features_and_sample:
    #         if self.octo_data.datasplit_type == "sample":
    #             return (
    #                 "Duplicates in features require datasplit type "
    #                 "`group_features` or `group_sample_and_features`.",
    #                 "Duplicates (rows) in features",
    #             )
    #         return (None, "Duplicates (rows) in features")

    #     if duplicated_features_and_sample:
    #         if self.octo_data.datasplit_type != "group_sample_and_features":
    #             return (
    #                 "Duplicates in features and sample require datasplit "
    #                 "type `group_sample_and_features`.",
    #                 "Duplicates (rows) in features and sample",
    #             )
    #         return (
    #             None,
    #             "Duplicates (rows) in features and sample",
    #         )
    #     return None

    # def not_allowed_values(self, values: List, name: str) -> str | None:
    #     """Check if all relevant columns are free of Infs."""
    #     columns_with_nan = [
    #         col
    #         for col in list(self._relevant_columns())
    #         if self.octo_data.data[col].isin(values).any()
    #     ]
    #     if columns_with_nan:
    #         return f"{name} in columns: {', '.join(columns_with_nan)}"

    # def check_nonnumeric(
    #     self, name: str, columns: List[str], allowed_dtypes: str
    # ) -> str | None:
    #     """Check if specified columns contain only allowed data types."""
    #     # Initialize list to store non-numeric column names
    #     non_numeric_columns = []

    #     # Check each column against the allowed data types
    #     for column in columns:
    #         if self.octo_data.data[column].dtype.kind not in allowed_dtypes:
    #             non_numeric_columns.append(column)

    #     if non_numeric_columns:
    #         return (
    #             f"{name} columns are not in types '{allowed_dtypes}': "
    #             f"{', '.join(non_numeric_columns)}"
    #         )
    #     return None
