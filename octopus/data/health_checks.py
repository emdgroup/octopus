"""Data Health Check."""

import pandas as pd
from attrs import define, field, validators

from .checks import (
    check_column_dtypes,
    check_duplicated_features,
    check_duplicated_rows,
    check_feature_feature_correlation,
    check_identical_features,
    check_infinity_values,
    check_missing_values,
    check_mixed_data_types,
    check_outlier_detection,
    check_single_value,
    check_string_mismatch,
    check_string_out_of_bounds,
    check_unique_column_names,
    check_unique_row_id_values,
)
from .report import DataHealthReport


@define
class DataHealthChecker:
    """Quality checker class for OctoData."""

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing the dataset."""

    feature_columns: list = field(
        default=None, validator=validators.optional(validators.instance_of(list))
    )
    """List of all feature columns in the dataset."""

    target_columns: list = field(
        default=None, validator=validators.optional(validators.instance_of(list))
    )
    """List of target columns in the dataset."""

    row_id: str = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    """Row identifier."""

    sample_id: str = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    """Row identifier."""

    def __attrs_post_init__(self):
        if self.feature_columns is not None or self.target_columns is not None:
            self._check_columns_exist()
            self.data = self.data[
                self.feature_columns
                + self.target_columns
                + [self.row_id]
                + [self.sample_id]
            ]

    def generate_report(
        self,
        column_dtypes=True,
        idendical_features=True,
        mixed_data_types=True,
        single_value=True,
        missing_values=True,
        str_mismatch=True,
        str_out_of_bounds=True,
        feature_feature_correlation=True,
        unique_column_names=True,
        unique_row_id_values=True,
        outlier_detection=True,
        duplicated_features=True,
        duplicated_rows=True,
        infinity_values=True,
    ):
        """Generate data health report."""
        report = DataHealthReport()

        # Adding multiple column info
        if column_dtypes:
            res_dtype = check_column_dtypes(self.data)
            report.add_multiple(
                "columns",
                {c: {"object/categorical dtype": v} for c, v in res_dtype.items()},
            )

        if single_value:
            res_single = check_single_value(self.data)
            report.add_multiple(
                "columns", {c: {"single_values": v} for c, v in res_single.items()}
            )

        if idendical_features:
            if self.feature_columns is not None:
                res_ident = check_identical_features(self.data, self.feature_columns)
                report.add_multiple(
                    "columns",
                    {c: {"identical_features": v} for c, v in res_ident.items()},
                )

        if mixed_data_types:
            res_mixed = check_mixed_data_types(self.data)
            report.add_multiple(
                "columns", {c: {"mixed data types": v} for c, v in res_mixed.items()}
            )

        if missing_values:
            res_miss = check_missing_values(self.data)
            report.add_multiple(
                "columns", {c: {"missing values share": v} for c, v in res_miss.items()}
            )

        if infinity_values:
            res_inf = check_infinity_values(self.data)
            report.add_multiple(
                "columns", {c: {"infinity values share": v} for c, v in res_inf.items()}
            )

        if str_mismatch:
            res_mismatch = check_string_mismatch(self.data)
            report.add_multiple(
                "columns",
                {c: {"string mismatch": v} for c, v in res_mismatch.items()},
            )

        if str_out_of_bounds:
            res_str_bounds = check_string_out_of_bounds(self.data)
            report.add_multiple(
                "columns",
                {c: {"string out of bounds": v} for c, v in res_str_bounds.items()},
            )

        if feature_feature_correlation:
            if self.feature_columns is not None:
                res_feat_cor = check_feature_feature_correlation(
                    self.data, self.feature_columns
                )
                report.add_multiple(
                    "columns",
                    {
                        c: {"high feature correlation": v}
                        for c, v in res_feat_cor.items()
                    },
                )

        if unique_column_names:
            if (
                self.feature_columns is not None
                or self.target_columns is not None
                or self.row_id is not None
            ):
                res_unique_col = check_unique_column_names(
                    self.feature_columns, self.target_columns, self.row_id
                )
                report.add_multiple(
                    "columns",
                    {c: {"unique colume name": v} for c, v in res_unique_col.items()},
                )

        if unique_row_id_values:
            if self.row_id is not None:
                res_row_id = check_unique_row_id_values(self.data, self.row_id)
                if res_row_id is not None:
                    report.add_multiple(
                        "columns",
                        {c: {"unique row id": v} for c, v in res_row_id.items()},
                    )

        if duplicated_features:
            if self.feature_columns is not None or self.target_columns is not None:
                res_dup_features, res_dup_features_samples = check_duplicated_features(
                    self.data, self.feature_columns, self.sample_id
                )
                if res_dup_features not in (None, False):
                    report.add("rows", "duplicated_features", res_dup_features)
                if res_dup_features_samples not in (None, False):
                    report.add(
                        "rows", "duplicated_features_samples", res_dup_features_samples
                    )

        if duplicated_rows:
            res_dup_rows = check_duplicated_rows(self.data)
            if res_dup_rows is not None:
                report.add("rows", "duplicated_rows", res_dup_rows)

        if outlier_detection:
            res_outliers = check_outlier_detection(self.data)
            if not res_outliers.empty:
                report.add("outliers", "scores", res_outliers.to_dict("records"))

        return report

    def _check_columns_exist(self):
        """Check if all relevant columns exists."""
        relevant_columns = self.feature_columns + self.target_columns
        if self.row_id is not None:
            relevant_columns.append(self.row_id)
        missing_columns = [
            col for col in relevant_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"The following columns are missing in the DataFrame: {', '.join(missing_columns)}"  # noqa: E501
            )
