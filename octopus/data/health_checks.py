"""Data Health Check."""

from typing import Optional

import pandas as pd
from attrs import Factory, define, field, validators

from .checks import (
    check_duplicated_features,
    check_duplicated_rows,
    check_feature_feature_correlation,
    check_identical_features,
    check_infinity_values,
    check_int_col_with_few_uniques,
    check_missing_values,
    check_mixed_data_types,
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
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Row identifier."""

    stratification_column: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """List of columns used for stratification."""

    def __attrs_post_init__(self):
        # Create a list of all potential columns
        potential_columns = (
            self.feature_columns
            + self.target_columns
            + [self.row_id]
            + [self.sample_id]
            + [self.stratification_column]
        )
        relevant_columns = list(
            set(col for col in potential_columns if col is not None)
        )
        self._check_columns_exist(relevant_columns)
        self.data = self.data[relevant_columns]

    def generate_report(
        self,
        stratification_dtype=True,
        identical_features=True,
        mixed_data_types=True,
        single_value=True,
        missing_values=True,
        str_mismatch=True,
        str_out_of_bounds=True,
        feature_feature_correlation=True,
        unique_column_names=True,
        unique_row_id_values=True,
        duplicated_features=True,
        duplicated_rows=True,
        infinity_values=True,
        few_int_values=True,
    ):
        """Generate data health report."""
        report = DataHealthReport()

        if single_value:
            res_single = check_single_value(self.data)
            report.add_multiple(
                "columns", {c: {"single_values": v} for c, v in res_single.items()}
            )

        if identical_features:
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
            res_miss_col, res_miss_row = check_missing_values(self.data)
            report.add_multiple(
                "columns",
                {c: {"missing values share": v} for c, v in res_miss_col.items()},
            )
            report.add_multiple(
                "rows",
                {c: {"missing values share": v} for c, v in res_miss_row.items()},
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
                res_feat_cor_pearson = check_feature_feature_correlation(
                    self.data, self.feature_columns, method="pearson"
                )
                report.add_multiple(
                    "columns",
                    {
                        c: {"high feature correlation (pearson)": v}
                        for c, v in res_feat_cor_pearson.items()
                    },
                )

                res_feat_cor_spearman = check_feature_feature_correlation(
                    self.data, self.feature_columns, method="spearman"
                )
                report.add_multiple(
                    "columns",
                    {
                        c: {"high feature correlation (spearman)": v}
                        for c, v in res_feat_cor_spearman.items()
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
                    {c: {"unique column name": v} for c, v in res_unique_col.items()},
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

        if few_int_values:
            res_few_unique_int_values = check_int_col_with_few_uniques(
                self.data, self.feature_columns
            )
            if res_few_unique_int_values is not None:
                report.add_multiple(
                    "columns",
                    {
                        c: {"unique_int_values": v}
                        for c, v in res_few_unique_int_values.items()
                    },
                )
        return report

    def _check_columns_exist(self, relevant_columns):
        """Check if all relevant columns exists."""
        missing_columns = [
            col for col in relevant_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"The following columns are missing in the DataFrame:"
                f" {', '.join(missing_columns)}"  # noqa: E501
            )
