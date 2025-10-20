"""OctoData Health Checker."""

import re
from itertools import combinations

import numpy as np
import pandas as pd
from attrs import define, field
from rapidfuzz import fuzz


@define
class OctoDataHealthChecker:
    """OctoDataHealthChecker."""

    data: pd.DataFrame
    feature_columns: list[str] | None
    target_columns: list[str] | None
    row_id: str | None
    sample_id: str | None
    stratification_column: str | None

    issues: list[dict] = field(factory=list)

    def add_issue(
        self,
        category: str,
        issue_type: str,
        affected_items: list[str],
        severity: str,
        description: str,
        action: str,
    ):
        """Add a health issue to the report."""
        issue = {
            "Category": category,
            "Issue Type": issue_type,
            "Affected Items": ", ".join(affected_items),
            "Severity": severity,
            "Description": description,
            "Recommended Action": action,
        }
        self.issues.append(issue)

    def generate_report(self):
        """Generate the full health report."""
        self._check_critical_column_missing_values()
        self._check_feature_column_missing_values()
        self._check_row_missing_values()
        self._check_int_col_with_few_uniques()
        self._check_duplicated_features()
        self._check_feature_feature_correlation()
        self._check_identical_features()
        self._check_duplicated_rows()
        self._check_infinity_values()
        self._check_string_mismatch()
        # self._check_string_out_of_bounds() # see issue#60

        return pd.DataFrame(self.issues)

    def _check_critical_column_missing_values(self):
        """Check for missing values in critical columns (target, sample_id, row_id)."""
        missing_value_share_col = self.data.isnull().mean()

        critical_columns = [
            *self.target_columns,
            self.sample_id,
            self.row_id,
            self.stratification_column,
        ]
        critical_columns = [col for col in critical_columns if col is not None]

        critical_missing = [col for col in critical_columns if missing_value_share_col.get(col, 0) > 0]
        if critical_missing:
            self.add_issue(
                category="columns",
                issue_type="critical_missing_values",
                affected_items=critical_missing,
                severity="Critical",
                description=("These critical columns (target, sample_id, or row_id) have missing values."),
                action=(
                    "Investigate and resolve missing values in these columns immediately. These are crucial for model training and data integrity."
                ),
            )

    def _check_feature_column_missing_values(self):
        """Check for missing values in feature columns."""
        missing_value_share_col = self.data.isnull().mean(axis=0)

        high_missing_cols = [col for col in self.feature_columns if missing_value_share_col.get(col, 0) > 0.25]
        low_missing_cols = [col for col in self.feature_columns if 0 < missing_value_share_col.get(col, 0) <= 0.25]

        if high_missing_cols:
            self.add_issue(
                category="columns",
                issue_type="high_missing_values",
                affected_items=high_missing_cols,
                severity="Critical",
                description="These feature columns have more than 10% missing values.",
                action=("Consider removing these columns or using advanced imputation techniques."),
            )

        if low_missing_cols:
            self.add_issue(
                category="columns",
                issue_type="low_missing_values",
                affected_items=low_missing_cols,
                severity="Info",
                description="These feature columns have some missing values (<=10%).",
                action="Consider appropriate imputation methods for these columns.",
            )

    def _check_row_missing_values(self):
        """Check for missing values in rows."""
        missing_value_share_row = self.data.isnull().mean(axis=1)

        high_missing_rows = missing_value_share_row[missing_value_share_row > 0.5]
        low_missing_rows = missing_value_share_row[(missing_value_share_row > 0) & (missing_value_share_row <= 0.5)]

        if not high_missing_rows.empty:
            self.add_issue(
                category="rows",
                issue_type="high_missing_values",
                affected_items=[str(idx) for idx in high_missing_rows.index],
                severity="Critical",
                description=(f"{len(high_missing_rows)} rows have more than 10% missing values."),
                action=("Consider removing these rows or using advanced imputation techniques."),
            )

        if not low_missing_rows.empty:
            self.add_issue(
                category="rows",
                issue_type="low_missing_values",
                affected_items=[str(idx) for idx in low_missing_rows.index],
                severity="Info",
                description=(f"{len(low_missing_rows)} rows have some missing values (<=10%)."),
                action="Review these rows and consider appropriate imputation methods.",
            )

    def _check_int_col_with_few_uniques(self, threshold: int = 5):
        """Check for integer columns with a small number of unique elements."""
        int_cols_with_few_uniques = {
            col: self.data[col].nunique()
            for col in self.feature_columns
            if pd.api.types.is_integer_dtype(self.data[col]) and 2 < self.data[col].nunique() <= threshold
        }

        if int_cols_with_few_uniques:
            affected_items = list(int_cols_with_few_uniques.keys())
            self.add_issue(
                category="columns",
                issue_type="int_columns_with_few_uniques",
                affected_items=affected_items,
                severity="Warning",
                description=(
                    f"These integer columns have between 3 and {threshold} unique values: {', '.join(affected_items)}"
                ),
                action=(
                    "Consider converting these columns to categorical variables or investigate if they should be ordinal features."
                ),
            )

    def _check_duplicated_features(self):
        """Check for duplicates (rows) in all features."""
        duplicated_features = self.data[self.feature_columns].duplicated().any()

        if self.sample_id is not None:
            duplicated_features_and_sample = self.data[[*self.feature_columns, self.sample_id]].duplicated().any()
        else:
            duplicated_features_and_sample = None

        if duplicated_features:
            self.add_issue(
                category="rows",
                issue_type="duplicated_features",
                affected_items=["all_features"],
                severity="Warning",
                description=("There are duplicated rows when considering all feature columns."),
                action=("Investigate the cause of these duplicates and consider removing or consolidating them."),
            )

        if duplicated_features_and_sample:
            self.add_issue(
                category="rows",
                issue_type="duplicated_features_and_sample",
                affected_items=["all_features_and_sample_id"],
                severity="Critical",
                description=("There are duplicated rows when considering all feature columns and the sample ID."),
                action=(
                    "This is a critical issue. Investigate and resolve these duplicates immediately as they may indicate data integrity problems."
                ),
            )

    def _check_feature_feature_correlation(self, method: str = "pearson", threshold: float = 0.8):
        """Find columns in the DataFrame that are highly correlated."""
        # Filter only numeric columns
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[float, int]).columns
        corr_matrix = self.data[numeric_features].corr(method=method)

        # Dictionary to store the columns with high correlation
        highly_correlated = {}

        # Iterate over the correlation matrix and find highly correlated columns
        for col in corr_matrix.columns:
            for row in corr_matrix.index:
                if col != row and abs(corr_matrix.loc[row, col]) > threshold:
                    if col not in highly_correlated:
                        highly_correlated[col] = set()
                    if row not in highly_correlated:
                        highly_correlated[row] = set()
                    highly_correlated[col].add(row)
                    highly_correlated[row].add(col)

        # Merge overlapping groups
        merged_groups = []
        for feature, correlated_features in highly_correlated.items():
            new_group = set(correlated_features) | {feature}
            merged = False
            for group in merged_groups:
                if not new_group.isdisjoint(group):
                    group.update(new_group)
                    merged = True
                    break
            if not merged:
                merged_groups.append(new_group)

        # Create issues for each merged group of highly correlated features
        for group in merged_groups:
            correlation_details = []
            for feat1, feat2 in combinations(sorted(group), 2):
                corr_value = corr_matrix.loc[feat1, feat2]
                if abs(corr_value) > threshold:
                    correlation_details.append(f"{feat1} - {feat2} ({corr_value:.2f})")

            correlation_description = ", ".join(correlation_details)

            self.add_issue(
                category="features",
                issue_type="high_correlation",
                affected_items=sorted(group),
                severity="Info",
                description=(f"The following features are highly correlated (>{threshold}): {correlation_description}"),
                action=("Consider removing or combining these highly correlated features to reduce multicollinearity."),
            )

    def _check_identical_features(self):
        """Identify features that have identical values but different column names."""
        identical_features = {col: [] for col in self.feature_columns}

        for col in self.feature_columns:
            for other_col in self.feature_columns:
                if col != other_col and self.data[col].equals(self.data[other_col]):
                    identical_features[col].append(other_col)

        # Remove entries with empty lists
        identical_features = {k: v for k, v in identical_features.items() if v}

        # Create issues for each group of identical features
        for feature, identical_list in identical_features.items():
            self.add_issue(
                category="features",
                issue_type="identical_features",
                affected_items=[feature, *identical_list],
                severity="Warning",
                description=(
                    f"The feature '{feature}' is identical to the following feature(s): {', '.join(identical_list)}"
                ),
                action=("Consider removing redundant features to simplify the dataset and improve model performance."),
            )

    def _check_duplicated_rows(self):
        """Check all duplicated rows."""
        duplicated_mask = self.data.duplicated()
        duplicated_rows = self.data[duplicated_mask]

        if not duplicated_rows.empty:
            num_duplicates = len(duplicated_rows)
            self.add_issue(
                category="rows",
                issue_type="duplicated_rows",
                affected_items=[str(idx) for idx in duplicated_rows.index],
                severity="Warning",
                description=f"Found {num_duplicates} duplicated row(s) in the dataset.",
                action=(
                    "Investigate these duplicates and consider removing or consolidating them based on your data requirements."
                ),
            )

    def _check_infinity_values(self):
        """Check for infinity values in the DataFrame."""
        # Ensure all data is numeric before checking for infinity
        numeric_df = self.data[self.feature_columns].apply(pd.to_numeric, errors="coerce")

        # Check for positive and negative infinity values
        infinity_mask = numeric_df.map(lambda x: np.isinf(x))

        # Calculate the proportion of infinity values in each column
        infinity_value_share = infinity_mask.mean()

        # Create a dictionary for columns with infinity values
        infinity_value_dict = {col: share for col, share in infinity_value_share.items() if share > 0}

        if infinity_value_dict:
            affected_items = list(infinity_value_dict.keys())
            description = "The following columns contain infinity values:\n"
            for col, share in infinity_value_dict.items():
                percentage = share * 100
                description += f"- {col}: {percentage:.2f}% of values\n"

            self.add_issue(
                category="columns",
                issue_type="infinity_values",
                affected_items=affected_items,
                severity="Info",
                description=description.strip(),
                action=(
                    "Investigate these columns and decide on an appropriate "
                    "strategy to handle infinity values (e.g., imputation, "
                    "removal, or capping)."
                ),
            )

    def _check_string_mismatch(self):
        """Find unique groups of similar strings, ignoring numeric suffixes."""
        string_mismatch = {}

        def remove_numbers(entry):
            """Remove numbers from the end of a string."""
            return re.sub(r"\d+$", "", str(entry))

        def determine_threshold(length):
            """Determine the similarity threshold based on the length of the string."""
            if length <= 7:
                return 80  # Lower threshold for shorter strings
            elif 7 <= length <= 12:
                return 85  # Medium threshold for medium-length strings
            else:
                return 90  # Higher threshold for longer strings

        def is_all_integers(series):
            """Check if all non-null values in a series are integers."""
            return series.dropna().apply(lambda x: str(x).isdigit()).all()

        for column in self.feature_columns:
            if self.data[column].dtype == object or self.data[column].dtype.name == "category":
                if is_all_integers(self.data[column]):
                    continue

                try:
                    # Remove numbers from the end of each entry
                    column_values = self.data[column].dropna().apply(remove_numbers).unique()
                    # Check if the column has more than one unique value
                    if len(column_values) > 2:
                        # Initialize a set to keep track of processed strings
                        processed = set()
                        similar_groups = []

                        for value in column_values:
                            if value not in processed:
                                threshold = determine_threshold(len(value))
                                # Find all similar strings to the current value,
                                # excluding identical strings
                                similar = {
                                    other
                                    for other in column_values
                                    if value != other and fuzz.ratio(value, other) >= threshold
                                }
                                if similar:
                                    similar.add(value)
                                    similar_groups.append(list(similar))
                                    processed.update(similar)
                        if similar_groups:
                            string_mismatch[column] = similar_groups
                except Exception as e:
                    print(f"An error occurred while processing column {column}: {e}")

        if string_mismatch:
            for column, similar_groups in string_mismatch.items():
                description = (
                    f"Column '{column}' contains similar strings that might be misspellings or inconsistencies:\n"
                )
                for group in similar_groups:
                    description += f"- {', '.join(group)}\n"

                self.add_issue(
                    category="columns",
                    issue_type="string_mismatch",
                    affected_items=[column],
                    severity="Warning",
                    description=description.strip(),
                    action=(
                        "Review these similar strings and consider standardizing them to improve data consistency."
                    ),
                )

    def _check_string_out_of_bounds(self, length_threshold_factor=2):
        """Find strings that are significantly longer than the average length."""
        long_string = {}
        for column in self.feature_columns:
            if self.data[column].dtype == object or self.data[column].dtype.name == "category":
                try:
                    column_values = self.data[column].dropna().tolist()  # Drop NaN values and convert to list

                    # Calculate the average length of strings in the column
                    avg_length = sum(len(str(value)) for value in column_values) / len(column_values)

                    # Identify strings that are significantly longer than the average
                    long_strings = [
                        value for value in column_values if len(str(value)) > length_threshold_factor * avg_length
                    ]
                    if long_strings:
                        long_string[column] = long_strings
                except:  # noqa: E722
                    pass

        if long_string:
            for column, long_strings in long_string.items():
                description = (
                    f"Column '{column}' contains strings that are significantly longer than the average length:\n"
                )
                for value in long_strings[:5]:  # Limit to first 5 for brevity
                    description += f"- {value[:50]}{'...' if len(value) > 50 else ''}\n"
                if len(long_strings) > 5:
                    description += f"(and {len(long_strings) - 5} more...)\n"

                self.add_issue(
                    category="columns",
                    issue_type="string_out_of_bounds",
                    affected_items=[column],
                    severity="Warning",
                    description=description.strip(),
                    action=(
                        "Review these unusually long strings and consider if they are valid or if they need cleaning or truncation."
                    ),
                )
