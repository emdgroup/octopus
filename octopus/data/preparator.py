"""OctoData Preparator."""

import numpy as np
import pandas as pd
from attrs import define

from ..logger import get_logger

logger = get_logger()

DEFAULT_NULL_VALUES = {"none", "null", "nan", "na", "", "\x00", "\x00\x00", "n/a"}
DEFAULT_INF_VALUES = {"inf", "infinity", "∞"}


@define
class OctoDataPreparator:
    """Validator for OctoData."""

    data: pd.DataFrame
    """DataFrame containing the dataset."""

    feature_columns: list[str]
    """List of all feature columns in the dataset."""

    target_columns: list[str]
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    sample_id: str
    """Identifier for sample instances."""

    row_id: str | None
    """Unique row identifier."""

    target_assignments: dict[str, str]
    """Mapping of target assignments."""

    def prepare(self):
        """Run all data preparation steps."""
        self._sort_features()
        self._standardize_null_values()
        self._standardize_inf_values()
        self._set_target_assignments()
        self._remove_singlevalue_features()
        self._transform_bool_to_int()
        self._create_row_id()
        self._add_group_features()  # see issue57 # needs to be done at the end
        return self.data, self.feature_columns, self.row_id, self.target_assignments

    def _sort_features(self):
        """Sort feature columns deterministically by length and lexicographically.

        This ensures that the results are always the same, preventing minor differences.
        """
        self.feature_columns = sorted(map(str, self.feature_columns), key=lambda x: (len(x), x))
        logger.info("Sorted features.")

    def _set_target_assignments(self):
        """Set default target assignments or validates provided ones.

        If there is one target column and no target assignments, assigns "default"
        to the first target.
        If there are multiple target columns, ensures the target assignments
        match their count.
        Raises a ValueError if assignments are incorrect or missing.
        """
        if (len(self.target_columns) == 1) & (not self.target_assignments):
            self.target_assignments["default"] = self.target_columns[0]

    def _remove_singlevalue_features(self):
        """Remove features that contain only a single unique value."""
        original_features = self.feature_columns.copy()

        # Keep only features with more than one unique value
        self.feature_columns = [feature for feature in self.feature_columns if self.data[feature].nunique() > 1]

        # Calculate removed features
        removed_features = list(set(original_features) - set(self.feature_columns))

        if removed_features:
            logger.info(f"Removed {len(removed_features)} features due to single unique values: {removed_features}")

    def _transform_bool_to_int(self):
        """Convert all boolean columns to integer."""
        bool_cols = self.data.select_dtypes(include="bool").columns
        if not bool_cols.empty:
            self.data[bool_cols] = self.data[bool_cols].astype(int)
            logger.info("Transformed %d boolean columns to int.", len(bool_cols))

    def _create_row_id(self):
        """Assign a unique row identifier to each entry in the DataFrame.

        If no `row_id` is provided, this method adds a `row_id` column using
        the DataFrame index.
        """
        if not self.row_id:
            if "row_id" in self.data.columns:
                raise Exception("""`row_id` is not allowed as column name.""")

            self.data["row_id"] = self.data.index
            self.row_id = "row_id"

            logger.info("Added `row_id`")

    def _add_group_features(self):
        """Initialize the preparation of the DataFrame by adding group feature columns.

        This method performs the following operations on the data:

        - Adds a column `group_features` to group entries with
            the same features.
        - Adds a column `group_sample_and_features` to group entries
            with the same features
        and/or the same `sample_id`.
        - Resets the DataFrame index.

        Example:
            If you have a DataFrame with features and sample IDs, this
            method will add columns to help identify and group rows with
            similar characteristics.

        """

        def custom_groupby(df, columns):
            """Create custom groupby that handles also None and Inf."""
            # Replace None with a placeholder for grouping
            df_for_grouping = df[columns].fillna("None_placeholder")

            # Convert infinite values to strings for consistent grouping
            for col in df_for_grouping.select_dtypes(include=[np.number]).columns:
                df_for_grouping[col] = df_for_grouping[col].replace([np.inf, -np.inf], ["inf", "-inf"])

            # Perform the groupby operation
            return df_for_grouping.groupby(columns).ngroup()

        self.data = (
            self.data
            # Create group with the same features
            .assign(group_features=lambda df_: custom_groupby(df_, self.feature_columns))
            # Create group with the same features and/or the same sample_id
            .assign(
                group_sample_and_features=lambda df_: df_.apply(
                    lambda row: pd.concat(
                        [
                            df_[df_[self.sample_id] == row[self.sample_id]],
                            df_[df_["group_features"] == row["group_features"]],
                        ]
                    ).index.min(),
                    axis=1,
                )
            )
            .reset_index(drop=True)
        )
        logger.info("Added `group_feaures` and `group_sample_features`")

    def _standardize_null_values(self):
        """Standardize null values across the DataFrame."""
        null_values_case_insensitive = {val.lower() for val in DEFAULT_NULL_VALUES}

        # Function to replace string null values with np.nan (case-insensitive)
        def replace_null(x):
            if isinstance(x, str) and x.strip().lower() in null_values_case_insensitive:
                return np.nan
            return x

        # Apply the replacement to all columns and check if any changes were made
        original_data = self.data.copy()
        self.data = self.data.map(replace_null)

        # Check if any values were actually replaced
        if not self.data.equals(original_data):
            logger.info("Standardized null values across the DataFrame.")

    def _standardize_inf_values(self):
        """Standardize inf values across the DataFrame."""
        # Create a case-insensitive version of infinity_values
        infinity_values_case_insensitive = {val.lower() for val in DEFAULT_INF_VALUES}

        # Replace string infinity values with np.inf or -np.inf (case-insensitive)
        def replace_infinity(x):
            if isinstance(x, str):
                if x.strip().lower() in infinity_values_case_insensitive:
                    return np.inf
                elif x.strip().lower() in {f"-{val}" for val in infinity_values_case_insensitive}:
                    return -np.inf
            return x

        original_data = self.data.copy()
        self.data = self.data.map(replace_infinity)

        # Check if any values were actually replaced
        if not self.data.equals(original_data):
            logger.info("Standardized infinity values across the DataFrame.")
