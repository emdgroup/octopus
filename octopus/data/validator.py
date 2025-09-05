"""OctoData Validator."""

import re
from typing import Dict, List, Optional

import pandas as pd
from attrs import define


@define
class OctoDataValidator:
    """Validator for OctoData."""

    data: pd.DataFrame
    """DataFrame containing the dataset."""

    feature_columns: List[str]
    """List of all feature columns in the dataset."""

    target_columns: List[str]
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    sample_id: str
    """Identifier for sample instances."""

    row_id: Optional[str]
    """Unique row identifier."""

    stratification_column: Optional[str]
    """List of columns used for stratification."""

    target_assignments: Dict[str, str]
    """Mapping of target assignments."""

    relevant_columns: List[str]
    """Relevant columns of the dataset."""

    def validate(self):
        """Run all validation checks."""
        self._validate_column_names()
        self._validate_column_names_characters()
        self._validate_columns_exist()
        self._check_for_duplicated_columns()
        self._validate_stratification_column()
        self._validate_target_assignments()
        self._validate_number_of_targets()
        self._validate_column_dtypes()
        self._validate_row_id_unique()

    def _validate_columns_exist(self):
        """Validate that all columns exists in the dataframe."""
        # Identify missing columns
        missing_columns = [col for col in self.relevant_columns if col not in self.data.columns]

        # Raise error if any columns are missing
        if missing_columns:
            missing_str = ", ".join(missing_columns)
            raise ValueError(f"Columns not found in the DataFrame: {missing_str}")

    def _check_for_duplicated_columns(self):
        # Combine all columns to check for duplicates
        columns_to_check = self.feature_columns + self.target_columns + [self.sample_id]

        if self.row_id:
            columns_to_check.append(self.row_id)

        # Check for duplicates
        duplicates = set([col for col in columns_to_check if columns_to_check.count(col) > 1])

        if duplicates:
            duplicates_str = ", ".join(duplicates)
            raise ValueError(f"Duplicate columns found: {duplicates_str}")

    def _validate_stratification_column(self):
        """Validate if stratification_column is not the same as row_id or sample_id."""
        if self.stratification_column and self.stratification_column in [
            self.sample_id,
            self.row_id,
        ]:
            raise ValueError("Stratification column cannot be the same as sample_id or row_id")

    def _validate_target_assignments(self):
        """Validate target_assignments.

        Values need to be unique and within target_columns.
        """
        if len(self.target_columns) == 1:
            if self.target_assignments:
                raise ValueError("Target assignments provided for a single target column. Assignments are only needed for multiple target columns.")
            return

        # Multiple target columns: validate assignments
        if not self.target_assignments:
            raise ValueError(
                f"Multiple target columns detected ({len(self.target_columns)}), "
                "but no target assignments provided. "
                f"Please specify assignments for: {', '.join(self.target_columns)}"
            )

        # Check if all target columns are assigned
        missing_assignments = set(self.target_columns) - set(self.target_assignments.values())
        if missing_assignments:
            raise ValueError(
                "Missing assignments for target column(s): "
                f"{', '.join(missing_assignments)}. "
                "Please provide assignments for all target columns: "
                f"{', '.join(self.target_columns)}"
            )

        # Check if all assignments are valid target columns
        invalid_assignments = set(self.target_assignments.values()) - set(self.target_columns)
        if invalid_assignments:
            raise ValueError(
                "Invalid assignment key(s) detected: "
                f"{', '.join(invalid_assignments)}. Assignments must be made "
                f"only for existing target columns: {', '.join(self.target_columns)}"
            )

        # Check for duplicate assignments
        if len(set(self.target_assignments.values())) != len(self.target_assignments):
            duplicate_values = [val for val in self.target_assignments.values() if list(self.target_assignments.values()).count(val) > 1]
            raise ValueError(
                f"Duplicate assignment(s) found: {', '.join(set(duplicate_values))}. "
                "Each target column must have a unique assignment. "
                f"Current assignments: {dict(self.target_assignments)}"
            )

    def _validate_number_of_targets(self):
        """Validate number of targets."""
        if len(self.target_columns) > 2:
            raise ValueError("More than two targets are not allowed")
        if len(self.target_columns) == 2 and not self.target_assignments:
            raise ValueError("Target assignments are required when two targets are selected.This is only for ml_type = 'timetoevent'.")

    def _validate_column_dtypes(self):
        """Validate that all relevant columns have correct dtypes."""
        non_matching_columns = []

        if self.stratification_column:
            columns_to_check = self.feature_columns + self.target_columns + [self.stratification_column]
        else:
            columns_to_check = self.feature_columns + self.target_columns

        # Check each relevant column's dtype
        for column in columns_to_check:
            dtype = self.data[column].dtype
            if not (
                pd.api.types.is_integer_dtype(dtype)
                or pd.api.types.is_float_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                non_matching_columns.append(f"{column} ({dtype})")

        # Raise error if any columns have wrong dtypes
        if non_matching_columns:
            non_matching_str = ", ".join(non_matching_columns)
            raise ValueError(f"Columns with wrong dtypes: {non_matching_str}")

    def _validate_column_names_characters(self):
        """Search for disallowed characters in column names."""
        # Define the pattern for disallowed characters
        # This requirement comes from miceforest/lightGBM.
        disallowed_chars_pattern = re.compile(r'[",:$$ \ $${}]')
        # Find columns with disallowed characters
        problematic_columns = [col for col in self.relevant_columns if disallowed_chars_pattern.search(col)]
        # Raise an error if any problematic columns are found
        if problematic_columns:
            raise ValueError(
                f"The following columns contain disallowed characters:"
                f" {', '.join(problematic_columns)}.\n"
                f'Disallowed characters are: ", : , [ , ] , {{ , }}.'
            )

    def _validate_column_names(self):
        """Validate not allowed column names."""
        columns_not_allowed = ["group_features", "group_sample_and_features"]

        problematic_columns = [col for col in columns_not_allowed if col in self.data.columns]

        if problematic_columns:
            raise ValueError(f"The following columns names are not allowed  {', '.join(problematic_columns)}.\n")

    def _validate_row_id_unique(self):
        if self.row_id and not self.data[self.row_id].is_unique:
            raise ValueError("Duplicate row IDs found. Each row ID must be unique.")
