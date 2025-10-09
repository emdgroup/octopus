"""OctoData Validator."""

from collections import Counter

import pandas as pd
from attrs import define


@define
class OctoDataValidator:
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

    stratification_column: str | None
    """List of columns used for stratification."""

    target_assignments: dict[str, str]
    """Mapping of target assignments."""

    relevant_columns: list[str]
    """Relevant columns of the dataset."""

    def validate(self):
        """Run all validation checks on the OctoData configuration.

        Performs comprehensive validation checks in a specific order to ensure
        data quality and structural integrity before processing. Validates basic
        structure, column names, relationships, data types, and data quality.

        Collects all validation errors and raises a single exception with all
        error messages if any validation fails.

        Raises:
            ValueError: If any validation check fails. Includes all validation
                errors in a single exception message.
        """
        validators = [
            self._validate_nonempty_dataframe,
            self._validate_reserved_column_conflicts,
            self._validate_columns_exist,
            self._validate_duplicated_columns,
            self._validate_feature_target_overlap,
            self._validate_stratification_column,
            self._validate_target_assignments,
            self._validate_number_of_targets,
            self._validate_column_dtypes,
        ]

        errors = []
        for validator in validators:
            try:
                validator()
            except ValueError as e:
                errors.append(f"- {str(e)}")

        if errors:
            error_message = "Multiple validation errors found:\n" + "\n".join(errors)
            raise ValueError(error_message)

    def _validate_columns_exist(self):
        """Validate that all relevant columns exist in the DataFrame.

        Checks that all columns specified in feature_columns, target_columns,
        sample_id, row_id, and stratification_column are present in the DataFrame.

        Raises:
            ValueError: If any relevant columns are missing from the DataFrame.
        """
        missing_columns = [col for col in self.relevant_columns if col not in self.data.columns]
        if missing_columns:
            missing_str = ", ".join(missing_columns)
            raise ValueError(f"Columns not found in the DataFrame: {missing_str}")

    def _validate_duplicated_columns(self):
        """Validate that no duplicate column names exist in the configuration.

        Validates that no column appears multiple times across feature_columns,
        target_columns, sample_id, and row_id. This prevents ambiguous column
        references.

        Raises:
            ValueError: If any column name appears more than once in the
                configuration.
        """
        columns_to_check = self.feature_columns + self.target_columns + [self.sample_id]

        if self.row_id:
            columns_to_check.append(self.row_id)

        duplicates = [col for col, count in Counter(columns_to_check).items() if count > 1]

        if duplicates:
            duplicates_str = ", ".join(duplicates)
            raise ValueError(f"Duplicate columns found: {duplicates_str}")

    def _validate_stratification_column(self):
        """Validate that stratification_column is not a reserved identifier.

        Ensures that the stratification column (if specified) is not the same as
        sample_id or row_id, which are reserved for data identification.

        Raises:
            ValueError: If stratification_column is the same as sample_id or row_id.
        """
        if self.stratification_column and self.stratification_column in [
            self.sample_id,
            self.row_id,
        ]:
            raise ValueError("Stratification column cannot be the same as sample_id or row_id")

    def _validate_target_assignments(self):
        """Validate target assignments for multi-target scenarios.

        For single target columns, ensures no assignments are provided (not needed).
        For multiple target columns, validates that:
        - All target columns have assignments
        - All assignments reference valid target columns
        - Each target column has a unique assignment

        Raises:
            ValueError: If target assignments are invalid, missing, or contain
                duplicates. Specific error messages indicate the exact issue.

        Returns:
            None: Returns early for single target columns after validation.
        """
        if len(self.target_columns) == 1:
            if self.target_assignments:
                raise ValueError(
                    "Target assignments provided for a single target column. Assignments are only needed for multiple target columns."
                )
            return
        if not self.target_assignments:
            raise ValueError(
                f"Multiple target columns detected ({len(self.target_columns)}), "
                "but no target assignments provided. "
                f"Please specify assignments for: {', '.join(self.target_columns)}"
            )

        missing_assignments = set(self.target_columns) - set(self.target_assignments.values())
        if missing_assignments:
            raise ValueError(
                "Missing assignments for target column(s): "
                f"{', '.join(missing_assignments)}. "
                "Please provide assignments for all target columns: "
                f"{', '.join(self.target_columns)}"
            )

        invalid_assignments = set(self.target_assignments.values()) - set(self.target_columns)
        if invalid_assignments:
            raise ValueError(
                "Invalid assignment key(s) detected: "
                f"{', '.join(invalid_assignments)}. Assignments must be made "
                f"only for existing target columns: {', '.join(self.target_columns)}"
            )

        if len(set(self.target_assignments.values())) != len(self.target_assignments):
            duplicate_values = [
                val for val in self.target_assignments.values() if list(self.target_assignments.values()).count(val) > 1
            ]
            raise ValueError(
                f"Duplicate assignment(s) found: {', '.join(set(duplicate_values))}. "
                "Each target column must have a unique assignment. "
                f"Current assignments: {dict(self.target_assignments)}"
            )

    def _validate_number_of_targets(self):
        """Validate the number of target columns.

        Ensures that:
        - No more than 2 target columns are specified
        - If 2 targets are specified, target_assignments must be provided
          (required for time-to-event modeling)

        Raises:
            ValueError: If more than 2 targets are specified, or if 2 targets
                are provided without target assignments.
        """
        if len(self.target_columns) > 2:
            raise ValueError("More than two targets are not allowed")
        if len(self.target_columns) == 2 and not self.target_assignments:
            raise ValueError(
                "Target assignments are required when two targets are selected. This is only for ml_type = 'timetoevent'."
            )

    def _validate_column_dtypes(self):
        """Validate that feature and target columns have supported data types.

        Checks that all feature columns, target columns, and stratification column
        (if present) have dtypes that are compatible with machine learning models.
        Supported dtypes are: integer, float, boolean, and categorical.

        Raises:
            ValueError: If any column has an unsupported dtype. Lists all columns
                with invalid dtypes along with their actual dtype.
        """
        non_matching_columns = []

        if self.stratification_column:
            columns_to_check = self.feature_columns + self.target_columns + [self.stratification_column]
        else:
            columns_to_check = self.feature_columns + self.target_columns

        for column in columns_to_check:
            dtype = self.data[column].dtype
            if not (
                pd.api.types.is_integer_dtype(dtype)
                or pd.api.types.is_float_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                non_matching_columns.append(f"{column} ({dtype})")

        if non_matching_columns:
            non_matching_str = ", ".join(non_matching_columns)
            raise ValueError(f"Columns with wrong dtypes: {non_matching_str}")

    def _validate_nonempty_dataframe(self):
        """Validate that the DataFrame is not empty.

        Ensures the DataFrame contains at least one row of data.

        Raises:
            ValueError: If the DataFrame has zero rows.
        """
        if len(self.data) == 0:
            raise ValueError("DataFrame is empty. Cannot proceed with empty dataset.")

    def _validate_feature_target_overlap(self):
        """Validate that no column is both a feature and a target.

        Ensures that feature_columns and target_columns do not share any column
        names, preventing logical conflicts in model training.

        Raises:
            ValueError: If any columns appear in both feature_columns and
                target_columns.
        """
        overlap = set(self.feature_columns) & set(self.target_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both features and targets: {', '.join(sorted(overlap))}")

    def _validate_reserved_column_conflicts(self):
        """Validate that reserved column names are not present in the DataFrame.

        Checks for conflicts with columns that will be created during data
        preparation: 'group_features', 'group_sample_and_features', and 'row_id'
        (if not provided by user).

        Raises:
            ValueError: If any reserved column names are found in the DataFrame.
        """
        reserved = ["group_features", "group_sample_and_features"]
        if not self.row_id:
            reserved.append("row_id")

        conflicts = [col for col in reserved if col in self.data.columns]
        if conflicts:
            raise ValueError(
                f"Reserved column names found in data: {', '.join(conflicts)}. "
                "These column names are used internally by Octopus."
            )
