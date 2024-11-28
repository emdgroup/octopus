"""Octo Data Class."""

import gzip
import logging
import pickle
import re
from typing import Optional

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators

from octopus.logger import configure_logging

from .health_checks import DataHealthChecker
from .report import DataHealthReport

configure_logging()
# tobedone:
# - identify categorical columns -- require pandas categorical
# - dtype check


@define
class OctoData:
    """Octopus data class."""

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing the dataset."""

    feature_columns: list = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    target_columns: list = field(validator=[validators.instance_of(list)])
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    datasplit_type: str = field(
        validator=[
            validators.in_(
                [None, "sample", "group_features", "group_sample_and_features"]
            )
        ]
    )
    """Type of datasplit. Allowed are `sample`, `group_features`
    and `group_sample_and_features`."""

    sample_id: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    row_id: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    target_assignments: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Mapping of target assignments."""

    stratification_column: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """List of columns used for stratification."""

    report: DataHealthReport = field(
        default=None,
        validator=validators.optional(validators.instance_of(DataHealthReport)),
    )
    """Enable data quality check."""

    @property
    def relevant_columns(self) -> list:
        """Relevant columns."""
        # Combine all necessary columns
        relevant_columns = set(
            self.feature_columns + self.target_columns + [self.sample_id]
        )

        # Add optional columns if they exist
        optional_columns = [
            self.row_id,
            self.stratification_column,
            "group_features",
            "group_sample_and_features",
        ]
        relevant_columns.update(filter(None, optional_columns))
        relevant_columns = [col for col in relevant_columns if col in self.data.columns]

        return list(set(relevant_columns))

    def __attrs_post_init__(self):
        logging.info("Automated data preparation:")

        # validate input
        self._check_column_names()
        self._check_columns_exist()
        self._check_for_duplicates()
        self._check_stratification_column()
        self._check_target_assignments()
        self._check_number_of_targets()
        self._check_column_dtypes()

        # prepare dataframe
        self._sort_features()
        self._set_target_assignments()
        self._remove_singlevalue_features()
        self._transform_bool_to_int()
        self._create_row_id()

        self._add_group_features()

        # data health check
        self.report = DataHealthChecker(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            row_id=self.row_id,
            sample_id=self.sample_id,
            stratification_column=self.stratification_column,
        ).generate_report()
        print(self.report)

        # encode categorical columns
        self._categorical_encoding()

    def _check_column_names(self):
        """Search for disallowed characters in column names."""
        # Define the pattern for disallowed characters
        # This requirement comes from miceforest/lightGBM.
        disallowed_chars_pattern = re.compile(r'[",:$$ \ $${}]')
        # Find columns with disallowed characters
        problematic_columns = [
            col for col in self.relevant_columns if disallowed_chars_pattern.search(col)
        ]
        # Raise an error if any problematic columns are found
        if problematic_columns:
            raise ValueError(
                f"The following columns contain disallowed characters:"
                f" {', '.join(problematic_columns)}.\n"
                f'Disallowed characters are: ", : , [ , ] , {{ , }}.'
            )

    def _categorical_encoding(self):
        """Process categorical columns."""
        # Identify categorical columns in self.relevant_columns
        categorical_columns = [
            col
            for col in self.relevant_columns
            if self.data[col].dtype.name == "category"
        ]
        # Split into ordinal and non-ordinal (nominal) categorical columns
        ordinal_columns = [
            col for col in categorical_columns if self.data[col].cat.ordered
        ]
        nominal_columns = [
            col for col in categorical_columns if not self.data[col].cat.ordered
        ]
        # Process non-ordinal categorical columns
        if nominal_columns:
            # (1) Count unique values and check for columns with more than 15 categories
            columns_with_many_categories = [
                col for col in nominal_columns if self.data[col].nunique() > 15
            ]
            if columns_with_many_categories:
                raise ValueError(
                    f"The following nominal categorical columns have more"
                    f" than 15 unique categories: "
                    f"{', '.join(columns_with_many_categories)}"
                )

            # (2) Perform dummy encoding
            dummies = pd.get_dummies(
                self.data[nominal_columns],
                prefix=nominal_columns,
                drop_first=True,  # remove first to avoid redundant information
            )

            # Drop original nominal columns from relevant_columns
            # we keep the nominal columns in the data
            self.feature_columns = [
                col for col in self.relevant_columns if col not in nominal_columns
            ]

            # Add dummy columns to data
            self.data = pd.concat([self.data, dummies], axis=1)

            # Update relevant_columns with new dummy column names
            self.feature_columns.extend(dummies.columns.tolist())

        # Process ordinal categorical columns
        if ordinal_columns:
            # Collect columns where categories are not integers
            problematic_columns = []
            for col in ordinal_columns:
                # Get the categories and check if they are all integers
                categories = self.data[col].cat.categories
                if not all(isinstance(cat, (int, np.integer)) for cat in categories):
                    problematic_columns.append(col)

            # Raise ValueError if there are problematic columns
            if problematic_columns:
                raise ValueError(
                    f"The following ordinal categorical columns have"
                    f" non-integer categories: "
                    f"{', '.join(problematic_columns)}"
                )

    def _check_column_dtypes(self):
        """Validate that all relevant columns have correct dtypes."""
        acceptable_dtypes = ["int64", "float64", "bool", "category"]
        non_matching_columns = []

        # Check each relevant column's dtype
        for column in self.feature_columns + self.target_columns:
            if self.data[column].dtype not in acceptable_dtypes:
                non_matching_columns.append(column)

        # Raise error if any columns are missing
        if non_matching_columns:
            non_matching_str = ", ".join(non_matching_columns)
            raise ValueError(f"Columns with wrong dtypes: {non_matching_str}")

    def _check_columns_exist(self):
        """Validate that all columns exists in the dataframe."""
        # Identify missing columns
        missing_columns = [
            col for col in self.relevant_columns if col not in self.data.columns
        ]

        # Raise error if any columns are missing
        if missing_columns:
            missing_str = ", ".join(missing_columns)
            raise ValueError(f"Columns not found in the DataFrame: {missing_str}")

    def _check_for_duplicates(self):
        # Combine all columns to check for duplicates
        columns_to_check = self.feature_columns + self.target_columns + [self.sample_id]

        if self.row_id:
            columns_to_check.append(self.row_id)

        # Check for duplicates
        duplicates = set(
            [col for col in columns_to_check if columns_to_check.count(col) > 1]
        )

        if duplicates:
            duplicates_str = ", ".join(duplicates)
            raise ValueError(f"Duplicate columns found: {duplicates_str}")

    def _check_stratification_column(self):
        """Check if stratification_column is the same as row_id or sample_id."""
        if self.stratification_column:
            if self.stratification_column in [self.sample_id, self.row_id]:
                raise ValueError(
                    "Stratification column cannot be the same as sample_id or row_id"
                )

    def _check_target_assignments(self):
        """Check if target_assignments values are unique and within target_columns."""
        assignment_values = list(self.target_assignments.values())

        # Check if all values are in target_columns
        invalid_values = [
            val for val in assignment_values if val not in self.target_columns
        ]
        if invalid_values:
            invalid_str = ", ".join(invalid_values)
            raise ValueError(f"Invalid target assignments found: {invalid_str}")

        # Check for duplicates in assignment values
        if len(assignment_values) != len(set(assignment_values)):
            raise ValueError("Duplicate values found in target assignments")

    def _check_number_of_targets(self):
        """Check number of targets."""
        if len(self.target_columns) > 2:
            raise ValueError("More than two targets are not allowed")
        if len(self.target_columns) == 2 and not self.target_assignments:
            raise ValueError(
                "Target assignments are required when two targets are selected."
                "This is only for ml_type = 'timetoevent'."
            )

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
        self.data = (
            self.data
            # Create group with the same features
            .assign(
                group_features=lambda df_: df_.groupby(self.feature_columns).ngroup()
            )
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
        logging.info("Added `group_feaures` and `group_sample_features`")

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

            logging.info("Added `row_id`")

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
        elif len(self.target_columns) > 1:
            if len(self.target_columns) != len(self.target_assignments):
                raise ValueError("Please provide correct target assignments")
        else:
            raise ValueError(
                f"Target assignments need to be provided: {self.target_assignments}"
            )

    def _remove_singlevalue_features(self):
        """Remove features that contain only a single unique value."""
        num_original_features = len(self.feature_columns)

        self.feature_columns = [
            feature
            for feature in self.feature_columns
            if self.data[feature].nunique() > 1
        ]
        num_new_features = len(self.feature_columns)

        if num_original_features > num_new_features:
            logging.info(
                "Features removed due to single unique values: %s",
                {num_original_features - num_new_features},
            )

    def _sort_features(self):
        """Sort feature columns deterministically by length and lexicographically.

        This ensures that the results are always the same, preventing minor differences.
        """
        self.feature_columns = sorted(
            map(str, self.feature_columns), key=lambda x: (len(x), x)
        )
        logging.info("Sorted features.")

    def _transform_bool_to_int(self):
        # Convert all boolean columns to integer
        bool_cols = self.data.select_dtypes(include="bool").columns
        self.data[bool_cols] = self.data[bool_cols].astype(int)
        logging.info("Transformed bool columns to int columns.")

    def save(self, path):
        """Save data to a human readable form, for long term storage."""
        self.data.to_csv(path.joinpath("data.csv"))

        # Needed: better way of serializing attrs.attributes
        # I failed with asdict() and removal of data
        # column_info=dict()
        # column_info['feature_columns']=self.feature_columns
        # column_info['target_columns']=self.target_columns
        # column_info['sample_id']=self.sample_id
        # column_info['row_id']=self.row_id
        # column_info['stratification_columns']=self.stratification_columns
        #
        # with open(path.joinpath('column_info.json'), "w", encoding="utf-8") as file:
        #    json.dump(column_info, file)

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "OctoData":
        """Load object to a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.

        Returns:
            OctoData: The loaded instance of OctoData.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
