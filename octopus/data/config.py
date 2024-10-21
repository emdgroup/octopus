"""Octo Data Class."""

import gzip
import logging
import pickle
from typing import List

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

    row_id: str = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    target_assignments: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Mapping of target assignments."""

    stratification_column: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """List of columns used for stratification."""

    report: DataHealthReport = field(
        default=None,
        validator=validators.optional(validators.instance_of(DataHealthReport)),
    )
    """Enable data quality check."""

    # not needed anymore
    @property
    def targets(self) -> List:
        """Targets columns."""
        return self.target_columns

    # not needed anymore
    @property
    def features(self) -> List:
        """List of features."""
        return self.feature_columns

    def __attrs_post_init__(self):
        logging.info("Automated data preparation:")

        # validate input
        self._validate_unique_columns()

        # create new column for row_id
        self._create_row_id()

        # add group features
        self._add_group_features()

        # sort features
        self._sort_features()

        # set target assignment
        self._set_target_assignments()

        self.report = DataHealthChecker(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            row_id=self.row_id,
            sample_id=self.sample_id,
            stratification_column=self.stratification_column,
        ).generate_report(outlier_detection=False)

        # remove features with only a single value
        self._remove_singlevalue_features()

    def _validate_unique_columns(self):
        # Gather all columns that should be unique
        columns_all = (
            self.feature_columns
            + self.target_columns
            + [self.sample_id]
            + self.stratification_column
        )

        columns_check_duplicated = (
            self.feature_columns + self.target_columns + [self.sample_id]
        )

        if self.row_id:
            columns_all.append(self.row_id)
            columns_check_duplicated.append(self.row_id)

        if self.stratification_column:
            columns_all.extend(self.stratification_column)

        # Check for duplicated columns
        duplicates = set(
            [
                col
                for col in columns_check_duplicated
                if columns_check_duplicated.count(col) > 1
            ]
        )

        if duplicates:
            raise ValueError(
                f"Columns selected more than once: {', '.join(duplicates)}"
            )

        # Check if all columns exist in the DataFrame
        missing_columns = [col for col in columns_all if col not in self.data.columns]

        if missing_columns:
            raise ValueError(
                f"Columns not found in the DataFrame: {', '.join(missing_columns)}"
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
            .assign(group_features=lambda df_: df_.groupby(self.features).ngroup())
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
        if self.row_id == "":
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
            self.target_assignments["default"] = self.targets[0]
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
                "Features removed due to single unique values: %s"
                % (num_original_features - num_new_features)
            )

    def _sort_features(self):
        """Sort feature columns deterministically by length and lexicographically.

        This ensures that the results are always the same, preventing minor differences.
        """
        self.feature_columns = sorted(
            map(str, self.feature_columns), key=lambda x: (len(x), x)
        )
        logging.info("Sorted features.")

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
