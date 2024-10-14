"""Octopus data classes."""

import gzip
import logging
import pickle
from typing import List

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators

from octopus.logger import configure_logging

# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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

    sample_id: str = field(validator=[validators.instance_of(str)])
    """Identifier for sample instances."""

    datasplit_type: str = field(
        validator=[
            validators.in_(["sample", "group_features", "group_sample_and_features"])
        ]
    )
    """Type of datasplit. Allowed are `sample`, `group_features`
    and `group_sample_and_features`."""

    row_id: str = field(
        default=Factory(lambda: ""), validator=[validators.instance_of(str)]
    )
    """Unique row identifier."""

    disable_checknan: bool = field(
        default=Factory(lambda: False), validator=[validators.instance_of(bool)]
    )
    """Flag to disable the check for NaN values. Defaults to False."""

    target_assignments: dict = field(
        default=Factory(dict), validator=[validators.instance_of(dict)]
    )
    """Mapping of target assignments."""

    stratification_column: list = field(
        default=Factory(list), validator=[validators.instance_of(list)]
    )
    """List of columns used for stratification."""

    data_quality_check: bool = field(
        default=Factory(lambda: True), validator=[validators.instance_of(bool)]
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
        # sort features
        self._sort_features()

        # add group features
        self._add_group_features()

        # create new column for row_id
        self._create_row_id()

        # set target assignment
        self._set_target_assignments()

        # remove features with only a single value
        self._remove_singlevalue_features()

        # quality check
        if self.data_quality_check:
            self.quality_check()
        else:
            logging.warning("Quality check is skipped.")

    def _sort_features(self):
        """Sort list of feature columns to enforce deterministic behaviour."""
        # enfore str items
        feature_columns_str = [str(item) for item in self.feature_columns]
        # sort list items
        self.feature_columns = sorted(feature_columns_str, key=lambda x: (len(x), x))
        logging.info(" - Features sorted")

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
            ).reset_index(drop=True)
        )
        logging.info(" - Added `group_feaures` and `group_sample_features`")

    def _create_row_id(self):
        """Assign a unique row identifier to each entry in the DataFrame.

        If no `row_id` is provided, this method adds a `row_id` column using
        the DataFrame index.
        """
        if self.row_id == "":
            self.data["row_id"] = self.data.index
            self.row_id = "row_id"

            logging.info(" - Added `row_id`")

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
                " - Features removed due to single unique values: %s"
                % (num_original_features - num_new_features)
            )

    def quality_check(self):
        """Quality check on OctoData."""
        checker = QualityChecker(self)
        report, warning = checker.perform_checks()

        if warning:
            logging.warning("Quality Check Warnings:")
            for item in warning:
                logging.warning(f" - {item}")

        if report:
            logging.error("Quality Check Failures:")
            for item in report:
                logging.error(f" - {item}")
            raise ValueError("Octo data quality check failed")

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


@define
class QualityChecker:
    """Quality checker class for OctoData."""

    octo_data: OctoData = field(validator=[validators.instance_of(OctoData)])

    def perform_checks(self):
        """Perform all quality checks."""
        report = []
        warning = []

        # # Check unique row id
        if unique_row_id := self.unique_rowid_values():
            report.append(unique_row_id)

        # Check unique features
        if unique_features := self.unique_column(
            self.octo_data.feature_columns, "features"
        ):
            report.append(unique_features)

        # Check unique targets
        if unique_targets := self.unique_column(
            self.octo_data.target_columns, "features"
        ):
            report.append(unique_targets)

        # check if features or targets overlap
        if overlap_feat_targ := self.overlap_columns(
            self.octo_data.feature_columns,
            self.octo_data.target_columns,
            "features",
            "targets",
        ):
            report.append(overlap_feat_targ)

        # check if features or sample_id overlap
        if overlap_feat_sample := self.overlap_columns(
            self.octo_data.feature_columns,
            [self.octo_data.sample_id],
            "features",
            "sample_id",
        ):
            report.append(overlap_feat_sample)

        # # check if targets or sample_id overlap
        if overlap_targ_sample := self.overlap_columns(
            self.octo_data.target_columns,
            [self.octo_data.sample_id],
            "targets",
            "sample_id",
        ):
            report.append(overlap_targ_sample)

        # missing columns in dataframe
        # is is not working anyway, because octodata can not be created
        if missing_columns := self.missing_columns():
            report.append(missing_columns)

        # check for duplicates in features and sample
        if datasplit_required := self.duplicates_features_samples():
            if datasplit_required[0]:
                report.append(datasplit_required[0])
            if datasplit_required[1]:
                warning.append(datasplit_required[1])

        # check values for Infs and NaN
        if nan := self.not_allowed_values([np.nan], "NaN"):
            report.append(nan)

        if infs := self.not_allowed_values([np.inf, -np.inf], "Inf"):
            report.append(infs)

        # check dtypes
        if dtype_feature := self.check_nonnumeric(
            "Feature", self.octo_data.feature_columns, "iuf"
        ):
            report.append(dtype_feature)
        if dtype_target := self.check_nonnumeric(
            "Target", self.octo_data.target_columns, "iufb"
        ):
            report.append(dtype_target)
        if dtype_stratifictaion := self.check_nonnumeric(
            "Stratification ", self.octo_data.stratification_column, "iub"
        ):
            report.append(dtype_stratifictaion)

        return report, warning

    def _relevant_columns(self) -> set:
        """Get relevant columns for checks."""
        return (
            set(self.octo_data.feature_columns)
            .union(set(self.octo_data.target_columns))
            .union({self.octo_data.sample_id, self.octo_data.row_id})
            .union(set(self.octo_data.stratification_column))
        )

    def unique_rowid_values(self) -> str | None:
        """Check if values of row_id are unique."""
        if not self.octo_data.data[self.octo_data.row_id].is_unique:
            return "Row_ID is not unique"
        return None

    def unique_column(self, columns: List[str], column_type: str | None) -> None:
        """Add non-unique columns check to the report."""
        non_unique_columns = list({item for item in columns if columns.count(item) > 1})
        if non_unique_columns:
            return (
                f"The following {column_type} are not unique: "
                f"{', '.join(non_unique_columns)}"
            )
        return None

    def overlap_columns(
        self,
        list1: List[str],
        list2: List[str],
        name1: str,
        name2: str,
    ) -> None:
        """Add overlap check between feature and target columns to the report."""
        overlapping_columns = set(list1).intersection(list2)
        if overlapping_columns:
            return (
                f"Columns shared between {name1} and {name2}: "
                f"{', '.join(overlapping_columns)}"
            )
        return None

    def missing_columns(self) -> str | None:
        """Adding missing columns in dataframe to the report."""
        missing_columns = list(
            self._relevant_columns() - set(self.octo_data.data.columns)
        )
        if missing_columns:
            return f"Missing columns in dataset: {', '.join(missing_columns)}"
        return None

    def duplicates_features_samples(self):
        """Check for duplicates (rows) in all features."""
        duplicated_features = (
            self.octo_data.data[self.octo_data.feature_columns].duplicated().any()
        )
        duplicated_features_and_sample = (
            self.octo_data.data[
                list(self.octo_data.feature_columns) + [self.octo_data.sample_id]
            ]
            .duplicated()
            .any()
        )
        if duplicated_features and not duplicated_features_and_sample:
            if self.octo_data.datasplit_type == "sample":
                return (
                    "Duplicates in features require datasplit type "
                    "`group_features` or `group_sample_and_features`.",
                    "Duplicates (rows) in features",
                )
            return (None, "Duplicates (rows) in features")

        if duplicated_features_and_sample:
            if self.octo_data.datasplit_type != "group_sample_and_features":
                return (
                    "Duplicates in features and sample require datasplit "
                    "type `group_sample_and_features`.",
                    "Duplicates (rows) in features and sample",
                )
            return (
                None,
                "Duplicates (rows) in features and sample",
            )
        return None

    def not_allowed_values(self, values: List, name: str) -> str | None:
        """Check if all relevant columns are free of Infs."""
        columns_with_nan = [
            col
            for col in list(self._relevant_columns())
            if self.octo_data.data[col].isin(values).any()
        ]
        if columns_with_nan:
            return f"{name} in columns: {', '.join(columns_with_nan)}"

    def check_nonnumeric(
        self, name: str, columns: List[str], allowed_dtypes: str
    ) -> str | None:
        """Check if specified columns contain only allowed data types."""
        # Initialize list to store non-numeric column names
        non_numeric_columns = []

        # Check each column against the allowed data types
        for column in columns:
            if self.octo_data.data[column].dtype.kind not in allowed_dtypes:
                non_numeric_columns.append(column)

        if non_numeric_columns:
            return (
                f"{name} columns are not in types '{allowed_dtypes}': "
                f"{', '.join(non_numeric_columns)}"
            )
        return None
