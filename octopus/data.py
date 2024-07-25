"""Octopus data classes."""

import gzip
import pickle
from typing import List

import numpy as np
import pandas as pd
from attrs import define, field, validators

# tobedone:
# - check that column definitions fit with pandas dataframe
# - in some scenarios we may want to keep NaNs (Martin, neural networks)
# - identify categorical columns -- require pandas categorical


@define
class OctoData:
    """Octopus data class."""

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    feature_columns: list = field(validator=[validators.instance_of(list)])
    target_columns: list = field(validator=[validators.instance_of(list)])
    sample_id: str = field(validator=[validators.instance_of(str)])
    datasplit_type: str = field(
        validator=[
            validators.in_(["sample", "group_features", "group_sample_and_features"])
        ],
    )
    row_id = field(default=None)
    disable_checknan: bool = field(
        default=False, validator=[validators.instance_of(bool)]
    )
    target_asignments: dict = field(
        default={}, validator=[validators.instance_of(dict)]
    )
    stratification_column: list = field(
        default=[], validator=[validators.instance_of(list)]
    )

    @property
    def targets(self) -> List:
        """Targets columns."""
        return self.target_columns

    @property
    def features(self) -> List:
        """List of features."""
        return self.feature_columns

    def __attrs_post_init__(self):
        self.modify_dataframe()  # index reset done here
        # create new column for row_id
        if self.row_id is None:
            self.data["row_id"] = self.data.index
            self.row_id = "row_id"
        # set default target assignment if single feature + empty assignment
        if (len(self.target_columns) == 1) & (not self.target_asignments):
            self.target_asignments["default"] = self.targets[0]
        elif len(self.target_columns) > 1:
            if len(self.target_columns) != len(self.target_asignments):
                raise ValueError("Please provide correct target assignments")
        else:
            raise ValueError("Target assignments need to be provided")
        # remove features with only a single value
        self.remove_singlevalue_features()

        # checks
        report = self.quality_check()
        if report:
            print("Quality Check Report:")
            for item in report:
                print(item)
            raise ValueError("Octo data quality check failed")

    def remove_singlevalue_features(self):
        """Remove feature that only contain a single value."""
        print("Original number of features:", len(self.feature_columns))
        data_features = self.data[self.feature_columns]
        singlevalue_features = data_features.columns[data_features.nunique() == 1]
        self.feature_columns = [
            feature
            for feature in self.feature_columns
            if feature not in singlevalue_features
        ]
        print(
            "Number of features after removal of single-valued features:",
            len(self.feature_columns),
        )

    def modify_dataframe(self):
        """Initialize the preparation of the dataframe.

        - add group col for entries with same features
        - add group col for entries with same features
            and/or same sample_id
        - reset index.

        """
        self.data = (
            self.data
            # create group with same features
            .assign(group_features=lambda df_: df_.groupby(self.features).ngroup())
            # create group with same features and/or same sample_id
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
            # reset index
            .reset_index(drop=True)
            # add fixed index col
            # .assign(RowID=lambda df_: df_.index)
        )

    def quality_check(self):
        """Quality check on octoData."""
        report = []

        if self.check_rowid_unique():
            report.append("Row_ID  raise ValueError()")
        if self.check_list_unique(self.feature_columns):
            report.append("Feature names are not unique")
        if self.check_list_unique(self.target_columns):
            report.append("Target names are not unique")
        if self.check_columns_exist():
            report.append("Defined columns missing in dataframe")
        if self.check_shared_columns():
            report.append("Shared columns in properties")
        if self.check_duplicates_features():
            print("Warning: Duplicates (rows) in features")
            if self.datasplit_type == "sample":
                report.append("Feature duplicates require different datasplit type")
        if self.check_duplicates_features_samples():
            print("Warning: Duplicates (rows) in features+sample-ID")
            if self.datasplit_type != "group_sample_and_features":
                report.append(
                    "Feature+Sample duplicates require different datasplit type"
                )
        if self.check_nans():
            if not self.disable_checknan:
                report.append("NaNs in dataframe")
        if self.check_infs():
            report.append("Infs in dataframe")
        if self.check_nonnumeric():
            report.append("Non-numeric relevant columns in dataframe")

        return report

    def check_rowid_unique(self) -> bool:
        """Check that rowid is unique."""
        row_column = self.data[self.row_id]
        return len(row_column) != len(set(row_column))

    def check_list_unique(self, columns) -> bool:
        """Check that list contains unique values."""
        return len(columns) != len(set(columns))

    def check_nans(self) -> bool:
        """Check if all relevant columns are free of NaNs."""
        relevant_columns = list(
            set(self.feature_columns)
            .union(set(self.target_columns))
            .union(set([self.sample_id]))
            .union(set([self.row_id]))
            .union(set(self.stratification_column))
        )
        return pd.isna(self.data[relevant_columns]).any().any()

    def check_infs(self) -> bool:
        """Check if all relevant columns are free of Infs."""
        relevant_columns = list(
            set(self.feature_columns)
            .union(set(self.target_columns))
            .union(set([self.sample_id]))
            .union(set([self.row_id]))
            .union(set(self.stratification_column))
        )
        return pd.isna(self.data[relevant_columns]).isin([np.inf, -np.inf]).any().any()

    def check_nonnumeric(self) -> bool:
        """Check if all relevant columns are numeric."""
        stratification_column = "".join(self.stratification_column)

        result = list()
        # stratification column
        if stratification_column:
            result.append(self.data[stratification_column].dtype.kind not in "iub")

        for column in self.feature_columns:
            result.append(self.data[column].dtype.kind not in "iuf")  # int/unit/float

        for column in self.target_columns:
            result.append(
                self.data[column].dtype.kind not in "iufb"
            )  # int/unit/float/bool
        return any(result)

    def check_shared_columns(self) -> bool:
        """Check for shared columns in properties."""
        # features/targets
        report = []
        intersection_features_targets = set(self.feature_columns).intersection(
            set(self.target_columns)
        )
        intersection_features_sample = set(self.feature_columns).intersection(
            set(self.sample_id)
        )
        intersection_targets_sample = set(self.target_columns).intersection(
            set(self.sample_id)
        )

        if intersection_features_targets:
            report.append(
                f"""Columns shared between features and targets:
                {intersection_features_targets}"""
            )

        if intersection_features_sample:
            report.append(
                f"""Columns shared between features and sample:
                {intersection_features_sample}"""
            )

        if intersection_targets_sample:
            report.append(
                f"""Columns shared between targets and sample:
                {intersection_targets_sample}"""
            )

        if not report:
            return False
        else:
            print(report)
            return True

    def check_columns_exist(self) -> bool:
        """Check that defined columns exist in dataframe."""
        defined_columns = (
            set(self.feature_columns)
            .union(set(self.target_columns))
            .union(set(self.sample_id))
            .union(set(self.stratification_column))
        )

        dataframe_columns = set(self.data.columns.tolist())

        if defined_columns.issubset(dataframe_columns):
            return True
        else:
            return False

    def check_duplicates_features(self) -> bool:
        """Warning only: check for duplicates (rows) in all features."""
        duplicate_rows = self.data[self.feature_columns].duplicated()
        return duplicate_rows.any()

    def check_duplicates_features_samples(self) -> bool:
        """Warning only: check for duplicates (rows) in all features+samples_id."""
        duplicate_rows = self.data[
            list(self.feature_columns) + [self.sample_id]
        ].duplicated()
        return duplicate_rows.any()

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
