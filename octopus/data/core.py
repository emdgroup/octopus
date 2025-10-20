"""Octo Data Class."""

import gzip
import json
import pickle

import pandas as pd
from attrs import Factory, define, field, fields, validators

from octopus.logger import LogGroup, get_logger

from .healthChecker import OctoDataHealthChecker
from .preparator import OctoDataPreparator
from .validator import OctoDataValidator

logger = get_logger()
logger.set_log_group(LogGroup.DATA_PREPARATION)


@define
class OctoData:
    """Octopus data class."""

    data: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    """DataFrame containing the dataset."""

    feature_columns: list[str] = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    target_columns: list[str] = field(validator=[validators.instance_of(list)])
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    datasplit_type: str = field(
        validator=[validators.in_([None, "sample", "group_features", "group_sample_and_features"])]
    )
    """Type of datasplit. Allowed are `sample`, `group_features`
    and `group_sample_and_features`."""

    sample_id: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    row_id: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    target_assignments: dict[str, str] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Mapping of target assignments."""

    stratification_column: str | None = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """List of columns used for stratification."""

    report: pd.DataFrame = field(
        default=None,
        validator=validators.optional(validators.instance_of(pd.DataFrame)),
    )
    """Enable data quality check."""

    @property
    def relevant_columns(self) -> list:
        """Relevant columns."""
        relevant_columns = list(set(self.feature_columns + self.target_columns + [self.sample_id]))

        if self.row_id:
            relevant_columns.append(self.row_id)
        if self.stratification_column:
            relevant_columns.append(self.stratification_column)
        if "group_features" in self.data.columns:
            relevant_columns.append("group_features")
        if "group_sample_and_features" in self.data.columns:
            relevant_columns.append("group_sample_and_features")

        return list(set(relevant_columns))

    def __attrs_post_init__(self):
        logger.set_log_group(LogGroup.DATA_PREPARATION)
        logger.info("Initializing OctoData")

        try:
            self._validate_data()
            self._prepare_data()
            self._generate_health_report()

        except Exception as e:
            logger.error("Error during OctoData initialization: %s", e, exc_info=True)
            raise

    def _validate_data(self):
        """Validate the input data."""
        validator = OctoDataValidator(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            sample_id=self.sample_id,
            row_id=self.row_id,
            stratification_column=self.stratification_column,
            target_assignments=self.target_assignments,
            relevant_columns=self.relevant_columns,
        )
        validator.validate()

    def _prepare_data(self):
        """Prepare the data for analysis."""
        preparator = OctoDataPreparator(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            sample_id=self.sample_id,
            row_id=self.row_id,
            target_assignments=self.target_assignments,
        )
        self.data, self.feature_columns, self.row_id, self.target_assignments = preparator.prepare()

    def _generate_health_report(self):
        """Generate the health report."""
        checker = OctoDataHealthChecker(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            row_id=self.row_id,
            sample_id=self.sample_id,
            stratification_column=self.stratification_column,
        )

        self.report = checker.generate_report()

    def save_attributes_to_parquet(self, file_path: str) -> None:
        """Save attributes to parquet."""
        parameters = []
        values = []

        for attr in fields(OctoData):
            attr_name = attr.name
            if attr_name not in ["data", "report"]:
                value = getattr(self, attr_name)

                if isinstance(value, list):
                    for item in value:
                        parameters.append(attr_name)
                        values.append(item)
                elif isinstance(value, dict):
                    parameters.append(attr_name)
                    values.append(json.dumps(value))
                else:
                    parameters.append(attr_name)
                    values.append(value)

        df = pd.DataFrame({"Parameter": parameters, "Value": values})
        df.to_parquet(file_path, index=False)

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
