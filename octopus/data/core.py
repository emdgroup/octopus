"""Octo Data Class."""

import gzip
import json
import pickle
from typing import Dict, List, Optional

import pandas as pd
from attrs import Factory, define, field, fields, validators

from octopus.logger import LogGroup, get_logger

from .encoder import DataEncoder
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

    feature_columns: List[str] = field(validator=[validators.instance_of(list)])
    """List of all feature columns in the dataset."""

    target_columns: List[str] = field(validator=[validators.instance_of(list)])
    """List of target columns in the dataset. For regression and classification,
    only one target is allowed. For time-to-event, two targets need to be provided.
    """

    datasplit_type: str = field(validator=[validators.in_([None, "sample", "group_features", "group_sample_and_features"])])
    """Type of datasplit. Allowed are `sample`, `group_features`
    and `group_sample_and_features`."""

    sample_id: str = field(validator=validators.instance_of(str))
    """Identifier for sample instances."""

    row_id: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )
    """Unique row identifier."""

    target_assignments: Dict[str, str] = field(default=Factory(dict), validator=[validators.instance_of(dict)])
    """Mapping of target assignments."""

    stratification_column: Optional[str] = field(
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
        # Combine all necessary columns
        relevant_columns = list(set(self.feature_columns + self.target_columns + [self.sample_id]))

        # Add optional columns if they exist
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
            self._encode_features()
            logger.info("OctoData initialization completed")

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

    def _encode_features(self):
        """Encode categorical features."""
        encoder = DataEncoder(
            data=self.data,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            stratification_column=self.stratification_column,
            target_assignments=self.target_assignments,
        )
        (
            self.data,
            self.feature_columns,
            self.stratification_column,
            self.target_assignments,
            self.target_columns,
        ) = encoder.encode()

    def save_attributes_to_parquet(self, file_path: str) -> None:
        """Save attributes to parquet."""
        # Create lists to store parameters and values
        parameters = []
        values = []

        # Get all attributes of the class
        for attr in fields(OctoData):
            attr_name = attr.name
            # Skip 'data' and 'report' attributes
            if attr_name not in ["data", "report"]:
                value = getattr(self, attr_name)

                if isinstance(value, list):
                    # Expand lists into multiple entries
                    for item in value:
                        parameters.append(attr_name)
                        values.append(item)
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    parameters.append(attr_name)
                    values.append(json.dumps(value))
                else:
                    parameters.append(attr_name)
                    values.append(value)

        # Create a pandas DataFrame
        df = pd.DataFrame({"Parameter": parameters, "Value": values})

        # Write the DataFrame to a Parquet file
        df.to_parquet(file_path, index=False)

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
