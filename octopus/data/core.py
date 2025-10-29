"""Octo Data Class."""

import gzip
import json
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from attrs import Factory, asdict, define, field, fields, validators

from octopus.logger import LogGroup, get_logger

from .healthChecker import HealthCheckConfig, OctoDataHealthChecker
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

    datasplit_type: Literal["sample", "group_features", "group_sample_and_features"] | None = field(
        validator=validators.optional(validators.in_(["sample", "group_features", "group_sample_and_features"]))
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

    health_check_config: HealthCheckConfig = field(
        factory=HealthCheckConfig,
        validator=validators.instance_of(HealthCheckConfig),
    )
    """Configuration for health check thresholds."""

    report: pd.DataFrame | None = field(
        default=None, validator=validators.optional(validators.instance_of(pd.DataFrame))
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
            config=self.health_check_config,
        )

        self.report = checker.generate_report()

    def save_attributes_to_parquet(self, file_path: str | Path) -> None:
        """Save attributes to parquet.

        Note: health_check_config is saved separately using save_health_check_config().
        """
        parameters = []
        values = []

        for attr in fields(OctoData):
            attr_name = attr.name
            if attr_name not in ["data", "report", "health_check_config"]:
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

    def save_health_check_config(self, file_path: str) -> None:
        """Save health check configuration to JSON file.

        Args:
            file_path: The path to save the health check configuration JSON file.
        """
        config_dict = asdict(self.health_check_config)
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_health_check_config(cls, file_path: str) -> HealthCheckConfig:
        """Load health check configuration from JSON file.

        Args:
            file_path: The path to load the health check configuration JSON file from.

        Returns:
            HealthCheckConfig: The loaded health check configuration.
        """
        with open(file_path) as f:
            config_dict = json.load(f)
        return HealthCheckConfig(**config_dict)

    def to_pickle(self, file_path: str | Path):
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str | Path) -> "OctoData":
        """Load object to a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.

        Returns:
            OctoData: The loaded instance of OctoData.

        Raises:
            TypeError: If the file does not contain an OctoData instance.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            data = pickle.load(file)

        if not isinstance(data, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return data
