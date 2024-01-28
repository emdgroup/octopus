"""Octo Experiment module."""
import pickle
from pathlib import Path

import pandas as pd
from attrs import define, field, validators


@define
class OctoExperiment:
    """Experiment."""

    id: str = field(validator=[validators.instance_of(str)])
    experiment_id: int = field(validator=[validators.instance_of(int)])
    sequence_item_id: int = field(validator=[validators.instance_of(int)])
    path_sequence_item: Path = field(validator=[validators.instance_of(Path)])
    config: dict = field(validator=[validators.instance_of(dict)])
    datasplit_column: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    feature_columns: list = field(validator=[validators.instance_of(list)])
    stratification_column: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    data_traindev: pd.DataFrame = field(
        validator=[validators.instance_of(pd.DataFrame)]
    )
    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])

    ml_module: str = field(
        init=False, default="", validator=[validators.instance_of(str)]
    )
    # number of cpus available for each experiment
    num_assigned_cpus: int = field(
        init=False, default=0, validator=[validators.instance_of(int)]
    )

    ml_config: dict = field(
        init=False, default={}, validator=[validators.instance_of(dict)]
    )
    selected_features: list = field(
        init=False, default=[], validator=[validators.instance_of(list)]
    )
    results: dict = field(
        init=False, default={}, validator=[validators.instance_of(dict)]
    )
    models: dict = field(
        init=False, default={}, validator=[validators.instance_of(dict)]
    )

    @property
    def path_study(self) -> Path:
        """Path study."""
        return Path(self.config["output_path"], self.config["study_name"])

    @property
    def ml_type(self) -> str:
        """Shortcut to ml_type."""
        return self.config["ml_type"]

    def to_pickle(self, filename):
        """Serialize experiment using pickle."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, filename):
        """Load experiment from pickle file."""
        with open(filename, "rb") as file:
            return pickle.load(file)
