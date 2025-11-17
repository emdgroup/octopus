"""Opto Config module."""

import gzip
import json
import pickle
from pathlib import Path

from attrs import define, field, validators

from octopus.config.manager import ConfigManager
from octopus.config.study import ConfigStudy
from octopus.config.workflow import ConfigWorkflow


def validate_config_study(instance, attribute, value):
    """Validate if start_with_empty_study is consistent with workflow tasks."""
    if value.start_with_empty_study:
        for item in instance.workflow.tasks:
            if item.load_task:
                raise ValueError("Loading workflow tasks requires start_with_empty_study=False")


@define
class OctoConfig:
    """Main configuration class that holds all other configurations."""

    study: ConfigStudy = field(
        validator=[validators.instance_of(ConfigStudy), validate_config_study],
    )
    """Configuration for study parameters."""

    manager: ConfigManager = field(factory=ConfigManager, validator=[validators.instance_of(ConfigManager)])
    """Configuration for manager parameters."""

    workflow: ConfigWorkflow = field(factory=ConfigWorkflow, validator=[validators.instance_of(ConfigWorkflow)])
    """Configuration for workflow parameters."""

    def to_pickle(self, file_path: str | Path):
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str | Path) -> "OctoConfig":
        """Load object from a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.


        Returns:
            OctoConfig: The loaded instance of OctoConfig.

        Raises:
            TypeError: If the file does not contain an OctoConfig instance.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            data = pickle.load(file)

        if not isinstance(data, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        return data

    @classmethod
    def from_json(cls, config_dir: Path) -> "OctoConfig":
        """Load config from JSON files in a directory.

        Args:
            config_dir: Path to the directory containing the JSON config files.

        Returns:
            OctoConfig: The loaded instance of OctoConfig.
        """
        # Load study config
        with open(config_dir / "config_study.json") as f:
            study_dict = json.load(f)
        study = ConfigStudy(**study_dict)

        # Load manager config
        with open(config_dir / "config_manager.json") as f:
            manager_dict = json.load(f)
        manager = ConfigManager(**manager_dict)

        # Load workflow config
        with open(config_dir / "config_workflow.json") as f:
            workflow_dict = json.load(f)
        workflow = ConfigWorkflow(**workflow_dict)

        return cls(study=study, manager=manager, workflow=workflow)
