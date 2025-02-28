"""Opto Config module."""

import gzip
import pickle

from attrs import define, field, validators

from octopus.config.manager import ConfigManager
from octopus.config.sequence import ConfigSequence
from octopus.config.study import ConfigStudy


def validate_config_study(instance, attribute, value):
    """Validate if start_with_empty_study is consistent with sequence items."""
    if value.start_with_empty_study:
        for item in instance.sequence.sequence_items:
            if item.load_sequence_item:
                raise ValueError(
                    "Loading sequence items requires start_with_empty_study=False"
                )


@define
class OctoConfig:
    """Main configuration class that holds all other configurations."""

    study: ConfigStudy = field(
        factory=ConfigStudy,
        validator=[validators.instance_of(ConfigStudy), validate_config_study],
    )
    """Configuration for study parameters."""

    manager: ConfigManager = field(
        factory=ConfigManager, validator=[validators.instance_of(ConfigManager)]
    )
    """Configuration for manager parameters."""

    sequence: ConfigSequence = field(
        factory=ConfigSequence, validator=[validators.instance_of(ConfigSequence)]
    )
    """Configuration for sequence parameters."""

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "OctoConfig":
        """Load object from a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.


        Returns:
            OctoConfig: The loaded instance of OctoConfig.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
