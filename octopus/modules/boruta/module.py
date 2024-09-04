"""Boruta Module."""

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Boruta(BaseSequenceItem):
    """Boruta Config."""

    module: str = field(default="sfs")
    """Module name."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""
