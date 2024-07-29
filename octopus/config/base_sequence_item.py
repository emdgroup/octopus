"""Config for Base Sequence item."""

from attrs import define, field, validators


@define
class BaseSequenceItem:
    """Base class for all sequence items.

    Contains all common parameters for all sequence items.
    """

    description: str = field(validator=[validators.instance_of(str)])
    """Description for the sequence."""
