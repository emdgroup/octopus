"""Config for Base Sequence item."""

from attrs import define, field, validators


@define
class BaseSequenceItem:
    """Base class for all sequence items.

    Contains all common parameters for all sequence items.
    """

    description: str = field(validator=[validators.instance_of(str)])
    """Description for the sequence."""

    sequence_id: int = field(validator=[validators.instance_of(int), validators.ge(0)])
    """Sequence item ID, greater or equal than 0."""

    input_sequence_id: int = field(default=-1, validator=[validators.instance_of(int), validators.ge(-1)])
    """Specify ID of input item. Input ID of start item: -1."""

    load_sequence_item: bool = field(default=False, validator=[validators.instance_of(bool)])
    """Whether to load the sequence item."""
