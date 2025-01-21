"""Config for Base Sequence item."""

from attrs import define, field, validators


@define
class BaseSequenceItem:
    """Base class for all sequence items.

    Contains all common parameters for all sequence items.
    """

    description: str = field(validator=[validators.instance_of(str)])
    """Description for the sequence."""

    item_id: int = field(validator=[validators.instance_of(int), validators.ge(1)])
    """Sequence item ID, greater or equal than 1."""

    input_item_id: int = field(
        validator=[validators.instance_of(int), validators.ge(0)]
    )
    """Specify ID of input item. Input ID of start item: 0."""

    load_sequence_item: bool = field(
        default=False, validator=[validators.instance_of(bool)]
    )
    """Whether to load the sequence item."""
