"""Config sequence."""

from typing import Any, List

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


def validate_sequence_items(
    instance: Any, attribute: Any, value: List[BaseSequenceItem]
) -> None:
    """Validate the `sequence_items` attribute.

    Ensures that the value is a non-empty list where each item is an instance of
    `BaseSequenceItem`.

    Args:
        instance: The instance that is being validated.
        attribute: The attribute that is being validated.
        value: The value of the attribute to validate.

    Raises:
        TypeError: If the value is not a list or if any item in the list is not
            an instance of `BaseSequenceItem`.
        ValueError: If the list is empty.
    """
    if not value:
        raise ValueError(f"{attribute.name} must contain at least one instance")
    for item in value:
        if not isinstance(item, BaseSequenceItem):
            raise TypeError(
                f"Each item in {attribute.name} must be an instance of BaseSequenceItem"
            )


@define
class ConfigSequence:
    """Configuration for sequence parameters.

    Attributes:
        sequence_items (List[BaseSequenceItem]):
    """

    sequence_items: List[BaseSequenceItem] = field(
        default=Factory(list),
        validator=[validators.instance_of(list), validate_sequence_items],
    )
    """A list of sequence items that define the sequence of operations.
    Each item in the list is an instance of `BaseSequenceItem` or its subclasses."""
