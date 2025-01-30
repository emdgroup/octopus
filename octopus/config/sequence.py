"""Config sequence."""

from typing import Any, List

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


def validate_sequence_items(
    _instance: Any, attribute: Any, value: List[BaseSequenceItem]
) -> None:
    """Validate the `sequence_items` attribute.

    Ensures that the value is a non-empty list where each item is an
    instance of `BaseSequenceItem`, and that the sequence meets specified
    conditions.

    Conditions:
    - The first sequence item must have `item_id=1`.
    - All items with `input_item_id=0` must be at the start of the list,
      before any other items with `input_item_id > 0`.
    - All elements in the list must be in increasing order of `item_id`.
    - For elements with `input_item_id > 0`, their `input_item_id` must
      refer to an `item_id` that comes before them in the list.
    - All `item_id`s should form a complete integer sequence with no
      missing values between the minimum and maximum `item_id`.

    Args:
        _instance: The instance that is being validated (unused).
        attribute: The attribute that is being validated.
        value: The value of the attribute to validate.

    Raises:
        TypeError: If any item in the list is not an instance of
            `BaseSequenceItem`.
        ValueError: If the list is empty or does not meet the specified
            conditions.
    """
    # Condition 1: Non-Empty List
    if not value:
        raise ValueError(
            f"'{attribute.name}' must contain at least one instance of "
            f"'BaseSequenceItem'."
        )

    # Condition 2: All Items are Instances of BaseSequenceItem
    for item in value:
        if not isinstance(item, BaseSequenceItem):
            raise TypeError(
                f"Each item in '{attribute.name}' must be an instance of "
                f"'BaseSequenceItem', but got '{type(item).__name__}'."
            )

    # Condition 2.5: First Item Must Have item_id=1
    if value[0].item_id != 1:
        raise ValueError(
            f"The first sequence item must have 'item_id=1', but got "
            f"'item_id={value[0].item_id}'."
        )

    # Build mapping of item_id to index and collect item_ids
    item_id_to_index = {}
    item_ids = []
    previous_item_id = None
    for idx, item in enumerate(value):
        # Ensure that item_ids are in increasing order
        if previous_item_id is not None and item.item_id <= previous_item_id:
            raise ValueError(
                f"Item at position {idx + 1} has 'item_id' {item.item_id}, which "
                f"is not greater than the previous 'item_id' {previous_item_id}. "
                "All 'item_id's must be in increasing order in the list."
            )
        previous_item_id = item.item_id

        if item.item_id in item_id_to_index:
            raise ValueError(
                f"Duplicate 'item_id' {item.item_id} found in the sequence."
            )
        item_id_to_index[item.item_id] = idx
        item_ids.append(item.item_id)

    # Condition 3: All item_ids form a complete integer sequence with no missing
    # values between min and max
    min_item_id = min(item_ids)
    max_item_id = max(item_ids)
    expected_item_ids = set(range(min_item_id, max_item_id + 1))
    actual_item_ids = set(item_ids)
    if expected_item_ids != actual_item_ids:
        missing_ids = expected_item_ids - actual_item_ids
        extra_ids = actual_item_ids - expected_item_ids
        message = (
            "All 'item_id's must form a complete integer sequence with no missing "
            "values between the minimum and maximum 'item_id'."
        )
        if missing_ids:
            message += f" Missing item_ids: {sorted(missing_ids)}."
        if extra_ids:
            message += f" Unexpected item_ids: {sorted(extra_ids)}."
        raise ValueError(message)

    # Condition 4: All items with input_item_id=0 must be at the start of the list
    reached_non_zero_input_item_id = False
    for idx, item in enumerate(value):
        if item.input_item_id == 0:
            if reached_non_zero_input_item_id:
                raise ValueError(
                    f"Item at position {idx + 1} has 'input_item_id=0' but appears "
                    "after items with 'input_item_id>0'. All items with "
                    "'input_item_id=0' must be at the start of the list."
                )
        else:
            reached_non_zero_input_item_id = True

    # Condition 6: For elements with input_item_id > 0, input_item_id must refer
    # to an item that comes before them
    for idx, item in enumerate(value):
        input_item_id = item.input_item_id
        if input_item_id > 0:
            if input_item_id not in item_id_to_index:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'input_item_id={input_item_id}', which does not correspond to "
                    "any 'item_id' in the sequence."
                )
            input_item_idx = item_id_to_index[input_item_id]
            if input_item_idx >= idx:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'input_item_id={input_item_id}', which refers to an item that "
                    "comes after it in the sequence. 'input_item_id' must refer to "
                    "a preceding 'item_id'."
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
    """A list of sequence items that defines the processing sequence.
    Each item in the list is an instance of `BaseSequenceItem` or its subclasses."""
