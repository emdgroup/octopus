"""Config sequence."""

from typing import Any, List

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


def validate_sequence_items(_instance: Any, attribute: Any, value: List[BaseSequenceItem]) -> None:
    """Validate the `sequence_items` attribute.

    Ensures that the value is a non-empty list where each item is an
    instance of `BaseSequenceItem`, and that the sequence meets specified
    conditions.

    Conditions:
    - The first sequence item must have `sequence_id=1`.
    - All items with `input_sequence_id=0` must be at the start of the list,
      before any other items with `input_sequence_id > 0`.
    - All elements in the list must be in increasing order of `sequence_id`.
    - For elements with `input_sequence_id > 0`, their `input_sequence_id` must
      refer to an `sequence_id` that comes before them in the list.
    - All `sequence_id`s should form a complete integer sequence with no
      missing values between the minimum and maximum `sequence_id`.

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
        raise ValueError(f"'{attribute.name}' must contain at least one instance of 'BaseSequenceItem'.")

    # Condition 2: All Items are Instances of BaseSequenceItem
    for item in value:
        if not isinstance(item, BaseSequenceItem):
            raise TypeError(f"Each item in '{attribute.name}' must be an instance of 'BaseSequenceItem', but got '{type(item).__name__}'.")

    # Condition 2.5: First Item Must Have sequence_id=0
    if value[0].sequence_id != 0:
        raise ValueError(f"The first sequence item must have 'sequence_id=0', but got 'sequence_id={value[0].sequence_id}'.")

    # Build mapping of sequence_id to index and collect item_ids
    item_id_to_index = {}
    item_ids = []
    previous_item_id = None
    for idx, item in enumerate(value):
        # Ensure that item_ids are in increasing order
        if previous_item_id is not None and item.sequence_id <= previous_item_id:
            raise ValueError(
                f"Item at position {idx + 1} has 'sequence_id' {item.sequence_id}, "
                "which is not greater than the previous "
                f"'sequence_id' {previous_item_id}. "
                "All 'sequence_id's must be in increasing order in the list."
            )
        previous_item_id = item.sequence_id

        if item.sequence_id in item_id_to_index:
            raise ValueError(f"Duplicate 'sequence_id' {item.sequence_id} found in the sequence.")
        item_id_to_index[item.sequence_id] = idx
        item_ids.append(item.sequence_id)

    # Condition 3: All item_ids form a complete integer sequence with no missing
    # values between min and max
    min_item_id = min(item_ids)
    max_item_id = max(item_ids)
    expected_item_ids = set(range(min_item_id, max_item_id + 1))
    actual_item_ids = set(item_ids)
    if expected_item_ids != actual_item_ids:
        missing_ids = expected_item_ids - actual_item_ids
        extra_ids = actual_item_ids - expected_item_ids
        message = "All 'sequence_id's must form a complete integer sequence with no missing values between the minimum and maximum 'sequence_id'."
        if missing_ids:
            message += f" Missing item_ids: {sorted(missing_ids)}."
        if extra_ids:
            message += f" Unexpected item_ids: {sorted(extra_ids)}."
        raise ValueError(message)

    # Condition 4: All items with input_sequence_id=-1 must be at the start of the list
    reached_non_zero_input_sequence_id = False
    for idx, item in enumerate(value):
        if item.input_sequence_id == -1:
            if reached_non_zero_input_sequence_id:
                raise ValueError(
                    f"Item at position {idx + 1} has 'input_sequence_id=-1' but"
                    " appears after items with 'input_sequence_id>=0'. All items with "
                    "'input_sequence_id=-1' must be at the start of the list."
                )
        else:
            reached_non_zero_input_sequence_id = True

    # Condition 6: For elements with input_sequence_id >= 0, input_sequence_id must
    # refer to an item that comes before them
    for idx, item in enumerate(value):
        input_sequence_id = item.input_sequence_id
        if input_sequence_id >= 0:
            if input_sequence_id not in item_id_to_index:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'input_sequence_id={input_sequence_id}', which does not"
                    "correspond to any 'sequence_id' in the sequence."
                )
            input_sequence_idx = item_id_to_index[input_sequence_id]
            if input_sequence_idx >= idx:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'input_sequence_id={input_sequence_id}', which refers to an item"
                    "that comes after it in the sequence. 'input_sequence_id' must"
                    "refer to a preceding 'sequence_id'."
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
