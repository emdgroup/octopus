"""AutoGluon module."""

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class AutoGluon(BaseSequenceItem):
    """AG Config."""

    module: str = field(default="ag")
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by AG."""

    # cv: int = field(validator=[validators.instance_of(int)], default=5)
    # """Number of CV folds."""
