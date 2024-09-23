"""SFS Module (sequential feature selection)."""

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Sfs(BaseSequenceItem):
    """SFS Config."""

    module: str = field(default="sfs")
    """Module name."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by SFS."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for RFE_CV."""

    sfs_type: str = field(
        validator=[
            validators.in_(
                ["forward", "backward", "floating_forward", "floating_backward"]
            )
        ],
        default="backward",
    )
    """Sfs type used."""
