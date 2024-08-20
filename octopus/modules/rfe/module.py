"""rfe module."""

from attrs import define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Rfe(BaseSequenceItem):
    """RFE Config."""

    module: str = field(default="rfe")
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by RFE."""

    step: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of features to remove at each iteration."""

    min_features_to_select: int = field(
        validator=[validators.instance_of(int)], default=1
    )
    """Minimum number of features to be selected."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for RFE_CV."""

    mode: str = field(validator=[validators.in_(["Mode1", "Mode2"])], default="Mode1")
    """Mode used by RFE."""
