"""SFS Module (sequential feature selection)."""

from typing import ClassVar

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Sfs(BaseSequenceItem):
    """SFS Config."""

    module: ClassVar[str] = "sfs"
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

    # k_features: int or tuple or str (default: 1)
    # Number of features to select, where k_features < the full feature set.
    # New in 0.4.2: A tuple containing a min and max value can be provided,
    # and the SFS will consider return any feature combination between min and max
    # that scored highest in cross-validation.

    # fixed_features: tuple (default: None)
    # If not None, the feature indices provided as a tuple will be regarded as fixed by
    # the feature selector

    # feature_groups : list or None (default: None)
    # Optional argument for treating certain features as a group.
