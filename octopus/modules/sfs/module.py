"""SFS Module (sequential feature selection)."""

from attrs import define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Sfs(BaseSequenceItem):
    """SFS Config."""

    module: str = field(default="sfs")
    """Module name."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

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
        default="floating_backward",
    )
    """Sfs type used."""

    # correlation_type: str = field(
    #    validator=[validators.in_(["pearson", "rdc"])], default="pearson"
    # )
    # """Selection of correlation type."""
    #
    # feature_importance_type: str = field(
    #    validator=[validators.in_(["mean", "count"])], default="mean"
    # )
    # """Selection of feature importance type."""
    #
    # feature_importance_method: str = field(
    #    validator=[validators.in_(["permutation", "shap", "internal"])],
    #    default="permutation",
    # )
    # """Selection of feature importance method."""
