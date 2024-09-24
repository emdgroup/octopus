"""rfe module."""

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Rfe(BaseSequenceItem):
    """RFE Config."""

    module: str = field(default="rfe")
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""

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

    # scoring: str, callable or None, default=None
    # A string (see model evaluation documentation) or a scorer callable object /
    # function with signature scorer(estimator, X, y).

    # importance_getter: str or callable, default=’auto’
    # If ‘auto’, uses the feature importance either through a coef_ or
    # feature_importances_ attributes of estimator.
