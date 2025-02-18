"""Mrmr module."""

from typing import ClassVar

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Mrmr(BaseSequenceItem):
    """MRMR Config."""

    module: ClassVar[str] = "mrmr"
    """Module name."""

    n_features: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 30)
    )
    """Number of features selected by MRMR."""

    correlation_type: str = field(
        validator=[validators.in_(["pearson", "rdc", "spearmanr"])],
        default=Factory(lambda: "spearmanr"),
    )
    """Selection of correlation type."""

    relevance_type: str = field(
        validator=[validators.in_(["permutation", "f-statistics"])],
        default="permutation",
    )
    """Selection of relevance measure."""

    model_name: str = field(
        validator=[validators.in_(["best", "ensel", "autosk"])],
        default=Factory(lambda: "best"),
    )
    """Selection of model from with feature importances were created."""

    feature_importance_type: str = field(
        validator=[validators.in_(["mean", "count"])], default=Factory(lambda: "mean")
    )
    """Selection of feature importance type."""

    feature_importance_method: str = field(
        validator=[validators.in_(["permutation", "shap", "internal", "lofo"])],
        default=Factory(lambda: "permutation"),
    )
    """Selection of feature importance method."""
