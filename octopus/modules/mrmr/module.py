"""Mrmr module."""

from typing import ClassVar, Literal

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Mrmr(BaseSequenceItem):
    """MRMR Config."""

    module: ClassVar[str] = "mrmr"
    """Module name."""

    n_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 30))
    """Number of features selected by MRMR."""

    correlation_type: Literal["pearson", "spearman", "rdc"] = field(
        validator=validators.in_(["pearson", "rdc", "spearmanr"]), default="spearmanr"
    )
    """Selection of correlation type."""

    relevance_type: Literal["permutation", "f-statistics"] = field(
        validator=validators.in_(["permutation", "f-statistics"]), default="permutation"
    )
    """Selection of relevance measure."""

    results_key: str = field(validator=validators.in_(["best", "ensel", "autogluon"]), default="best")
    """Selection of model from with feature importances were created."""

    feature_importance_type: Literal["mean", "count"] = field(
        validator=validators.in_(["mean", "count"]), default="mean"
    )
    """Selection of feature importance type."""

    feature_importance_method: Literal["permutation", "shap", "internal", "lofo"] = field(
        validator=validators.in_(["permutation", "shap", "internal", "lofo"]), default="permutation"
    )
    """Selection of feature importance method."""
