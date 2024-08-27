"""ROC core (removal of correlated features)."""

from attrs import define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Roc(BaseSequenceItem):
    """Roc Config."""

    module: str = field(default="roc")
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

    threshold: float = field(validator=[validators.instance_of(float)], default=0.8)
    """Threshold for feature removal."""

    correlation_type: str = field(
        validator=[validators.in_(["spearmanr", "rdc"])], default="spearmanr"
    )
    """Selection of correlation type."""

    filter_type: str = field(
        validator=[validators.in_(["mutual_info", "f_statistics"])],
        default="f_statistics",
    )
    """Selection of filter type for correlated features."""
