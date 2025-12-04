"""ROC core (removal of correlated features)."""

from typing import ClassVar

from attrs import define, field, validators

from octopus.task import Task


@define
class Roc(Task):
    """Roc Config."""

    module: ClassVar[str] = "roc"
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    threshold: float = field(validator=[validators.instance_of(float)], default=0.8)
    """Threshold for feature removal."""

    correlation_type: str = field(validator=[validators.in_(["spearmanr", "rdc"])], default="spearmanr")
    """Selection of correlation type."""

    filter_type: str = field(
        validator=[validators.in_(["mutual_info", "f_statistics"])],
        default="f_statistics",
    )
    """Selection of filter type for correlated features."""
