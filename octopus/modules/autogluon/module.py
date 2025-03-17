"""AutoGluon module."""

from typing import ClassVar, List, Literal, Optional, Union

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class AutoGluon(BaseSequenceItem):
    """AG Config."""

    module: ClassVar[str] = "autogluon"
    """Module name."""

    description: Optional[str] = field(
        default="", validator=validators.instance_of(str)
    )
    """Description."""

    time_limit: Optional[int] = field(
        default=None, validator=validators.optional(validators.instance_of(int))
    )
    """Approximately, how long a fit should run, in seconds. Default: No limit."""

    presets: List[str] = field(
        default=Factory(lambda: ["medium_quality"]),
        validator=validators.deep_iterable(
            member_validator=validators.and_(
                validators.instance_of(str),
                validators.in_(
                    [
                        "best_quality",
                        "high_quality",
                        "good_quality",
                        "medium_quality",
                        "experimental_quality",
                        "optimize_for_deployment",
                        "interpretable",
                        "ignore_text",
                    ]
                ),
            ),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """Autogluon presets."""

    num_cpus: Union[int, Literal["auto"]] = field(
        default="auto",
        validator=validators.optional(
            validators.or_(validators.instance_of(int), validators.in_(["auto"]))
        ),
    )
    """Number of CPUs used by Autogluon instance. Can be an integer or "auto"."""

    num_bag_folds: int = field(
        default=5, validator=[validators.instance_of(int), validators.gt(1)]
    )
    """Number of cross validation folds."""
