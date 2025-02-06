"""Boruta Module."""

from typing import ClassVar

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Boruta(BaseSequenceItem):
    """Boruta Config."""

    module: ClassVar[str] = "boruta"
    """Module name."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""

    model: str = field(validator=[validators.instance_of(str)], default="")
    """Model used by Boruta."""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of folds for CV."""

    perc: int = field(validator=[validators.instance_of(int)], default=100)
    """Percentile (threshold) for comparison between shadow and real features."""

    alpha: int = field(validator=[validators.instance_of(float)], default=0.05)
    """Level at which the corrected p-values will get rejected."""

    # two_step: bool = field(
    #     validator=validators.instance_of(bool),
    #     default=Factory(lambda: True),
    # )
    # """To use the original implementation of Boruta, set this to False"""

    # max_iter: int = field(validator=[validators.instance_of(int)], default=100)
    # """The number of maximum iterations to perform."""

    # early_stopping: bool = field(
    #     validator=validators.instance_of(bool),
    #     default=Factory(lambda: False),
    # )
    # """To terminate the selection process before reaching `max_iter` iterations"""

    # n_iter_no_change: int = field(validator=[validators.instance_of(int)], default=20)
    # """Iterations without confirming a tentative feature."""
