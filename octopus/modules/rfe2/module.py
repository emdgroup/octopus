"""Rfe2 module."""

from typing import ClassVar

from attrs import define, field, validators

from octopus.modules.octo.module import Octo


@define
class Rfe2(Octo):
    """Rf2 sequence config."""

    module: ClassVar[str] = "rfe2"
    """Module name."""

    # step: int = field(validator=[validators.instance_of(int)], default=1)
    # """Number of features to remove at each iteration."""

    min_features_to_select: int = field(
        validator=[validators.instance_of(int)], default=1
    )
    """Minimum number of features to be selected."""

    fi_method_rfe: str = field(
        validator=[validators.in_(["permutation", "shap"])], default="permutation"
    )
    """Feature importance method for RFE."""

    selection_method: str = field(
        validator=[validators.in_(["best", "parsimonious"])], default="best"
    )
    """Method to select best solution. Parimonious: smallest solutions within sem."""

    abs_on_fi: bool = field(validator=[validators.instance_of(bool)], default=False)
    """Convert negative feature importances to positive (abs())."""

    def __attrs_post_init__(self):
        # overwrite fi_methods_bestbag
        self.fi_methods_bestbag = [self.fi_method_rfe]
