"""Rfe2 module."""

from attrs import define, field, validators

from octopus.modules.octo.module import Octo


@define
class Rfe2(Octo):
    """Rf2 sequence config."""

    module: str = field(default="rfe2")
    """Models for ML."""

    step: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of features to remove at each iteration."""

    min_features_to_select: int = field(
        validator=[validators.instance_of(int)], default=1
    )
    """Minimum number of features to be selected."""

    fi_method_rfe: str = field(
        validator=[validators.in_(["permutation", "shap"])], default="permutation"
    )
    """Featur importance method for RFE."""

    def __attrs_post_init__(self):
        # overwrite fi_mehods_bestbag
        self.fi_methods_bestbag = [self.fi_method_rfe]
