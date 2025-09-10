"""Octo module."""

from typing import ClassVar

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem
from octopus.logger import get_logger

logger = get_logger()


@define
class Octo(BaseSequenceItem):
    """Octofull sequence config."""

    models: list = field(default=Factory(lambda: ["ExtraTreesClassifier"]))
    """Models for ML."""

    module: ClassVar[str] = "octo"
    """Module name."""

    n_folds_inner: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 5))
    """Number of inner folds."""

    datasplit_seeds_inner: list[int] = field(
        default=Factory(lambda: [0]),
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(int),
            iterable_validator=validators.instance_of(list),
        ),
    )
    """List of integers used as seeds for data splitting."""
    # model training

    model_seed: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Model seed."""

    n_jobs: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 1))
    """Number of parallel jobs. Ideal if n_folds_inner * number of datasplit seeds."""

    dim_red_methods: list = field(default=Factory(lambda: [""]))
    """Methods for dimension reduction."""

    max_outl: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 3))
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods_bestbag: list[str] = field(
        default=Factory(lambda: ["permutation"]),
        validator=validators.deep_iterable(
            member_validator=validators.in_(["permutation", "shap", "constant"]),
            iterable_validator=validators.instance_of(list),
        ),
    )

    inner_parallelization: bool = field(validator=[validators.instance_of(bool)], default=Factory(lambda: False))
    """Enable inner paralization. Defaults is False."""

    n_workers: int = field(default=Factory(lambda: None))
    """Number of workers."""

    optuna_seed: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Seed for Optuna TPESampler, default=0"""

    n_optuna_startup_trials: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 10))
    """Number of Optuna startup trials (random sampler)"""

    global_hyperparameter: bool = field(validator=[validators.in_([True, False])], default=Factory(lambda: True))
    """Selection of hyperparameter set."""

    ensemble_selection: bool = field(validator=[validators.in_([True, False])], default=Factory(lambda: False))
    """Whether to perform ensemble selection."""

    ensel_n_save_trials: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 50))
    """Number of top trials to be saved for ensemble selection (bags)."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 100))
    """Number of Optuna trials."""

    hyperparameters: dict = field(validator=[validators.instance_of(dict)], default=Factory(dict))
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=Factory(lambda: 0))
    """Maximum features to constrain hyperparameter optimization."""

    penalty_factor: float = field(validator=[validators.instance_of(float)], default=Factory(lambda: 1.0))
    """Factor to penalyse optuna target related to feature constraint."""

    mrmr_feature_numbers: list = field(validator=[validators.instance_of(list)], default=Factory(list))
    """List of feature numbers to be investigated by mrmr."""

    resume_optimization: bool = field(validator=[validators.instance_of(bool)], default=Factory(lambda: False))
    """Resume HPO, use existing optuna.db, don't delete optuna.db"""

    optuna_return: str = field(default="pool", validator=[validators.in_(["pool", "average"])])
    """How to calculate the bag performance for the optuna optimization target."""

    def __attrs_post_init__(self):
        # set default of n_workers to n_folds_inner
        if self.n_workers is None:
            self.n_workers = self.n_folds_inner
        if self.n_workers != self.n_folds_inner:
            logger.warning(
                f"Octofull Warning: n_workers ({self.n_workers}) does not match n_folds_inner ({self.n_folds_inner})",
            )
