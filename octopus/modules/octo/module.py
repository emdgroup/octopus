"""Octo module."""

from typing import List

from attrs import Factory, define, field, validators

from octopus.config.base_sequence_item import BaseSequenceItem


@define
class Octo(BaseSequenceItem):
    """Octofull sequence config.

    Model options:
        Classification:
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
            "CatBoostClassifier",
            "TabPFNClassifier"


        Regression:
            "ExtraTreesRegressor",
            "RandomForestRegressor",
            "XGBRegressor",
            "RidgeRegressor",
            "ElasticNetRegressor",
            "ARDRegressor",
            "GradientBoostingRegressor",
            "SvrRegressor",
            "CatBoostRegressor"


        Time to event:
            "ExtraTreesSurv"


    """

    models: List = field()
    """Models for ML."""

    module: str = field(default=Factory(lambda: "octo"))
    """Models for ML."""

    load_sequence_item: bool = field(
        validator=validators.instance_of(bool),
        default=Factory(lambda: False),
    )
    """Load existing sequence item. Default is False"""

    n_folds_inner: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 5)
    )
    """Number of inner folds."""

    datasplit_seed_inner: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 0)
    )
    """Data split seed for inner loops."""
    # model training

    model_seed: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 0)
    )
    """Model seed."""

    n_jobs: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 1)
    )
    """Number of parallel jobs."""

    dim_red_methods: List = field(default=Factory(lambda: [""]))
    """Methods for dimension reduction."""

    max_outl: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 5)
    )
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods_bestbag: list = field(
        validator=[validators.instance_of(list)],
        default=Factory(lambda: ["permutation"]),
    )
    """Feature importance methods to be appleid to best bag"""

    inner_parallelization: bool = field(
        validator=[validators.instance_of(bool)], default=Factory(lambda: False)
    )
    """Enable inner paralization. Defaults is False."""

    n_workers: int = field(default=Factory(lambda: None))
    """Number of workers."""

    optuna_seed: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 0)
    )
    """Seed for Optuna TPESampler, default=no seed"""

    n_optuna_startup_trials: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 10)
    )
    """Number of Optuna startup trials (random sampler)"""

    global_hyperparameter: bool = field(
        validator=[validators.in_([True, False])], default=Factory(lambda: True)
    )
    """Selection of hyperparameter set."""

    ensemble_selection: bool = field(
        validator=[validators.in_([True, False])], default=Factory(lambda: False)
    )
    """Whether to perform ensemble selection."""

    ensel_n_save_trials: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 50)
    )
    """Number of top trials to be saved for ensemble selection (bags)."""

    n_trials: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 100)
    )
    """Number of Optuna trials."""

    hyper_parameters: dict = field(
        validator=[validators.instance_of(dict)], default=Factory(dict)
    )
    """Bring own hyperparameter space."""

    max_features: int = field(
        validator=[validators.instance_of(int)], default=Factory(lambda: 0)
    )
    """Maximum features."""

    penalty_factor: float = field(
        validator=[validators.instance_of(float)], default=Factory(lambda: 1.0)
    )
    """Factor to penalyse optuna target related to feature constraint."""

    resume_optimization: bool = field(
        validator=[validators.instance_of(bool)], default=Factory(lambda: False)
    )
    """Resume HPO, use existing optuna.db, don't delete optuna.de"""

    def __attrs_post_init__(self):
        # set default of n_workers to n_folds_inner
        if self.n_workers is None:
            self.n_workers = self.n_folds_inner
        if self.n_workers != self.n_folds_inner:
            print(
                f"Octofull Warning: n_workers ({self.n_workers}) "
                f"does not match n_folds_inner ({self.n_folds_inner})",
            )
