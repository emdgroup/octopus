"""Octopus Config."""

from typing import List

from attrs import define, field, validators


@define
class OctopusFullConfig:
    """OctopusLightConfig."""

    models: List = field(
        # validator=[validators.in_(["ExtraTreesRegressor", "RandomForestRegressor"])],
    )
    """Models for ML."""

    module: List = field(default="octofull")
    """Models for ML."""

    description: str = field(validator=[validators.instance_of(str)], default=None)
    """Description."""

    load_sequence_item: bool = field(
        validator=[validators.instance_of(bool)], default=False
    )
    """Load existing sequence item."""

    # datasplit
    n_folds_inner: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of inner folds."""

    datasplit_seed_inner: int = field(
        validator=[validators.instance_of(int)], default=0
    )
    """Data split seed for inner loops."""
    # model training

    model_seed: int = field(validator=[validators.instance_of(int)], default=0)
    """Model seed."""

    n_jobs: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of parallel jobs."""

    dim_red_methods: List = field(default=[""])
    """Methods for dimension reduction."""

    max_outl: int = field(validator=[validators.instance_of(int)], default=5)
    """Maximum number of outliers, optimized by Optuna"""

    fi_methods_bestbag: list = field(
        validator=[validators.instance_of(list)], default=[]
    )
    """Feature importance methods to be appleid to best bag"""

    # parallelization
    inner_parallelization: bool = field(
        validator=[validators.instance_of(bool)], default=False
    )

    n_workers: int = field(validator=[validators.instance_of(int)], default=None)
    """Number of workers."""
    # hyperparamter optimization
    optuna_seed: int = field(validator=[validators.instance_of(int)], default=None)
    """Seed for Optuna TPESampler, default=no seed"""

    n_optuna_startup_trials: int = field(
        validator=[validators.instance_of(int)], default=10
    )
    """Number of Optuna startup trials (random sampler)"""

    global_hyperparameter: bool = field(
        validator=[validators.in_([True, False])], default=True
    )
    """Selection of hyperparameter set."""

    ensemble_selection: bool = field(
        validator=[validators.in_([True, False])], default=False
    )
    """Whether to perform ensemble selection."""

    ensel_n_save_trials: int = field(
        validator=[validators.instance_of(int)], default=50
    )
    """Number of top trials to be saved for ensemble selection (bags)."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=100)
    """Number of Optuna trials."""

    hyperparameter: dict = field(validator=[validators.instance_of(dict)], default={})
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=0)
    """Maximum features."""

    penalty_factor: float = field(
        validator=[validators.instance_of(float)], default=1.0
    )
    """Factor to penalyse optuna target related to feature constraint."""

    resume_optimization: bool = field(
        validator=[validators.instance_of(bool)], default=False
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
