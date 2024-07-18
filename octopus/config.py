"""Opto Config module."""

import gzip
import pickle
from typing import Any, List

from attrs import define, field, validators

from octopus.modules.metrics import metrics_inventory


# Custom validator for target_metric
def validate_target_metric(instance: "ConfigStudy", attribute: Any, value: str) -> None:
    """Validate the target_metric based on the ml_type.

    Args:
        instance: The ConfigStudy instance being validated.
        attribute: The name of the attribute being validated.
        value: The value of the target_metric being validated.

    Raises:
        ValueError: If the target_metric is not valid for the given ml_type.
    """
    ml_type = instance.ml_type
    valid_metrics = [
        metric
        for metric, details in metrics_inventory.items()
        if details["ml_type"] == ml_type
    ]
    if value not in valid_metrics:
        raise ValueError(
            f"Invalid target metric '{value}' for ml_type '{ml_type}'. "
            f"Valid options are: {valid_metrics}"
        )


@define
class BaseSequenceItem:
    """Base class for all sequence items.

    Contains all common parameters for all sequence items.
    """

    description: str = field(validator=[validators.instance_of(str)])
    """Description for the sequence."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item. Default is False"""


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

    module: str = field(default="octofull")
    """Models for ML."""

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
    hyper_parameters: dict = field(validator=[validators.instance_of(dict)], default={})

    inner_parallelization: bool = field(
        validator=[validators.instance_of(bool)], default=False
    )

    n_workers: int = field(default=None)
    """Number of workers."""

    optuna_seed: int = field(validator=[validators.instance_of(int)], default=0)
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


@define
class MrmrConfig(BaseSequenceItem):
    """MRMR Config."""

    module: str = field(default="mrmr")
    """Models for ML."""

    n_features: int = field(validator=[validators.instance_of(int)], default=30)
    """Number of features selected by MRMR."""

    correlation_type: str = field(
        validator=[validators.in_(["pearson", "rdc"])], default="pearson"
    )
    """Selection of correlation type."""

    feature_importance_type: str = field(
        validator=[validators.in_(["mean", "count"])], default="mean"
    )
    """Selection of feature importance type."""

    feature_importance_method: str = field(
        validator=[validators.in_(["permutation", "shap", "internal"])],
        default="permutation",
    )
    """Selection of feature importance method."""


@define
class ConfigStudy:
    """Configuration for study parameters."""

    name: str = field()
    """The name of the study."""

    ml_type: str = field(
        validator=[
            validators.in_(["classification", "regression", "timetoevent"]),
        ]
    )
    """The type of machine learning model.
    Choose from "classification", "regression" or "timetoevent"."""

    target_metric: str = field(validator=[validate_target_metric])
    """The primary metric used for model evaluation."""

    path: str = field(default="./studies/")
    """The path where study outputs are saved. Defaults to "./studies/"."""

    start_with_empty_study: bool = field(default=True)

    n_folds_outer: int = field(default=5)
    """The number of outer folds for cross-validation. Defaults to 5."""

    datasplit_seed_outer: int = field(default=0)
    """The seed used for data splitting in outer cross-validation. Defaults to 0."""

    # is this really useful?
    metrics: List = field(
        default=["AUCROC", "ACCBAL", "ACC", "LOGLOSS", "MAE", "MSE", "R2", "CI"],
    )
    """A list of metrics to be calculated.
    Defaults to ["AUCROC", "ACCBAL", "ACC", "LOGLOSS", "MAE", "MSE", "R2", "CI"]."""


@define
class ConfigManager:
    """Configuration for manager parameters.

    Will later be used to connect to HPC.
    """

    outer_parallelization: bool = field(default=False)
    """Indicates whether outer parallelization is enabled. Defaults to False."""

    run_single_experiment_num: int = field(default=-1)
    """Select a single experiment to execute. Defaults to -1 to run all experiments"""

    production_mode: bool = field(default=True)
    """Indicates whether the study is in production mode. Defaults to True."""


@define
class ConfigSequence:
    """Configuration for sequence parameters.

    Attributes:
        sequence_items (List[BaseSequenceItem]):
    """

    sequence_items: List[BaseSequenceItem] = field(factory=list)
    """A list of sequence items that define the sequence of operations.
    Each item in the list is an instance of `BaseSequenceItem` or its subclasses."""


@define
class OctoConfig:
    """Main configuration class that holds all other configurations."""

    study: ConfigStudy = field(factory=ConfigStudy)
    """Configuration for study parameters."""

    manager: ConfigManager = field(factory=ConfigManager)
    """Configuration for manager parameters."""

    sequence: ConfigSequence = field(factory=ConfigSequence)
    """Configuration for sequence parameters."""

    def to_pickle(self, file_path: str) -> None:
        """Save object to a compressed pickle file.

        Args:
            file_path: The name of the file to save the pickle data to.
        """
        with gzip.GzipFile(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file_path: str) -> "OctoConfig":
        """Load object from a compressed pickle file.

        Args:
            file_path: The path to the file to load the pickle data from.


        Returns:
            OctoConfig: The loaded instance of OctoConfig.
        """
        with gzip.GzipFile(file_path, "rb") as file:
            return pickle.load(file)
