"""TabPFN Module."""

import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment

# TabPFN module


# TASKS:
# (1) limit to classification tasks
# (2) model initialization
# (3) model training
# (4) evaluation of model performance
# (6) calculation of feature importances
# (7) extract features used
# (8) save to experiment:
#      - predictions
#      - model
#      - feature importance tables,
#      - selected features
# (9) save performance results to json file


@define
class Tabpfn:
    """MRMR."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def path_module(self) -> Path:
        """Module path."""
        return self.experiment.path_study.joinpath(self.experiment.path_sequence_item)

    @property
    def path_results(self) -> Path:
        """Results path."""
        return self.path_module.joinpath("results")

    @property
    def x_traindev(self) -> pd.DataFrame:
        """x_train."""
        return self.experiment.data_traindev[self.experiment.feature_columns]

    @property
    def y_traindev(self) -> pd.DataFrame:
        """y_train."""
        return self.experiment.data_traindev[
            self.experiment.target_assignments.values()
        ]

    @property
    def x_test(self) -> pd.DataFrame:
        """x_test."""
        return self.experiment.data_test[self.experiment.feature_columns]

    @property
    def y_test(self) -> pd.DataFrame:
        """y_test."""
        return self.experiment.data_test[self.experiment.target_assignments.values()]

    @property
    def config(self) -> dict:
        """Module configuration."""
        return self.experiment.ml_config["config"]

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted
        # create directory if it does not exist
        for directory in [self.path_results]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """Run mrmr module on experiment."""
        # run experiment and return updated experiment object

        # limit to classification tasks
        # if self.experiment.ml_type == "classification":
        #    self.model = Model(**self.params)
        # else:
        #    raise ValueError(f"ML-type {self.experiment.ml_type} not supported")

        # model training
        # self.model.fit(
        #    self.x_traindev,
        #    self.y_traindev,
        # )
        print("fitting completed")

        # save features selected by tabpfn
        # self.experiment.selected_features =

        return self.experiment


# check input parameters
@define
class TabpfnConfig:
    """Tabpfn Config."""

    module: str = field(default="mrmr")
    """Models for ML."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

    description: str = field(validator=[validators.instance_of(str)], default=None)
    """Description."""

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
