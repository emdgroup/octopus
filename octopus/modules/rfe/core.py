"""Rfe core."""

import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment


@define
class RfeCore:
    """RFE Core."""

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
        return self.experiment.ml_config

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted
        # create directory if it does not exist
        for directory in [self.path_results]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """Run RFE module on experiment."""
        # run experiment and return updated experiment object

        # Inputs:
        # - self.config (see above) gives you access all config parameters
        # - self.experiment.ml_type == "classification"  ["regression"]
        # - self.x_traindev
        # - self.y_traindev
        # - self.x_test
        # - self.y_test
        # - self.experiment.feature_columns
        # - target_metric = self.experiment.config["target_metric"]  #target metric
        # - scoring_type = scorer_string_inventory[target_metric] # scoring type string
        # - model = model_inventory[model_type]
        # - model_type = "CatBoostRegressor" or "CatBoostClassifier"

        # class sklearn.feature_selection.RFECV(estimator, *,
        # step=1, min_features_to_select=1,
        # cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto')
        # I would suggest:
        #  step=1 (configurable)
        #  min_features_to_select=1 (configurable)
        #  cv = 5 (configurable)
        #  scoring - needs to be adjusted to the target metric
        #  verbose =  0
        #  n_jobs = 1 (configurate in module config)
        #  importance_getter = 'auto' # here it would be nice to use
        #  permutation feature importance
        #
        #

        # scoring - we need a sklearn scoring functions.
        # This is provided with metrics_inventory
        # or use scoring_type, see above
        # scorer = metrics_inventory[target_metric]["method"]

        print("RFE completed")

        # save features selected by RFE
        self.experiment.selected_features = []  # update

        return self.experiment
