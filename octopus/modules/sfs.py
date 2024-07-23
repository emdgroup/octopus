"""SFS Module (sequential feature selection)."""

import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment
from octopus.models.config import model_inventory
from octopus.modules.metrics import metrics_inventory


scorer_string_inventory = {
    "AUCROC": "roc_auc",
    "ACC": "accuracy_score",
    "ACCBAL": "balanced_accuracy",
    "LOGLOSS": "neg_log_loss",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}

# SFS module


# TASKS:
# - (1) use (A) catboost  model for RFE, as they work well with default parameters
# - (2) perform SFS on the given traindev dataset,
#   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector
# - (3) measure model performance in each step, select best model
# - (4) return list of selected features


@define
class SfsModule:
    """SFS Module."""

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

        # class sklearn.feature_selection.RFECV(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto')
        # I would suggest:
        #  step=1 (configurable)
        #  min_features_to_select=1 (configurable)
        #  cv = 5 (configurable)
        #  scoring - needs to be adjusted to the target metric
        #  verbose =  0
        #  n_jobs = 1 (configurate in module config)
        #  importance_getter = 'auto' # here it would be nice to use permutation feature importance
        #
        #

        # scoring - we need a sklearn scoring functions. This is provided with metrics_inventory
        # or use scoring_type, see above
        # scorer = metrics_inventory[target_metric]["method"]

        print("SFS completed")

        # save features selected by RFE
        self.experiment.selected_features = []  # update

        return self.experiment


# check input parameters
@define
class Sfs:
    """SFS Config."""

    module: str = field(default="sfs")
    """Module name."""

    description: str = field(validator=[validators.instance_of(str)], default="")
    """Description."""

    load_sequence_item: bool = field(
        init=False, validator=validators.instance_of(bool), default=False
    )
    """Load existing sequence item, fixed, set to False"""

    cv: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of CV folds for RFE_CV."""

    # correlation_type: str = field(
    #    validator=[validators.in_(["pearson", "rdc"])], default="pearson"
    # )
    # """Selection of correlation type."""
    #
    # feature_importance_type: str = field(
    #    validator=[validators.in_(["mean", "count"])], default="mean"
    # )
    # """Selection of feature importance type."""
    #
    # feature_importance_method: str = field(
    #    validator=[validators.in_(["permutation", "shap", "internal"])],
    #    default="permutation",
    # )
    # """Selection of feature importance method."""
