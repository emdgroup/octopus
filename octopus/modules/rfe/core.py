"""Rfe core."""

import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators
from sklearn.feature_selection import RFECV

from octopus.experiment import OctoExperiment
from octopus.models.models_inventory import model_inventory

scorer_string_inventory = {
    "AUCROC": "roc_auc",
    "ACC": "accuracy_score",
    "ACCBAL": "balanced_accuracy",
    "LOGLOSS": "neg_log_loss",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}
# TOBEDONE:
# - put scorer_string_inventory in central place
# - print dev and test model performance after completion of rfe
# - do RFE with model hyperparameter optimization at each RFE step
# - use fixed parameters to silence catboost

# RFE with grid search for HPO
# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import RFECV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# Create a synthetic dataset
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
# Define the model
# model = RandomForestClassifier(random_state=42)
# Define the parameter grid for hyperparameter tuning
# param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [None, 10, 20, 30],
#    'min_samples_split': [2, 5, 10]
# }
# Define the cross-validation strategy
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Create the GridSearchCV object
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
# Create the RFECV object with GridSearchCV as the estimator
# rfecv = RFECV(estimator=grid_search, step=1, cv=cv, scoring='accuracy', n_jobs=-1)
# Fit the RFECV model
# rfecv.fit(X, y)
# Print the optimal number of features
# print("Optimal number of features: %d" % rfecv.n_features_)
# Print the selected features
# print("Selected features: %s" % np.where(rfecv.support_)[0])
# Print the best hyperparameters found
# print("Best hyperparameters found: %s" % rfecv.estimator_.best_params_)


@define
class RfeCore:
    """RFE Module."""

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

    @property
    def feature_columns(self) -> pd.DataFrame:
        """Feature Columns."""
        return self.experiment.feature_columns

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

        # Configuration, define default model
        if self.experiment.ml_type == "classification":
            default_model = "CatBoostClassifier"
        elif self.experiment.ml_type == "regression":
            default_model = "CatBoostRegressor"
        else:
            raise ValueError(f"{self.experiment.ml_type} not supported")

        model_type = self.config.model
        if model_type == "":
            model_type = default_model
        print("Model used:", model_type)

        # set up model and scoring type
        model = model_inventory[model_type]["model"]()
        target_metric = self.experiment.configs.study.target_metric
        scoring_type = scorer_string_inventory[target_metric]

        rfecv = RFECV(
            estimator=model,
            step=self.config.step,
            min_features_to_select=self.config.min_features_to_select,
            cv=self.config.cv,
            scoring=scoring_type,
            verbose=0,
            n_jobs=1,
            importance_getter="auto",
        )

        rfecv.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))

        print("RFE completed")
        print(f"Optimal number of features: {rfecv.n_features_}")

        # save features selected by RFE
        self.experiment.selected_features = [
            self.feature_columns[i]
            for i in range(len(rfecv.support_))
            if rfecv.support_[i]
        ]
        print(f"Selected features: {self.experiment.selected_features}")

        return self.experiment
