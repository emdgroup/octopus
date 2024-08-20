"""Rfe core."""

import json
import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

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

supported_models = {
    "CatBoostClassifier",
    "CatBoostRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "XGBClassifier",
    "XGBRegressor",
}

# param_grid = {
#    'n_estimators': [50, 100, 200],
#    'max_depth': [None, 10, 20, 30],
#    'min_samples_split': [2, 5, 10]
# }

# param_grid = {
#    'iterations': [100, 200],
#    'depth': [4, 6],
#    'learning_rate': [0.01, 0.1],
#    #'random_state': [42],
#    'l2_leaf_reg': [1, 3]
# }


# handle feature importance for different modes
def get_feature_importances(estimator, *args, **kwargs):
    if hasattr(estimator, "best_estimator_"):
        return estimator.best_estimator_.feature_importances_
    else:
        return estimator.feature_importances_


# Hyperparameters initialization
def get_param_grid(model_type):
    if model_type == "CatBoostClassifier" or model_type == "CatBoostRegressor":
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "depth": [3, 5],
            "l2_leaf_reg": [2, 5],
            "random_strength": [5, 10],
            "rsm": [0.1, 1],
            "iterations": [500],
        }
    elif model_type == "XGBClassifier" or model_type == "XGBRegressor":
        param_grid = {
            "learning_rate": [0.0001, 0.01, 0.3],
            "min_child_weight": [2, 8, 15],
            "subsample": [0.15, 0.5, 1],
            "n_estimators": [30, 120, 200],
            "max_depth": [3, 5, 7, 9],
        }
    else:
        # RF and ExtraTrees
        param_grid = {
            "max_depth": [2, 15, 32],
            "min_samples_split": [2, 50, 100],
            "min_samples_leaf": [1, 25, 50],
            "max_features": [0.1, 0.5, 1],
            "n_estimators": [100, 250, 500],
        }
    return param_grid


# TOBEDONE:
# - put scorer_string_inventory in central place
# - print dev and test model performance after completion of rfe
# - do RFE with model hyperparameter optimization at each RFE step
# - use fixed parameters to silence catboost


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

        if model_type not in supported_models:
            raise ValueError(f"{model_type} not supported")
        print("Model used:", model_type)

        # set up model and scoring type
        model = model_inventory[model_type]["model"](random_seed=42)
        target_metric = self.experiment.configs.study.target_metric
        scoring_type = scorer_string_inventory[target_metric]

        # Silence catboost output
        if model_type == default_model:
            model.set_params(verbose=False)

        # Hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=get_param_grid(model_type),
            cv=self.config.cv,
            scoring=scoring_type,
            n_jobs=1,
        )

        # Perform Grid Search and Cross-Validation
        grid_search.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Report performance
        print(f"Best CV score: {best_cv_score}")
        print(f"Best params: {best_params}")

        # Mode selection
        if self.config.mode == "Mode1":
            # RFE with the trained model
            estimator = best_model
        elif self.config.mode == "Mode2":
            # RFE with hyperparameter optimization at each step
            estimator = grid_search

        print(f"Number of features before RFE: {self.x_traindev.shape[1]}")

        rfecv = RFECV(
            estimator=estimator,
            step=self.config.step,
            min_features_to_select=self.config.min_features_to_select,
            cv=self.config.cv,
            scoring=scoring_type,
            verbose=0,
            n_jobs=1,
            importance_getter=get_feature_importances,  # auto
        )

        rfecv.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        optimal_features = rfecv.n_features_
        self.experiment.selected_features = [
            self.feature_columns[i]
            for i in range(len(rfecv.support_))
            if rfecv.support_[i]
        ]

        print("RFE completed")
        print(f"Optimal number of features: {optimal_features}")
        print(f"Selected features: {self.experiment.selected_features}")
        print(f"CV Results: {rfecv.cv_results_}")

        # Report performance on test set
        test_score = rfecv.score(self.x_test, self.y_test)
        print(f"Test set performance: {test_score}")

        # Save results to JSON
        results = {
            "best_cv_score": best_cv_score,
            "best_params": best_params,
            "optimal_features": int(optimal_features),  # int32
            "selected_features": self.experiment.selected_features,
            "Mean CV Scores": (
                rfecv.cv_results_["mean_test_score"]
            ).tolist(),  # nd.array
            "Dev set performance": test_score,
        }
        with open(
            self.path_results.joinpath("results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=4)

        return self.experiment
