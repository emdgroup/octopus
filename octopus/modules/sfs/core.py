"""SFS Core (sequential feature selection)."""

import copy
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define, field, validators
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from octopus.experiment import OctoExperiment
from octopus.models.models_inventory import model_inventory
from octopus.results import ModuleResults

# Ignore all Warnings
warnings.filterwarnings("ignore")

scorer_string_inventory = {
    "AUCROC": "roc_auc",
    "ACC": "accuracy",
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


def get_param_grid(model_type):
    """Hyperparameter grid initialization."""
    if model_type in ("CatBoostClassifier", "CatBoostRegressor"):
        param_grid = {
            "learning_rate": [0.001, 0.01, 0.1],
            "depth": [3, 6, 8, 10],
            "l2_leaf_reg": [2, 5, 7, 10],
            #'random_strength': [2, 5, 7, 10],
            #'rsm': [0.1, 0.5, 1],
            "iterations": [500],
        }
    elif model_type in ("XGBClassifier", "XGBRegressor"):
        param_grid = {
            "learning_rate": [0.0001, 0.001, 0.01, 0.3],
            "min_child_weight": [2, 5, 10, 15],
            "subsample": [0.15, 0.3, 0.7, 1],
            "n_estimators": [30, 70, 140, 200],
            "max_depth": [3, 5, 7, 9],
        }
    else:
        # RF and ExtraTrees
        param_grid = {
            "max_depth": [3, 6, 10],  # [2, 10, 20, 32],
            # "min_samples_split": [2, 25, 50, 100],
            # "min_samples_leaf": [1, 15, 30, 50],
            # "max_features": [0.1, 0.5, 1],
            "n_estimators": [500],  # [100, 250, 500],
        }
    return param_grid


# TOBEDONE/IDEAS:
# - put scorer_string_inventory in central place


@define
class SfsCore:
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

    @property
    def stratification_column(self) -> list:
        """Stratification Column."""
        return self.experiment.stratification_column

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
        model = model_inventory[model_type]["model"](random_state=42)
        target_metric = self.experiment.configs.study.target_metric
        scoring_type = scorer_string_inventory[target_metric]

        stratification_column = self.experiment.stratification_column
        if stratification_column:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = self.config.cv

        # Silence catboost output
        if model_type == default_model:
            model.set_params(verbose=False, allow_writing_files=False)

        # Hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=get_param_grid(model_type),
            cv=cv,
            scoring=scoring_type,
            n_jobs=1,
        )
        print("Optimize base model....")
        # Perform Grid Search and Cross-Validation
        grid_search.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        best_model = grid_search.best_estimator_
        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        # Report performance
        print(f"Dev start performance: {best_cv_score:.3f}")
        print(f"Best params: {best_params}")

        print(f"Number of features before SFS: {self.x_traindev.shape[1]}")

        # Select type of SFS
        if self.config.sfs_type == "forward":
            forward = True
            floating = False
        elif self.config.sfs_type == "backward":
            forward = False
            floating = False
        elif self.config.sfs_type == "floating_forward":
            forward = True
            floating = True
        elif self.config.sfs_type == "floating_backward":
            forward = False
            floating = True
        else:
            raise ValueError(f"Unsupported SFS type: {self.config.sfs_type}")

        sfs = SFS(
            estimator=best_model,
            k_features="best",
            forward=forward,
            floating=floating,
            cv=cv,
            scoring=scoring_type,
            verbose=0,
            n_jobs=1,
        )

        sfs.fit(self.x_traindev, self.y_traindev.squeeze(axis=1))
        n_optimal_features = len(sfs.k_feature_idx_)
        self.experiment.selected_features = list(sfs.k_feature_names_)

        print("SFS completed")
        # print(sfs.subsets_)
        print(f"Optimal number of features: {n_optimal_features}")
        print(f"Selected features: {self.experiment.selected_features}")
        print(f"Dev set performance: {sfs.k_score_:.3f}")

        # Report performance on test set
        best_estimator = copy.deepcopy(best_model)
        x_traindev_sfs = sfs.transform(self.x_traindev)
        x_test_sfs = sfs.transform(self.x_test)

        cv_score = cross_val_score(
            best_model, x_test_sfs, self.y_test, scoring=scoring_type, cv=cv
        )
        test_score_cv = np.mean(cv_score)
        print(f"Test set (cv) performance : {test_score_cv:.3f}")

        # retrain best model on x_traindev
        best_estimator.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))
        test_score_refit = best_estimator.score(x_test_sfs, self.y_test)
        print(f"Test set (refit) performance: {test_score_refit:.3f}")

        # gridsearch + retrain best model on x_traindev
        grid_search.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))
        best_gs_parameters = grid_search.best_params_
        best_gs_estimator = grid_search.best_estimator_
        best_gs_estimator.fit(x_traindev_sfs, self.y_traindev.squeeze(axis=1))  # refit
        test_score_gsrefit = best_gs_estimator.score(x_test_sfs, self.y_test)
        print(f"Test set (gridsearch+refit) performance: {test_score_gsrefit:.3f}")

        # save results to experiment
        self.experiment.results["Sfs"] = ModuleResults(
            id="SFS",
            model=best_gs_estimator,
            # scores=scores,
            selected_features=self.experiment.selected_features,
        )

        # Save results to JSON
        results = {
            "Dev score start": best_cv_score,
            "best_params": best_gs_parameters,
            "optimal_features": int(n_optimal_features),
            "selected_features": self.experiment.selected_features,
            "Dev set performance": sfs.k_score_,
            "Test set performance": test_score_gsrefit,
        }
        with open(
            self.path_results.joinpath("results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=4)

        return self.experiment
