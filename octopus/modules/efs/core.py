"""Efs core."""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define, field, validators
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
)

from octopus.experiment import OctoExperiment
from octopus.metrics import metrics_inventory
from octopus.models.models_inventory import model_inventory

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
            "learning_rate": [0.03],
            # "learning_rate": [0.001, 0.03, 0.1],
            # "depth": [3, 6, 8, 10],
            # "l2_leaf_reg": [2, 5, 7, 10],
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
            "max_depth": [2, 10, 20, 32],
            # "min_samples_split": [2, 25, 50, 100],
            # "min_samples_leaf": [1, 15, 30, 50],
            # "max_features": [0.1, 0.5, 1],
            # "n_estimators": [100, 250, 500],
        }
    return param_grid


# TOBEDONE:
# - should be done with a proper datasplit (see octo, stratification + groups)
# - should be done with optuna
# - make it work with timetoevent
# - paramters
#   + size of subset
#   + number of models to be trained
# - train a large number of models using feature subsets
# - perform hillclimb
# - perform optimization
# - reproducibility issue, seems to come from earlier sequence item


@define
class EfsCore:
    """EFS Module."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    model_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
    )
    scan_table: pd.DataFrame = field(
        init=False,
        default=pd.DataFrame(),
        validator=[validators.instance_of(pd.DataFrame)],
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
    def ml_type(self) -> str:
        """ML type."""
        return self.experiment.ml_type

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

    @property
    def stratification_column(self) -> list:
        """Stratification Column."""
        return self.experiment.stratification_column

    @property
    def target_metric(self) -> list:
        """Target metric."""
        return self.experiment.configs.study.target_metric

    @property
    def score_type(self) -> str:
        """Score type."""
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            return "dev_pool_soft"
        else:
            return "dev_pool_hard"

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted
        # create directory if it does not exist
        for directory in [self.path_results]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """Run EFS module on experiment."""
        # run experiment and return updated experiment object
        np.random.seed(0)
        random.seed(0)

        # print()
        # print("features: ", self.feature_columns)

        # Configuration, define default model
        if self.ml_type == "classification":
            default_model = "CatBoostClassifier"
        elif self.ml_type == "regression":
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
        scoring_type = scorer_string_inventory[self.target_metric]

        # needs general improvements (consider groups and stratification column)
        stratification_column = self.experiment.stratification_column
        if stratification_column:
            cv = StratifiedKFold(n_splits=self.config.cv, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.config.cv, shuffle=True, random_state=42)

        # Silence catboost output
        if model_type == default_model:
            model.set_params(verbose=False)

        # Define GridSearch for HPO
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=get_param_grid(model_type),
            cv=cv,
            scoring=scoring_type,
            n_jobs=1,
        )

        # Create features subsets

        subsets = []
        for _ in range(self.config.n_subsets):
            subset = random.sample(self.feature_columns, self.config.subset_size)
            subsets.append(subset)

        # (A) create model table
        # train (gridsearch) on all subsets
        df_lst = list()
        for i, subset in enumerate(subsets):
            # Perform Grid Search and Cross-Validation
            # print("subset:", subset)
            x = self.x_traindev[subset]
            y = self.y_traindev.squeeze(axis=1)
            grid_search.fit(x, y)
            best_model = grid_search.best_estimator_
            # best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            print(f"Subset {i}, best_cv_score: {best_cv_score:.4f}")
            # cv_score = cross_val_score(best_model, x, y, cv=cv, scoring=scoring_type)

            # predictions
            if self.ml_type == "classification":
                cv_predictions = cross_val_predict(
                    best_model, x, y, cv=cv, method="predict"
                )
                cv_probabilities = cross_val_predict(
                    best_model, x, y, cv=cv, method="predict_proba"
                )[
                    :, 1
                ]  # binary only
            elif self.ml_type == "regression":
                cv_predictions = cross_val_predict(
                    best_model, x, y, cv=cv, method="predict"
                )
                cv_probabilities = None

            # ensemble metric
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                best_ensel_score = metrics_inventory[self.target_metric]["method"](
                    y, cv_probabilities
                )
            else:
                best_ensel_score = metrics_inventory[self.target_metric]["method"](
                    y, cv_predictions
                )
            print(f"Subset {i}, best ensemble score: {best_ensel_score:.4f}")
            print()

            # store results
            s = pd.Series()
            s["id"] = i
            s["score"] = best_cv_score
            s["features"] = subset
            s["predictions"] = cv_predictions
            s["probabilities"] = cv_probabilities
            df_lst.append(s)

        self.model_table = pd.concat(df_lst, axis=1).T

        # oder of model table is important, since we are using
        # the score, we look for maximum score (ascending=False)
        self.model_table = self.model_table.sort_values(
            by="score", ascending=False
        ).reset_index(drop=True)

        print(self.model_table)

        ## (B) perform ensemble scan, hillclimb
        if self.score_type == "dev_pool_soft":
            self.scan_table = pd.DataFrame(
                columns=[
                    "#models",
                    "dev_pool_hard",
                    "dev_pool_soft",
                ]
            )
        else:
            self.scan_table = pd.DataFrame(columns=["#models", "dev_pool_hard"])

        for i in range(1, len(self.model_table)):
            scores = self._ensemble_models(self.model_table[:i])
            if self.score_type == "dev_pool_soft":
                self.scan_table.loc[i] = [
                    i,
                    scores["dev_pool_hard"],
                    scores["dev_pool_soft"],
                ]
            else:
                self.scan_table.loc[i] = [
                    i,
                    scores["dev_pool_hard"],
                ]

        print("EFS completed")

        # update selected features in experiment

        # Report performance on test set

        # Save results to JSON
        results = {
            "best_cv_score": best_cv_score,
            # "best_params": best_params,
            # "optimal_features": int(optimal_features),
            # "selected_features": self.experiment.selected_features,
            # "Best Mean CV Score": max(rfecv.cv_results_["mean_test_score"]),
            # "Dev set performance": test_score,
        }
        with open(
            self.path_results.joinpath("results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=4)

        return self.experiment

    def _ensemble_models(self, model_table):
        pass
