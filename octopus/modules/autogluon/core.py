"""Module: Autogluon Tabular."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define, field, validators
from autogluon.core.metrics import (
    accuracy,
    average_precision,
    balanced_accuracy,
    f1,
    log_loss,
    mcc,
    mean_absolute_error,
    precision,
    r2,
    recall,
    roc_auc,
    root_mean_squared_error,
)
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from octopus.experiment import OctoExperiment
from octopus.logger import LogGroup, get_logger
from octopus.modules.autogluon.module import AutoGluon
from octopus.modules.octo.ray_parallel import setup_ray_for_external_library
from octopus.modules.utils import (
    get_fi_group_shap,
    get_fi_shap,
    get_score_from_model,
)
from octopus.results import ModuleResults

logger = get_logger()


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn classifier wrapper."""

    def __init__(self, predictor: TabularPredictor):
        self.predictor = predictor
        self.classes_ = self.predictor.class_labels  # Class labels
        self.n_classes_ = len(self.classes_)  # Number of classes
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)
        self._is_fitted = True  # Indicate that the model is fitted

    def predict(self, x):
        """Predict."""
        return self.predictor.predict(x, as_pandas=False)

    def predict_proba(self, x):
        """Predict proba."""
        probabilities = self.predictor.predict_proba(x, as_pandas=False, as_multiclass=True)
        return probabilities  # Return as NumPy array

    def fit(self, x, y):
        """Fit."""
        raise NotImplementedError("This classifier is already fitted. Only for inference use.")


class SklearnRegressor(BaseEstimator, RegressorMixin):
    """Sklearn regressor wrapper."""

    def __init__(self, predictor: TabularPredictor):
        self.predictor = predictor
        self.feature_names_in_ = self.predictor.original_features
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_outputs_ = 1  # Assuming single-output regression;
        self._is_fitted = True  # Indicate that the model is fitted

    def predict(self, x):
        """Predict."""
        # Call the predictor's predict method with as_pandas=False
        return self.predictor.predict(x, as_pandas=False)

    def fit(self, x, y):
        """Fit."""
        raise NotImplementedError("This regressor is already fitted. Only for inference use.")


# mapping of metrics
try:
    metrics_inventory_autogluon = {
        "AUCROC": roc_auc,
        "ACC": accuracy,
        "ACCBAL": balanced_accuracy,
        "AUCPR": average_precision,
        "F1": f1,
        "LOGLOSS": log_loss,
        "MAE": mean_absolute_error,
        "MCC": mcc,
        "MSE": root_mean_squared_error,
        "NEGBRIERSCORE": "brier_score_loss",
        "PRECISION": precision,
        "RECALL": recall,
        "R2": r2,
        "RMSE": root_mean_squared_error,  # RMSE is equivalent to MSE in autogluon
    }
except Exception as e:  # pylint: disable=W0718 # noqa: F841
    metrics_inventory_autogluon = {}


@define
class AGCore:
    """Autogluon."""

    experiment: OctoExperiment[AutoGluon] = field(validator=[validators.instance_of(OctoExperiment)])
    model = field(init=False)
    num_cpus = field(init=False)  # TODO: this is also in the AutoGluon class

    @property
    def path_module(self) -> Path:
        """Module path."""
        return self.experiment.path_study.joinpath(self.experiment.sequence_item_path)

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
        return self.experiment.data_traindev[self.experiment.target_assignments.values()]

    @property
    def x_test(self) -> pd.DataFrame:
        """x_test."""
        return self.experiment.data_test[self.experiment.feature_columns]

    @property
    def y_test(self) -> pd.DataFrame:
        """y_test."""
        return self.experiment.data_test[self.experiment.target_assignments.values()]

    @property
    def target_assignments(self) -> dict:
        """Target assignments."""
        return self.experiment.target_assignments

    @property
    def target_metric(self) -> str:
        """Target metric."""
        return self.experiment.configs.study.target_metric

    @property
    def metrics(self) -> list[str]:
        """Metrics."""
        return self.experiment.configs.study.metrics

    @property
    def config(self) -> AutoGluon:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def feature_columns(self) -> list[str]:
        """Feature Columns."""
        return self.experiment.feature_columns

    @property
    def ag_train_data(self) -> pd.DataFrame:
        """Autogluon (ag) training data."""
        return pd.concat([self.x_traindev, self.y_traindev], axis=1)

    @property
    def ag_test_data(self) -> pd.DataFrame:
        """Autogluon (ag) test data."""
        return pd.concat([self.x_test, self.y_test], axis=1)

    def __attrs_post_init__(self):
        # delete directories /results to ensure clean state
        # create directory if it does not exist
        for directory in [self.path_results]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        # set and validate resources assigned to the experiment
        self._set_resources()

    def _set_resources(self):
        """Set and validate resources."""
        logger.set_log_group(LogGroup.AUTOGLUON, f"EXP {self.experiment.experiment_id}")

        if self.experiment.ml_config.num_cpus == "auto":
            self.num_cpus = self.experiment.num_assigned_cpus
        else:
            self.num_cpus = min(self.experiment.num_assigned_cpus, self.experiment.ml_config.num_cpus)

        logger.info(
            f"Resource allocation | CPUs | Available: {self.experiment.num_assigned_cpus} | Requested: {self.experiment.ml_config.num_cpus}"
        )
        logger.info(
            f"""CPU Resources | \
        Available: {self.experiment.num_assigned_cpus} | \
        Requested: {self.experiment.ml_config.num_cpus} | \
        Allocated: {self.num_cpus}"""
        )

    def run_experiment(self):
        """Run experiment."""
        # Ensure AutoGluon uses existing Ray instance if available
        setup_ray_for_external_library()

        if len(self.target_assignments) == 1:
            target = next(iter(self.target_assignments.values()))
        else:
            raise ValueError(f"Single target expected. Got keys: {self.target_assignments.keys()} ")

        # set up model and scoring type
        scoring_type = metrics_inventory_autogluon[self.target_metric]

        # initialization of predictor
        self.model = TabularPredictor(
            label=target,
            eval_metric=scoring_type,
            verbosity=self.experiment.ml_config.verbosity,
            log_to_file=False,  # avoid file logs
        )

        # predictor fit
        self.model.fit(
            self.ag_train_data,
            time_limit=self.experiment.ml_config.time_limit,
            infer_limit=self.experiment.ml_config.infer_limit,
            memory_limit=self.experiment.ml_config.memory_limit,
            presets=self.experiment.ml_config.presets,
            fit_strategy=self.experiment.ml_config.fit_strategy,
            # disabled for preset 'medium_quality'.
            num_bag_folds=self.experiment.ml_config.num_bag_folds,
            included_model_types=self.experiment.ml_config.included_model_types,
        )

        logger.set_log_group(LogGroup.AUTOGLUON, f"EXP {self.experiment.experiment_id}")
        logger.info("Fitting completed")

        # save failure info in case of crashes
        with open(self.path_results.joinpath("debug_info.txt"), "w", encoding="utf-8") as text_file:
            print(self.model.model_failures(), file=text_file)

        # save leaderboard and model information
        self._save_leaderboard_info()

        # save results to experiment
        self.experiment.results["autogluon"] = ModuleResults(
            id="autogluon",
            experiment_id=self.experiment.experiment_id,
            sequence_id=self.experiment.sequence_id,
            model=self._get_sklearn_model(),
            feature_importances=self._get_feature_importances(),
            scores=self._get_scores(),
            selected_features=self.feature_columns,  # no feature selection
            predictions={"test": self._get_predictions()},
        )

        # return updated experiment object
        return self.experiment

    def _get_sklearn_model(self):
        """Get sklearn compatible model."""
        if self.experiment.ml_type == "classification":
            return SklearnClassifier(self.model)
        if self.experiment.ml_type == "regression":
            return SklearnRegressor(self.model)
        else:
            raise ValueError(f"{self.experiment.ml_type} not supported")

    def _get_feature_importances(self):
        """Calculate feature importances."""
        logger.set_log_group(LogGroup.AUTOGLUON, f"EXP {self.experiment.experiment_id}")
        logger.info("Calculating test permutation feature importances...")
        fi = {}
        # set seed
        np.random.seed(42)

        # autogluon permutation feature importances
        fi["autogluon_permutation_test"] = self.model.feature_importance(
            # model- default as best
            data=self.ag_test_data,
            subsample_size=5000,
            time_limit=None,
            include_confidence_band=True,
            confidence_level=0.95,
            num_shuffle_sets=15,
            silent=True,
        )

        # permutation feature importances
        # Calculate group feature importances using feature_groups
        if hasattr(self.experiment, "feature_groups") and self.experiment.feature_groups:
            group_importances = {}
            for group_name, features in self.experiment.feature_groups.items():
                # Calculate feature importance for the current group
                group_importance = self.model.feature_importance(
                    data=self.ag_test_data,
                    features=[(group_name, features)],
                    subsample_size=5000,
                    time_limit=None,
                    include_confidence_band=True,
                    confidence_level=0.95,
                    num_shuffle_sets=15,
                    silent=True,
                )
                # Store the group importance
                group_importances[group_name] = group_importance

            # Create a list to hold combined importances
            combined_feature_importances = [fi["autogluon_permutation_test"]]

            # Append group feature importances
            for group_name, importance in group_importances.items():
                # Create a new row for the group importance
                group_row = importance.copy()
                # Set the index to indicate it's a group
                group_row.index = [f"{group_name}"] * len(group_row)
                combined_feature_importances.append(group_row)

            # Concatenate the list items
            fi["autogluon_permutation_test"] = pd.concat(combined_feature_importances)

        fi["autogluon_permutation_test"] = fi["autogluon_permutation_test"].sort_values(
            by="importance", ascending=False
        )

        # only save permutation feature importances
        combined_importances = {
            "autogluon_permutation": fi["autogluon_permutation_test"].to_dict(orient="index"),
        }

        # shap feature importance - turned off
        if False:
            # SHAP feature importances
            logger.info("Calculating SHAP feature importances...")
            if hasattr(self.experiment, "feature_groups") and self.experiment.feature_groups:
                # Group SHAP feature importances
                fi["octopus_shap_test"] = get_fi_group_shap(
                    experiment={
                        "id": self.experiment.id,
                        "model": self.model,
                        "data_test": self.ag_test_data,
                        "feature_columns": self.feature_columns,
                        "ml_type": self.model.problem_type,
                        "feature_group_dict": self.experiment.feature_groups,
                    },
                    data=None,
                    shap_type="kernel",
                )
            else:
                shap_fi_df, _, _ = get_fi_shap(
                    experiment={
                        "id": self.experiment.id,
                        "model": self.model,
                        "data_test": self.ag_test_data,
                        "feature_columns": self.feature_columns,
                        "ml_type": self.model.problem_type,
                    },
                    data=None,
                    shap_type="kernel",
                )
                fi["octopus_shap_test"] = shap_fi_df

            # Combine all feature importances into a single dictionary
            combined_importances = {
                "autogluon_permutation": fi["autogluon_permutation_test"].to_dict(orient="index"),
                "octopus_shap": fi["octopus_shap_test"].to_dict(orient="records"),
            }

        # print combined feature_importance
        with open(
            self.path_results.joinpath("combined_feature_importances.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(combined_importances, f, indent=4)

        return fi

    def _get_scores(self):
        """Get train/dev/test scores."""
        # Evaluate the model on the test set
        test_performance = self.model.evaluate(self.ag_test_data, detailed_report=True, silent=True)
        test_performance_with_suffix = {f"{key}_test": value for key, value in test_performance.items()}

        # Extract the best model's validation performance
        leaderboard = self.model.leaderboard(silent=True)
        best_model_info = leaderboard.iloc[0].to_dict()
        dev_performance = {key: value for key, value in best_model_info.items() if "val" in key or "score" in key}
        dev_performance_with_suffix = {f"{key}_dev": value for key, value in dev_performance.items()}

        # Evaluate the model on the training set to get training performance
        train_performance = self.model.evaluate(self.ag_train_data, detailed_report=True, silent=True)

        # Modify the keys to add "_train" suffix for training performance
        train_performance_with_suffix = {f"{key}_train": value for key, value in train_performance.items()}

        # test scores calculated by octo method, for comparison
        all_metrics = list(dict.fromkeys([*self.metrics, self.target_metric]))
        test_performance_octo = {}
        for metric in all_metrics:
            performance = get_score_from_model(
                self.model,
                self.ag_test_data,
                self.feature_columns,
                metric,
                self.target_assignments,
                positive_class=self.experiment.configs.study.positive_class,
            )
            test_performance_octo[metric + "_test_octo"] = performance

        # Combine all dictionaries
        combined_performance = {
            **dev_performance_with_suffix,
            **train_performance_with_suffix,
            **test_performance_with_suffix,
            **test_performance_octo,
        }

        # save performance results
        if isinstance(combined_performance, dict):
            for key, value in combined_performance.items():
                if isinstance(value, pd.DataFrame):
                    combined_performance[key] = value.to_dict(orient="records")
        with open(
            self.path_results.joinpath("performance_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(combined_performance, f, indent=4)

        return combined_performance

    def _save_leaderboard_info(self):
        """Save leaderboard information."""
        # save leaderboard
        leaderboard = self.model.leaderboard(
            self.ag_test_data,
            extra_info=True,
            # extra_metrics=
            # display=True
        )
        leaderboard.to_csv(self.path_results.joinpath("leaderboard.csv"))

        # Best test result
        best_model = leaderboard.iloc[0]
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_df.to_csv(self.path_results.joinpath("best_model_result.csv"))
        # print(best_result_df)

        # save model info
        model_info = self.model.info()
        with open(self.path_results.joinpath("model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, default=str, indent=4)

        # show and save model summary
        fit_summary = self.model.fit_summary(
            # show_plot=True
        )
        with open(self.path_results.joinpath("model_stats.txt"), "w", encoding="utf-8") as text_file:
            print(fit_summary, file=text_file)

    def _get_predictions(self):
        """Get validation and test predictions."""
        predictions = {}
        best_model_name = self.model.model_best
        problem_type = self.model.problem_type
        row_column = self.experiment.row_column

        # (A) test predictions
        # DataFrame with 'row_id' from test data
        rowid_test = pd.DataFrame({row_column: self.experiment.data_test[row_column]})

        if problem_type == "regression":
            # Predictions for regression on test data
            test_pred_data = self.model.predict(self.ag_test_data)
            # Create DataFrame with predictions
            test_pred = pd.DataFrame({"prediction": test_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            # Predicted probabilities for classification on test data
            test_pred = self.model.predict_proba(self.ag_test_data)
            # Assign class labels as column names
            class_labels = self.model.class_labels
            test_pred.columns = class_labels
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        # Verify alignment
        assert len(rowid_test) == len(test_pred), "Mismatch in number of test rows!"
        # Combine 'row_id' and test predictions
        predictions["test"] = pd.concat(
            [rowid_test.reset_index(drop=True), test_pred.reset_index(drop=True)],
            axis=1,
        )

        # (B) validation predictions
        # DataFrame with 'row_id' from validation data
        rowid_val = pd.DataFrame({row_column: self.experiment.data_traindev[row_column]})

        if problem_type == "regression":
            # Out-of-fold (OOF) predictions for regression
            oof_pred_data = self.model.predict_oof(model=best_model_name)
            # Create DataFrame with OOF predictions
            oof_pred = pd.DataFrame({"prediction": oof_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            # OOF predicted probabilities for classification
            oof_pred = self.model.predict_proba_oof(model=best_model_name)
            # Assign class labels as column names
            class_labels = self.model.class_labels
            oof_pred.columns = class_labels
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        # Verify alignment
        assert len(rowid_val) == len(oof_pred), "Mismatch in number of validation rows!"
        # Combine 'row_id' and validation predictions
        predictions["val"] = pd.concat(
            [rowid_val.reset_index(drop=True), oof_pred.reset_index(drop=True)],
            axis=1,
        )

        return predictions
