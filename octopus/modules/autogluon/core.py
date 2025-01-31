"""Module: Autogluon Tabular."""

import json
import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators
from autogluon.core.metrics import (
    accuracy,
    balanced_accuracy,
    log_loss,
    mean_absolute_error,
    r2,
    roc_auc,
    root_mean_squared_error,
)
from autogluon.tabular import TabularPredictor

# from sklearn.utils.multiclass import unique_labels
from octopus.experiment import OctoExperiment
from octopus.results import ModuleResults

# TOBEDONE
# - fix scores
# - fix fi
#
# - add more metrics: F1, AUCPR, NEGBRIERSCORE
# - make predictions compatible with Octopus
# - make feature importances compatible with Octopus
# - replace print() with logging
# - do we need to set problem type?
# - add includes_model_types
# - add verbosity setting


# mapping of metrics
try:
    metrics_inventory_autogluon = {
        "AUCROC": roc_auc,
        "ACC": accuracy,
        "ACCBAL": balanced_accuracy,
        "LOGLOSS": log_loss,
        "MAE": mean_absolute_error,
        "MSE": root_mean_squared_error,
        "R2": r2,
    }
except Exception as e:  # pylint: disable=W0718 # noqa: F841
    metrics_inventory_autogluon = {}


@define
class AGCore:
    """Autogluon."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )
    model = field(init=False)
    num_cpus = field(init=False)

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
    def target_assignments(self) -> dict:
        """Target assignments."""
        return self.experiment.target_assignments

    @property
    def target_metric(self) -> str:
        """Target metric."""
        return self.experiment.configs.study.target_metric

    @property
    def config(self) -> dict:
        """Module configuration."""
        return self.experiment.ml_config

    @property
    def feature_columns(self) -> pd.DataFrame:
        """Feature Columns."""
        return self.experiment.feature_columns

    @property
    def train_data(self) -> pd.DataFrame:
        """AG training data."""
        return pd.concat([self.x_traindev, self.y_traindev], axis=1)

    @property
    def test_data(self) -> pd.DataFrame:
        """AG test data."""
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
        print()
        print("Checking resources:")
        print(
            "Number of CPUs available to this experiment:",
            self.experiment.num_assigned_cpus,
        )
        print("Requested number of CPUs:", self.experiment.ml_config.num_cpus)

        if self.experiment.ml_config.num_cpus == "auto":
            self.num_cpus = self.experiment.num_assigned_cpus
        else:
            self.num_cpus = min(
                self.experiment.num_assigned_cpus, self.experiment.ml_config.num_cpus
            )

        print(f"Allocated number of CPUs for this experiment: {self.num_cpus}")
        print()

    def run_experiment(self):
        """Run experiment."""
        if len(self.target_assignments) == 1:
            target = next(iter(self.target_assignments.values()))
        else:
            raise ValueError(
                f"Single target expected. Got keys: {self.target_assignments.keys()} "
            )

        # set up model and scoring type
        scoring_type = metrics_inventory_autogluon[self.target_metric]

        # if self.experiment.ml_type == "classification":
        #     classes = unique_labels(self.y_traindev)
        #     if len(classes) == 1:
        #         raise ValueError("Classifier can't train, only 1 class is present.")
        #     elif len(classes) == 2:
        #         problem_type = "binary"
        #     else:
        #         problem_type = "multiclass"
        # elif self.experiment.ml_type == "regression":
        #     problem_type= 'regression'
        # else:i
        #     raise ValueError(f"{self.experiment.ml_type} not supported")

        # initialization of predictor
        self.model = TabularPredictor(
            label=target,
            # problem_type= problem_type,
            eval_metric=scoring_type,
            verbosity=2,
        )

        # predictor fit
        self.model.fit(
            self.train_data,
            time_limit=self.experiment.ml_config.time_limit,
            presets=self.experiment.ml_config.time_limit,
            num_bag_folds=self.experiment.ml_config.num_bag_folds,
            # add
            # - included_model_typeslist, default = None
            # - verbosity
        )
        print("fitting completed")

        # save failure info in case of crashes
        with open(
            self.path_results.joinpath("debug_info.txt"), "w", encoding="utf-8"
        ) as text_file:
            print(self.model.model_failures(), file=text_file)

        # save leaderboard
        leaderboard = self.model.leaderboard(
            self.test_data,
            extra_info=True,
            # extra_metrics=
            # display= True
            # silent= True ?
        )
        # print(leaderboard)
        leaderboard.to_csv(self.path_results.joinpath("leaderboard.csv"))

        # Best test result
        best_model = leaderboard.iloc[0]  # name of best model
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_df.to_csv(self.path_results.joinpath("best_model_result.csv"))
        # print(best_result_df)

        # save model info
        model_info = self.model.info()
        with open(
            self.path_results.joinpath("model_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_info, f, default=str, indent=4)

        # save feature importances
        feature_importance = self.model.feature_importance(
            # model- default as best
            # features, subsample_size: int = 5000, time_limit: float | None = None
            # num_shuffle_sets: int | None = None, confidence_level: float = 0.99
            self.test_data
        )
        # print(feature_importance)
        feature_importance.to_csv(self.path_results.joinpath("feature_importance.csv"))

        # show and save model summary
        fit_summary = self.model.fit_summary(
            # show_plot=True
        )
        with open(
            self.path_results.joinpath("model_stats.txt"), "w", encoding="utf-8"
        ) as text_file:
            print(fit_summary, file=text_file)

        # predict on test set
        # predict_proba + eval_pred -> evaluate
        # predict ?, predict_multi ?
        # i = 0  # index of model to use
        # model_to_use = self.model.model_names()[i]
        # model_pred = self.model.predict(test_data, model=model_to_use)
        perf = self.model.evaluate(
            self.test_data,
            # auxiliary_metrics=False,
            # display= True,
            detailed_report=True,
        )
        # serialize DataFame
        if isinstance(perf, dict):
            for key in perf:
                if isinstance(perf[key], pd.DataFrame):
                    perf[key] = perf[key].to_dict(orient="records")
        with open(
            self.path_results.joinpath("results_test.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(perf, f, indent=4)

        # Get information about all models
        # all_models = self.model.get_model_names()
        # model_configs = dict()
        # for model in all_models:
        #     specific_model = self.model._trainer.load_model(model)
        #     model_info = specific_model.get_info()
        #     model_configs[model] = model_info
        # print(model_configs_df)
        model_configs = self.model.info()["model_info"]
        # print(model_info)
        with open(
            self.path_results.joinpath("model_configs.txt"), "w", encoding="utf-8"
        ) as f:
            print(model_configs, file=f)

        # save results to experiment
        self.experiment.results["Autogluon"] = ModuleResults(
            id="ag",
            model=self.model.model_best,
            # scores=perf,
            feature_importances=feature_importance.to_dict(),
            scores=self._get_scores(),
            # features_importances [dict]
            # selected_features [dict]
            predictions=self._get_predictions(),
        )

        # return updated experiment object
        return self.experiment

    def _get_scores(self):
        """Get test scores."""

    def _get_predictions(self):
        """Get validation and test predictions."""
        predictions = dict()
        best_model_name = self.model.model_best
        problem_type = self.model.problem_type
        row_column = self.experiment.row_column

        # (A) test predictions
        # DataFrame with 'row_id' from test data
        rowid_test = pd.DataFrame({row_column: self.experiment.data_test[row_column]})

        if problem_type == "regression":
            # Predictions for regression on test data
            test_pred_data = self.model.predict(self.test_data)
            # Create DataFrame with predictions
            test_pred = pd.DataFrame({"prediction": test_pred_data})
        elif problem_type in ["binary", "multiclass"]:
            # Predicted probabilities for classification on test data
            test_pred = self.model.predict_proba(self.test_data)
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
        rowid_val = pd.DataFrame(
            {row_column: self.experiment.data_traindev[row_column]}
        )

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
