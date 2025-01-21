"""Module: Autogluon Tabular."""

try:
    from autogluon.core.metrics import (
        accuracy,
        balanced_accuracy,
        log_loss,
        mean_absolute_error,
        r2,
        roc_auc,
        root_mean_squared_error,
    )
except ImportError:
    print("AutoGluon not installed in this conda environment")


import json
import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators
from autogluon.tabular import TabularPredictor

#from sklearn.utils.multiclass import unique_labels
from octopus.experiment import OctoExperiment
from octopus.results import ModuleResults

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
    def data_test(self) -> pd.DataFrame:
        """data_test."""
        return self.experiment.data_test

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

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # create directory if it does not exist
        for directory in [self.path_results]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """Run experiment."""
        # load train data, test data and label
        train_data = pd.concat([self.x_traindev, self.y_traindev], axis=1)
        # train_data = TabularDataset(self.experiment.data_traindev)
        test_data = pd.concat([self.x_test, self.y_test], axis=1)
        # test_data = TabularDataset(self.experiment.data_test)

        if isinstance(self.target_assignments, dict):
            target = next(iter(self.target_assignments.values()))
        elif isinstance(self.target_assignments, list):
            if self.target_assignments:
                target = self.target_assignments[0]
            else:
                raise ValueError("target_assignments list is empty")
        else:
            raise ValueError("Expected target_assignments to be a dictionary or a list")

        # set up model and scoring type
        scoring_type = metrics_inventory_autogluon[
            self.target_metric
        ]

        # if self.experiment.ml_type == "classification":
        #     classes = unique_labels(self.y_traindev)
        #     if len(classes) == 1:
        #         raise ValueError("Classifier can't train when only one class is present.")
        #     elif len(classes) == 2:
        #         problem_type = "binary"
        #     else:
        #         problem_type = "multiclass"
        # elif self.experiment.ml_type == "regression":
        #     problem_type= 'regression'
        # else:
        #     raise ValueError(f"{self.experiment.ml_type} not supported")

        self.model = TabularPredictor(
            label= target,
            #problem_type= problem_type,
            eval_metric= scoring_type,
            verbosity= 2,
        )


        self.model.fit(
            train_data,
            # time_limit= ,
            # presets= 'best_quality', # includes cv
            #[‘best_quality’, ‘high_quality’, ‘good_quality’, ‘medium_quality’ - default,
            # ‘optimize_for_deployment’, ‘interpretable’, ‘ignore_text’]
            # included_model_typeslist, default = None
            # verbosity
            num_bag_folds=5
        )
        print("fitting completed")

        # save failure info in case of crashes
        with open(
            self.path_results.joinpath("debug_info.txt"), "w", encoding="utf-8"
        ) as text_file:
            print(self.model.model_failures(), file=text_file)

        # save leaderboard
        leaderboard = self.model.leaderboard(
            test_data,
            extra_info= True,
            # extra_metrics=
            # display= True
            # silent= True ?
        )
        # print(leaderboard)
        leaderboard.to_csv(self.path_results.joinpath("leaderboard.csv"))

        # save feature importances
        feature_importance = self.model.feature_importance(
            # model- default as best
            # features, subsample_size: int = 5000, time_limit: float | None = None
            # num_shuffle_sets: int | None = None, confidence_level: float = 0.99
            test_data
        )
        # print(feature_importance)
        feature_importance.to_csv(self.path_results.joinpath("feature_importance.csv"))

        # show and save model summary
        fit_summary = self.model.fit_summary(
            #show_plot=True
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
            test_data,
            # auxiliary_metrics=False,
            # display= True,
            detailed_report= True
        )
        # serialize DataFame
        if isinstance(perf, dict):
            for key in perf:
                if isinstance(perf[key], pd.DataFrame):
                    perf[key] = perf[key].to_dict(orient='records')
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
        model_configs = self.model.info()['model_info']
        # print(model_info)
        with open(
            self.path_results.joinpath("model_configs.txt"), "w", encoding="utf-8"
        ) as f:
            print(model_configs, file=f)

        # Best test result
        best_model = leaderboard.iloc[0]
        best_result_df = pd.DataFrame(best_model).transpose()
        best_result_df.to_csv(self.path_results.joinpath("best_model_result.csv"))
        # print(best_result_df)

        # save model info
        model_info = self.model.info()
        with open(
            self.path_results.joinpath("model_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_info, f, default=str, indent=4)

        # save results to experiment
        self.experiment.results["Autogluon"] = ModuleResults(
            id="ag",
            model=self.model.model_best,
            scores=perf,
            feature_importances=feature_importance.to_dict(),
            #results=
        )

        # return updated experiment object
        return self.experiment
