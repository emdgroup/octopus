"""Module: Autosklern."""

try:
    import autosklearn.classification
    import autosklearn.metrics
    import autosklearn.regression
except ImportError:
    print("Auto-Sklearn not installed in this conda environment")

import json
import shutil
from pathlib import Path

import pandas as pd
from attrs import define, field, validators
from sklearn.metrics import mean_absolute_error

from octopus.experiment import OctoExperiment

# TOBEDONE:
# - (1) calculate scores and save to experiment (see octofull)
# - (2) function to calculate feature importances (standard, permutation, shapley)
#       https://automl.github.io/auto-sklearn/master/examples/
#       40_advanced/example_inspect_predictions.html
# - (3) save feature importances in experiment
# - (4) save selected features (needs features importances)
# - (5) check config regarding available CPUs
# - (6) use defined properties in predict function
# - (7) autosklearn refit() functionality
# - (8) check what other functionality from autosk is missing
# - understand autosk cost function
#   https://github.com/automl/auto-sklearn/issues/1717
# - turn off data preprocessing:
#   https://automl.github.io/auto-sklearn/development/examples/
#   80_extending/example_extending_data_preprocessor.html
#   #sphx-glr-examples-80-extending-example-extending-data-preprocessor-py
# - openblas issue: export OPENBLAS_NUM_THREADS=1 (set in terminal)
#   check: https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
#   try to set in the code, maybe in manager
#   https://superfastpython.com/numpy-number-blas-threads/
#   #Need_to_Configure_the_Number_of_Threads_Used_By_BLAS
#   it needs to be set before autosk and even pandas is imported
# - how to run vanilla autosklearn:
#   ensemble_class=autosklearn.ensembles.SingleBest
#   initial_configurations_via_metalearning=0
#   https://automl.github.io/auto-sklearn/master/faq.html

# Notes:
# - autosklearn in version 0.15 requires numpy==1.23.5, otherwise some jobs will fail

# mapping of metrics
metrics_inventory = {
    "AUCROC": autosklearn.metrics.roc_auc,
    "ACC": autosklearn.metrics.accuracy,
    "ACCBAL": autosklearn.metrics.balanced_accuracy,
    "LOGLOSS": autosklearn.metrics.log_loss,
    "MAE": autosklearn.metrics.mean_absolute_error,
    "MSE": autosklearn.metrics.root_mean_squared_error,
    "R2": autosklearn.metrics.r2,
}


@define
class Autosklearn:
    """Autosklearn."""

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
    def path_tmp(self) -> Path:
        """Results path."""
        return self.path_module.joinpath("tmp")

    @property
    def x_train(self) -> pd.DataFrame:
        """x_train."""
        return self.experiment.data_traindev[self.experiment.feature_columns]

    @property
    def y_train(self) -> pd.DataFrame:
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
    def params(self) -> pd.DataFrame:
        """Auto-sklearn parameters."""
        params = self.experiment.ml_config["config"]
        # add metric based on target metric
        params["metric"] = metrics_inventory[self.experiment.config["target_metric"]]
        # overwrite tmp folder, makes temp space usage visible
        params["tmp_folder"] = (str(self.path_tmp),)
        return params

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_results, self.path_tmp]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """Run experiment."""
        # overwrite tmp directory
        # self.params['tmp_folder']=......

        if self.experiment.ml_type == "classification":
            self.model = autosklearn.classification.AutoSklearnClassifier(**self.params)
        elif self.experiment.ml_type == "regression":
            self.model = autosklearn.regression.AutoSklearnRegressor(**self.params)
        else:
            raise ValueError(
                f"Autosklearn ml_type {self.experiment.ml_type} not supported"
            )

        self.model.fit(
            self.x_train,
            self.y_train,
            dataset_name=f"Octopus experiment:{self.experiment.id}",
        )
        print("fitting completed")

        # save debug info - interesting in case of crashes
        with open(
            self.path_results.joinpath("debug_info.txt"), "w", encoding="utf-8"
        ) as text_file:
            print(self.model.automl_.runhistory_.data, file=text_file)

        # save leaderboard
        leaderboard = self.model.leaderboard(
            detailed=True,
        )
        leaderboard.to_csv(self.path_results.joinpath("leaderboard.csv"))

        # save detailed model info with all parameters
        models = self.model.show_models()
        model_configs = dict()
        for key in models.keys():
            sel_model = models[key]
            model_id = sel_model["model_id"]
            model_dict = dict(self.model.automl_.runhistory_.ids_config[model_id])
            model_dict["model_id"] = model_id
            model_dict["rank"] = sel_model["rank"]
            model_dict["ensemble_weight"] = sel_model["ensemble_weight"]
            model_configs[str(key)] = model_dict
        with open(
            self.path_results.joinpath("model_configs.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(model_configs, f, default=int)

        # show and save model statistics
        print("Statistics:")
        statistics = self.model.sprint_statistics()
        print(statistics)
        with open(
            self.path_results.joinpath("model_stats.txt"), "w", encoding="utf-8"
        ) as text_file:
            print(statistics, file=text_file)

        print("Best result:")
        results_df = pd.DataFrame(self.model.cv_results_)
        # display(results_df)
        best_validation_result_df = results_df[results_df["rank_test_scores"] == 1]
        # display(best_validation_result_df)
        self.experiment.results["best_validation_result_df"] = best_validation_result_df

        # predict on test set
        preds = self.model.predict(self.x_test)
        test_predictions_df = pd.DataFrame()
        test_predictions_df[self.experiment.row_column] = self.experiment.data_test[
            self.experiment.row_column
        ]
        test_predictions_df["prediction"] = preds
        test_predictions_df["target"] = self.y_test

        if self.experiment.ml_type == "classification":
            probs_df = pd.DataFrame(self.model.predict_proba(self.x_test))
            probs_df.columns = ["prob_" + str(x) for x in probs_df.columns]
            test_predictions_df = pd.concat([test_predictions_df, probs_df], axis=1)
        self.experiment.results["test_predictions_df"] = test_predictions_df

        # save scores to experiment (missing)
        # show and save test results, MAE
        print(
            f"Experiment: {self.experiment.id} "
            f"Test MAE: {mean_absolute_error(self.y_test, preds)}"
        )
        results_test = {
            "experiment_id": self.experiment.id,
            "test MAE": mean_absolute_error(self.y_test, preds),
            # "test predictions": preds, #json error
        }
        with open(
            self.path_results.joinpath("results_test.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results_test, f)

        # save model
        self.experiment.models["best"] = self.model

        # return updated experiment object
        return self.experiment

    def predict(self, dataset: pd.DataFrame):
        """Predict on new dataset."""
        model = self.experiment.models["best"]
        return model.predict(dataset[self.experiment.feature_columns])

    def predict_proba(self, dataset: pd.DataFrame):
        """Predict_proba on new dataset."""
        if self.experiment.ml_type == "classification":
            self.model = self.experiment.models["best"]
            return self.model.predict_proba(dataset[self.experiment.feature_columns])
        else:
            raise ValueError("predict_proba only supported for classifications")
