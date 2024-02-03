"""Module: Autosklern."""

try:
    import autosklearn.classification
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
# - selection of autosk metric based on octoconfig,
#    then no need to import autosk in workflow
# - saving of results
# - check config regardning available CPUs
# - implement that predictions are done on the reduced features
# - autosklearn refit() functionality
# - autosklearn in version 0.15 requires numpy==1.23.5, otherwise some jobs will fail


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
        return self.experiment.ml_config["config"]

    def __attrs_post_init__(self):
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_results]:
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

        # for debugging, why jobs crashed, etc....
        # print('AutoSk Debug info',self.model.automl_.runhistory_.data)

        # print('Show models:')
        # pprint(self.model.show_models(), indent=4)

        print("Leaderboard:")
        print(self.model.leaderboard())

        print("Statistics:")
        print(self.model.sprint_statistics())

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
        self.experiment.models["model_0"] = self.model

        # return updated experiment object
        return self.experiment

    def predict(self, dataset: pd.DataFrame):
        """Predict on new dataset."""
        model = self.experiment.models["model_0"]
        return model.predict(dataset[self.experiment.feature_columns])

    def predict_proba(self, dataset: pd.DataFrame):
        """Predict_proba on new dataset."""
        if self.experiment.ml_type == "classification":
            self.model = self.experiment.models["model_0"]
            return self.model.predict_proba(dataset[self.experiment.feature_columns])
        else:
            raise ValueError("predict_proba only supported for classifications")
