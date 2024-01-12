"""Module: Linear Regression."""
import numpy as np
import optuna
import pandas as pd
from attrs import define, field, validators

from octopus.experiment import OctoExperiment
from octopus.modules.utils import (
    create_innerloop,
    get_score,
    model_linear_regression,
    optuna_direction,
)


@define
class LinearRegressionAve:
    """Linear Regression with one set of hyparamters."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def innerloop(self):
        """Inner split."""
        return create_innerloop(
            df_train_dev=self.experiment.data_traindev,
            feature_columns=self.experiment.feature_columns,
            target_columns=self.experiment.target_assignments.values(),
            group_column=self.experiment.sample_column,
        )

    @property
    def params(self) -> pd.DataFrame:
        """Get parameters from config."""
        return self.experiment.ml_config["config"]

    def _create_optuna_config(self, trial, params):
        """Create config file for optuna optimization.

        missing for linear regression
        # max_iter: Int | None = None,
        # random_state: Int | RandomState | None = None
        """
        config = {}

        for parameter in params:
            if parameter == "alpha":
                config["alpha"] = trial.suggest_float(
                    "alpha", params["alpha"][0], params["alpha"][1]
                )
            elif parameter == "fit_intercept":
                config["fit_intercept"] = trial.suggest_categorical(
                    "fit_intercept", params["fit_intercept"]
                )

            elif parameter == "tol":
                config["tol"] = trial.suggest_float(
                    "tol", params["tol"][0], params["tol"][1]
                )

            elif parameter == "copy_X":
                config["copy_X"] = trial.suggest_categorical("copy_X", params["copy_X"])

            elif parameter == "positive":
                config["positive"] = trial.suggest_categorical(
                    "positive", params["positive"]
                )
            elif parameter == "solver":
                config["solver"] = trial.suggest_categorical("solver", params["solver"])
            elif parameter == "random_state":
                config["random_state"] = trial.suggest_int(
                    "random_state", params["random_state"]
                )
            elif parameter == "optuna_trails":
                pass
            else:
                raise ValueError(
                    f"Parameter {parameter} is not defined for linear regression"
                )

        return config

    def _objective(self, trial):
        """Optuna object to create one set of hpyerparameters for all inner fold."""
        # Create configs for optuna
        config = self._create_optuna_config(trial, self.params)

        # Get model
        model = model_linear_regression(self.experiment.ml_type, config)

        # Create one hyperparameters for each split
        metric_dev = []
        for split in self.innerloop:
            x_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.feature_columns
            ]
            y_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.target_assignments.values()
            ]

            x_dev = self.experiment.data_traindev.loc[
                split["dev"], self.experiment.feature_columns
            ]

            y_dev = self.experiment.data_traindev.loc[
                split["dev"], self.experiment.target_assignments.values()
            ]

            # Train the model
            model.fit(x_train, y_train)

            # Predict on the validation set
            y_pred = model.predict(x_dev)

            # Calculate the mean squared error
            metric_dev.append(
                get_score(self.experiment.config["target_metric"], y_dev, y_pred)
            )

        return np.mean(metric_dev)

    def run_experiment(self):
        """Run experiment."""
        prediction = []

        study = optuna.create_study(
            direction=optuna_direction(self.experiment.config["target_metric"])
        )
        study.optimize(self._objective, n_trials=self.params["optuna_trails"])

        # retrain the model with the average hyperparemters
        for split in self.innerloop:
            # get train and test data
            x_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.feature_columns
            ]
            y_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.target_assignments.values()
            ]
            x_test = self.experiment.data_test[self.experiment.feature_columns]

            # create model and fit
            model = model_linear_regression(self.experiment.ml_type, study.best_params)
            model.fit(x_train, y_train)

            prediction.append(model.predict(x_test))

        # calculate the test score for the mean value of all models values
        y_test = self.experiment.data_test[self.experiment.target_assignments.values()]
        x_test_average = np.mean(np.array(prediction), axis=0)
        score = get_score(
            self.experiment.config["target_metric"], y_test, x_test_average
        )
        print(self.experiment.config["target_metric"], score)

        # save results

        return self.experiment


@define
class LinearRegressionUni:
    """Linear Regression with unique hyperparamters for each split."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )

    @property
    def innerloop(self):
        """Inner split."""
        return create_innerloop(
            df_train_dev=self.experiment.data_traindev,
            feature_columns=self.experiment.feature_columns,
            target_columns=self.experiment.target_assignments.values(),
            group_column=self.experiment.sample_column,
        )

    @property
    def params(self) -> pd.DataFrame:
        """Get parameters from config."""
        return self.experiment.ml_config["config"]

    def _create_optuna_config(self, trial, params):
        """Create config file for optuna optimization.

        missing for linear regression
        # max_iter: Int | None = None,
        # random_state: Int | RandomState | None = None
        """
        config = {}

        for parameter in params:
            if parameter == "alpha":
                config["alpha"] = trial.suggest_float(
                    "alpha", params["alpha"][0], params["alpha"][1]
                )
            elif parameter == "fit_intercept":
                config["fit_intercept"] = trial.suggest_categorical(
                    "fit_intercept", params["fit_intercept"]
                )

            elif parameter == "tol":
                config["tol"] = trial.suggest_float(
                    "tol", params["tol"][0], params["tol"][1]
                )

            elif parameter == "copy_X":
                config["copy_X"] = trial.suggest_categorical("copy_X", params["copy_X"])

            elif parameter == "positive":
                config["positive"] = trial.suggest_categorical(
                    "positive", params["positive"]
                )
            elif parameter == "solver":
                config["solver"] = trial.suggest_categorical("solver", params["solver"])
            elif parameter == "random_state":
                config["random_state"] = trial.suggest_int(
                    "random_state", params["random_state"]
                )
            elif parameter == "optuna_trails":
                pass
            else:
                raise ValueError(
                    f"Parameter {parameter} is not defined for linear regression"
                )

        return config

    def _objective(self, trial, x_train, y_train, x_dev, y_dev):
        """Optuna objective."""
        # create config for optuna
        # need to add all of them
        config = self._create_optuna_config(trial, self.params)

        # Select model
        model = model_linear_regression(self.experiment.ml_type, config)

        # Train the model
        model.fit(x_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(x_dev)

        return get_score(self.experiment.config["target_metric"], y_dev, y_pred)

    def run_experiment(self):
        """Run experiment."""
        prediction = []
        for split in self.innerloop:
            x_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.feature_columns
            ]
            y_train = self.experiment.data_traindev.loc[
                split["train"], self.experiment.target_assignments.values()
            ]
            x_dev = self.experiment.data_traindev.loc[
                split["dev"], self.experiment.feature_columns
            ]
            y_dev = self.experiment.data_traindev.loc[
                split["dev"], self.experiment.target_assignments.values()
            ]
            x_test = self.experiment.data_test[self.experiment.feature_columns]

            objective = lambda trial: self._objective(  # noqa E731
                trial, x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev
            )
            study = optuna.create_study(
                direction=optuna_direction(self.experiment.config["target_metric"])
            )
            study.optimize(objective, n_trials=self.params["optuna_trails"])

            # create model and fit
            model = model_linear_regression(self.experiment.ml_type, study.best_params)
            model.fit(x_train, y_train)

            prediction.append(model.predict(x_test))

        # calculate the test score for the mean value of all models values
        y_test = self.experiment.data_test[self.experiment.target_assignments.values()]
        average_prediction = np.mean(np.array(prediction), axis=0)
        score = get_score(
            self.experiment.config["target_metric"], y_test, average_prediction
        )
        print(self.experiment.config["target_metric"], score)

        # save results

        return self.experiment
