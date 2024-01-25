"""OctoFull Module."""
import concurrent.futures
import time
from pathlib import Path
from statistics import mean

import optuna
import pandas as pd
from attrs import define, field, validators
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# from sklearn.inspection import permutation_importance
from octopus.datasplit import DataSplit
from octopus.experiment import OctoExperiment
from octopus.models.config import model_inventory, parameters_inventory
from octopus.modules.utils import optuna_direction

# TOBEDONE OCTOFULL
# - folder structure!!!
# - default values for octofull
# - show best results with performance metrics after optuna completion
# - xgoost class weights need to be set in training! How to solve that?
# - validate input parameters: dim_reduction_methods, ml_model_types
# - develop MultiTraining
# - parallelization (HPO)
# - saving of trainings

# FOLDER STRUCTURE -- just create it parallel to existing structure
# Experiment_0/Sequenc_0/ with
# - experiment.pkl
# - Trials/
# - Optuna/


# TOBEDONE OPTUNA
# - save in db
# - define study name


# TOBEDONE TRAINING
# - all multi target models are done separately, shap and permutation
#   importance may not work anyways
# - include shapley and permutation importance
# - include dimensionality reduction
# - include outlier elimination


@define
class OctoFull:
    """OctoFull."""

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )
    model = field(init=False)
    data_splits = field(default=dict(), validator=[validators.instance_of(dict)])

    def __attrs_post_init__(self):
        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seed=self.experiment.ml_config["config"]["datasplit_seed_inner"],
            num_folds=self.experiment.ml_config["config"]["k_inner"],
            stratification_col=self.experiment.stratification_column,
        ).get_datasplits()

    def run_experiment(self):
        """Run experiment."""
        # self.run_globalhp_optimization()
        self.run_individualhp_optimization()

        return self.experiment

    def run_globalhp_optimization(self):
        """Optimization run with a global HP set over all inner folds."""
        print("Running Optuna Optimization with a global HP set")

        # run Optuna study with a global HP set
        self.optimize_splits(self.data_splits)

    def run_individualhp_optimization(self):
        """Optimization runs with an individual HP set for each inner datasplit."""
        print("Running Optuna Optimizations for each inner datasplit separately")

        def wait_func():
            time.sleep(10)
            print("10s waited")

        # covert to list of dicts for compatibility with OptimizeOptuna
        splits = [{key: value} for key, value in self.data_splits.items()]

        # For the parallelization of Optuna tasks, we have several options:
        # (a) n_jobs parameter in Optuna, however this is a threaded operation.
        #     In this case, the splits are executed sequentially but each split
        #     uses several optuna instances
        # (b) we parallelize the Optuna execution per split
        optuna_execution = "parallel"
        max_workers = 5

        if optuna_execution == "sequential":
            for split in splits:
                # self.optimize_splits(split)
                wait_func()
                print("Optimization of single split completed")
        elif optuna_execution == "parallel":
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                futures = [
                    # executor.submit(self.optimize_splits(split)) for split in splits
                    executor.submit(wait_func())
                    for split in splits
                ]
                for _ in concurrent.futures.as_completed(futures):
                    print("Optimization of single split completed")
        else:
            raise ValueError("Execution type not supported")

    def optimize_splits(self, splits):
        """Optimize splits.

        Works if splits contain several splits as well as
        when splits only contains a single split
        """
        # set up Optuna study
        objective = ObjectiveOptuna(experiment=self.experiment, data_splits=splits)

        sampler = optuna.samplers.TPESampler(
            multivariate=True, group=True
        )  # multivariate
        study = optuna.create_study(
            study_name="test",
            direction=optuna_direction(self.experiment.config["target_metric"]),
            sampler=sampler,
            # storage="sqlite:///example.db",
        )

        study.optimize(
            objective,
            n_jobs=1,
            n_trials=self.experiment.ml_config["config"]["HPO_trials"],
        )
        # optuna.study.get_all_study_summaries(storage="sqlite:///example.db")
        # best_parameters = len(innersplit) * [study.best_params]
        print()
        best_parameters = study.best_params
        print("best parameters:", best_parameters)
        print("Experiment completed")

    def predict(self, dataset: pd.DataFrame):
        """Predict on new dataset."""
        # this is old and not working code
        model = self.experiment.models["model_0"]
        return model.predict(dataset[self.experiment.feature_columns])

    def predict_proba(self, dataset: pd.DataFrame):
        """Predict_proba on new dataset."""
        # this is old and not working code
        if self.experiment.ml_type == "classification":
            self.model = self.experiment.models["model_0"]
            return self.model.predict_proba(dataset[self.experiment.feature_columns])
        else:
            raise ValueError("predict_proba only supported for classifications")


class ObjectiveOptuna:
    """Callable optuna objective for a single HP set (unique)."""

    def __init__(
        self,
        experiment,
        data_splits,
    ):
        self.experiment = experiment
        self.data_splits = data_splits
        # parameters potentially used for optimizations
        self.ml_model_types = self.experiment.ml_config["config"]["ml_model_types"]
        self.dim_red_methods = self.experiment.ml_config["config"]["dim_red_methods"]
        self.max_outl = self.experiment.ml_config["config"]["max_outl"]
        # fixed parameters
        self.ml_seed = self.experiment.ml_config["config"]["ml_seed"]
        self.ml_jobs = self.experiment.ml_config["config"]["ml_jobs"]
        # training parameters
        self.execution_type = self.experiment.ml_config["config"]["execution_type"]
        self.num_workers = self.experiment.ml_config["config"]["num_workers"]

    def __call__(self, trial):
        """Call.

        We have different types of parameters:
        (a) non-model parameters that are needed in
            the training
        (b) model parameters that are varied by optuna
            (defined by default or optuna_model_settings)
        (c) model parameters that are kept constant
            during the training and are non-default -
            these should go into (b) using 'fixed' dtype.

        """
        # get non-model parameters
        # (1) dimension reduction
        if len(self.dim_red_methods) > 1:
            dim_reduction = trial.suggest_categorical(
                name="dim_red_method", choices=self.dim_red_methods
            )
        else:
            dim_reduction = self.dim_red_methods[0]

        # (2) ml_model_type
        if len(self.ml_model_types) > 1:
            ml_model_type = trial.suggest_categorical(
                name="ml_model_type", choices=self.ml_model_types
            )
        else:
            ml_model_type = self.ml_model_types[0]

        # (3) number of outliers to be detected
        if self.max_outl > 0:
            num_outl = trial.suggest_int(name="num_outl", low=0, high=self.max_outl)
        else:
            num_outl = 0

        #  fixed parameters
        model_params_fixed = {
            "ml_jobs": self.ml_jobs,
            "ml_seed": self.ml_seed,
        }

        # overwrite optuna HP settings
        optuna_model_settings = None  # use default

        # get model parameters
        model_params = parameters_inventory[ml_model_type](
            trial, model_params_fixed, optuna_model_settings
        )

        # create trainings
        row_column = self.experiment.row_column
        trainings = list()
        for key, split in self.data_splits.items():
            trainings.append(
                Training(
                    training_id=self.experiment.id + "_" + str(key),
                    ml_type=self.experiment.ml_type,
                    target_assignments=self.experiment.target_assignments,
                    feature_columns=self.experiment.feature_columns,
                    row_column=row_column,
                    data_train=split["train"],  # inner datasplit, train
                    data_dev=split["test"],  # inner datasplit, dev
                    data_test=self.experiment.data_test,
                    dim_reduction=dim_reduction,
                    outl_reduction=num_outl,
                    ml_model_type=ml_model_type,
                    ml_model_params=model_params,
                    target_metric=self.experiment.config["target_metric"],
                )
            )

        bag_trainings = TrainingsBag(
            trainings=trainings,
            execution_type=self.execution_type,
            num_workers=self.num_workers,
            target_metric=self.experiment.config["target_metric"],
            row_column=self.experiment.row_column,
            # path?
        )

        bag_trainings.run_trainings()

        # evaluate trainings using target metric
        scores = bag_trainings.get_scores()

        # add scores info to the optuna trial
        for key, value in scores.items():
            trial.set_user_attr(key, value)

        # print scores info to console
        print(f"Trial scores for metric: {self.experiment.config['target_metric']}")
        for key, value in scores.items():
            if isinstance(value, list):
                print(f"{key}:{value}")
            else:
                print(f"{key}:{value:.3f}")

        return scores["dev_avg"]  # dev target metric


@define
class Training:
    """Model Training Class."""

    training_id: str = field(validator=[validators.instance_of(str)])
    ml_type: str = field(validator=[validators.instance_of(str)])
    target_assignments: dict = field(validator=[validators.instance_of(dict)])
    feature_columns: list = field(validator=[validators.instance_of(list)])
    row_column: str = field(validator=[validators.instance_of(str)])
    data_train: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_dev: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    data_test: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    dim_reduction: str = field(validator=[validators.instance_of(str)])
    outl_reduction: int = field(validator=[validators.instance_of(int)])
    ml_model_type: str = field(validator=[validators.instance_of(str)])
    ml_model_params: dict = field(validator=[validators.instance_of(dict)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    model = field(init=False)
    predictions: dict = field(default=dict(), validator=[validators.instance_of(dict)])

    @property
    def x_train(self):
        """x_train."""
        return self.data_train[self.feature_columns]

    @property
    def x_dev(self):
        """x_dev."""
        return self.data_dev[self.feature_columns]

    @property
    def x_test(self):
        """x_test."""
        return self.data_test[self.feature_columns]

    @property
    def y_train(self):
        """y_train."""
        return self.data_train[self.target_assignments.values()]

    @property
    def y_dev(self):
        """y_dev."""
        return self.data_dev[self.target_assignments.values()]

    @property
    def y_test(self):
        """y_dev."""
        return self.data_test[self.target_assignments.values()]

    # perform:
    # (1) dim_reduction
    # (2) outlier removal
    # (3) training
    # (4) standard feature importance
    # (4) permutation feature importance
    # (5) shapley feature importance

    # output:
    # (1) predictions
    # (2) probabilities in case of classification
    # (3) feature_importances, which
    # (4)

    def __attrs_post_init__(self):
        # reset index
        pass

    def run_training(self):
        """Run trainings."""
        # missing: dim reduction
        # missing: outlier removal
        self.model = model_inventory[self.ml_model_type](**self.ml_model_params)

        if len(self.target_assignments) == 1:
            # standard sklearn single target models
            self.model.fit(
                self.x_train,
                self.y_train.squeeze(axis=1),
            )
        else:
            # multi target models, incl. time2event
            self.model.fit(self.x_train, self.y_train)

        # missing: include row_id
        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = self.data_train[self.row_column]
        self.predictions["train"]["target"] = self.y_train.squeeze(axis=1)
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train)

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_column] = self.data_dev[self.row_column]
        self.predictions["dev"]["target"] = self.y_dev.squeeze(axis=1)
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev)

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_column] = self.data_test[self.row_column]
        self.predictions["test"]["target"] = self.y_test.squeeze(axis=1)
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test)

        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test)

        # missing: other feature reduction methods
        # result = permutation_importance(
        #    self.model, X=self.x_dev, y=self.y_dev, n_repeats=10, random_state=0
        # )
        # print(result)

    def to_pickle(self, path):
        """Save training."""

    @classmethod
    def from_pickle(cls, path):
        """Load training."""


@define
class TrainingsBag:
    """Container for Trainings.

    Supports:
    - execution of trainings, sequential/parallel
    - saving/loading
    """

    trainings: list = field(validator=[validators.instance_of(list)])
    execution_type: str = field(validator=[validators.instance_of(str)])
    num_workers: int = field(validator=[validators.instance_of(int)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    path: Path = field(default=Path(), validator=[validators.instance_of(Path)])
    train_status: bool = field(default=False)

    def run_trainings(self):
        """Run all available trainings."""
        if self.execution_type == "sequential":
            for training in self.trainings:
                training.run_training()
                print("Inner sequential training completed")
        elif self.execution_type == "parallel":
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
            ) as executor:
                futures = [executor.submit(i.run_training()) for i in self.trainings]
                for _ in concurrent.futures.as_completed(futures):
                    print("Inner parallel training completed")
        else:
            raise ValueError("Execution type not supported")
        self.train_status = True

    def get_scores(self):
        """Get scores."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.run_trainings()

        scores = dict()
        metrics_inventory = {
            "AUCROC": roc_auc_score,
            "ACC": accuracy_score,
            "ACCBAL": balanced_accuracy_score,
            "LOGLOSS": log_loss,
            "MAE": mean_absolute_error,
            "MSE": mean_squared_error,
            "R2": r2_score,
        }

        storage = {key: [] for key in ["train", "dev", "test"]}
        pool = {key: [] for key in ["train", "dev", "test"]}

        for training in self.trainings:
            # averaging
            if self.target_metric in ["AUCROC", "LOGLOSS"]:
                for part in storage.keys():
                    probabilities = training.predictions[part][1]  # binary only!!
                    target = training.predictions[part]["target"]
                    storage[part].append(
                        metrics_inventory[self.target_metric](target, probabilities)
                    )
            else:
                for part in storage.keys():
                    predictions = training.predictions[part]["prediction"]
                    target = training.predictions[part]["target"]
                    storage[part].append(
                        metrics_inventory[self.target_metric](target, predictions)
                    )
            # pooling
            for part in pool.keys():
                pool[part].append(training.predictions[part])

        # calculate averaging scores
        scores["train_avg"] = mean(storage["train"])
        scores["train_lst"] = storage["train"]
        scores["dev_avg"] = mean(storage["dev"])
        scores["dev_lst"] = storage["dev"]
        scores["test_avg"] = mean(storage["test"])
        scores["test_lst"] = storage["test"]
        # stack pooled data and groupby
        for part in pool.keys():
            pool[part] = pd.concat(pool[part], axis=0)
            pool[part] = pool[part].groupby(by=self.row_column).mean()
        # calculate pooling scores (soft and hard)
        if self.target_metric in ["AUCROC", "LOGLOSS"]:
            for part in pool.keys():
                probabilities = pool[part][1]  # binary only!!
                predictions = pool[part]["prediction"]
                target = pool[part]["target"]
                scores[part + "_pool_soft"] = metrics_inventory[self.target_metric](
                    target, probabilities
                )
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )
        else:
            for part in pool.keys():
                predictions = pool[part]["prediction"]
                target = pool[part]["target"]
                scores[part + "_pool_hard"] = metrics_inventory[self.target_metric](
                    target, predictions
                )

        return scores

    def to_pickle(self, path):
        """Save Bag."""

    @classmethod
    def from_pickle(cls, path):
        """Load Bag."""
