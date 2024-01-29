"""OctoFull Module."""
import concurrent.futures
import pickle
import shutil
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

from octopus.experiment import OctoExperiment
from octopus.models.config import model_inventory
from octopus.models.parameters import parameters_inventory
from octopus.models.utils import create_trialparams_from_config
from octopus.modules.utils import optuna_direction

# from sklearn.inspection import permutation_importance
from octopus.utils import DataSplit

# TOBEDONE BASE
# - fix experiment.num_assigned_cpus -- consider ("ml_only_first": True) status
#   copy from moduleAW

# TOBEDONE OCTOFULL
# - (4) show best results with performance metrics after optuna completion
# - default: num_workers set to k_inner as default, warning if num_workers != k_inner
# - better study name for global studies - problem with large k_outer -> "0-89"
# - check_resources: consider real n_jobs parameter
# - functionality to overwrite single defaults in model default parameter config
# - module are big and should be directories
# - create final bags to collect result in the two streams
# - xgoost class weights need to be set in training! How to solve that?
# - validate input parameters: dim_reduction_methods, ml_model_types
# - implement survival model


# TOBEDONE OPTUNA
# - check "Exception occurred with execution task", global hp optimization
# - Enqueue trials - how to implement that?
# - Bring optuna parallelization to the next level (process based!):
#   (a) multiple parallel optuna optimization instances per split in
#       run_individualhp_optimization for individual split HP optimizations
#   (b) multiple parallel executions of self.run_individualhp_optimization()
#       to parallelize the optuna study with global HPs


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
    # model = field(init=False)
    data_splits = field(default=dict(), validator=[validators.instance_of(dict)])

    @property
    def path_module(self) -> Path:
        """Module path."""
        return self.experiment.path_study.joinpath(self.experiment.path_sequence_item)

    @property
    def path_optuna(self) -> Path:
        """Optuna db path."""
        return self.path_module.joinpath("optuna")

    @property
    def path_trials(self) -> Path:
        """Trials path."""
        return self.path_module.joinpath("trials")

    @property
    def hpo_type(self) -> str:
        """Trials path."""
        return self.experiment.ml_config["config"]["HPO_type"]

    def __attrs_post_init__(self):
        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seed=self.experiment.ml_config["config"]["datasplit_seed_inner"],
            num_folds=self.experiment.ml_config["config"]["k_inner"],
            stratification_col=self.experiment.stratification_column,
        ).get_datasplits()
        # delete directories /trials /optuna to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_optuna, self.path_trials]:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        # check if there is a mismatch between configured resources
        # and resources assigned to the experiment
        self.check_resources()

    def check_resources(self):
        """Check resources, assigned vs requested."""
        print()
        print("Checking resources:")
        print(
            "Number of CPUs available to this experiment:",
            self.experiment.num_assigned_cpus,
        )
        exec_type = self.experiment.ml_config["config"]["execution_type"]
        num_workers = self.experiment.ml_config["config"]["num_workers"]

        # assuming n_jobs=1 for every model
        if exec_type == "parallel":
            num_requested_cpus = num_workers  # n_jobs=1
        else:
            num_requested_cpus = 1  # n_jobs=1

        print(
            f"Number of requested CPUs for this experiment: {num_requested_cpus}"
            f" (assuming n_jobs=1 for every model)."
        )
        print()

    def run_experiment(self):
        """Run experiment."""
        if self.hpo_type == "global":
            self.run_globalhp_optimization()
        elif self.hpo_type == "individual":
            self.run_individualhp_optimization()
        else:
            raise ValueError(f"HPO type: {self.hpo_type} not supported")

        return self.experiment

    def run_globalhp_optimization(self):
        """Optimization run with a global HP set over all inner folds."""
        print("Running Optuna Optimization with a global HP set")

        # run Optuna study with a global HP set
        self.optimize_splits(self.data_splits)

    def run_individualhp_optimization(self):
        """Optimization runs with an individual HP set for each inner datasplit."""
        print("Running Optuna Optimizations for each inner datasplit separately")

        # covert to list of dicts for compatibility with OptimizeOptuna
        splits = [{key: value} for key, value in self.data_splits.items()]

        # For the parallelization of Optuna tasks, we have several options:
        # (a) n_jobs parameter in Optuna, however this is a threaded operation.
        #     In this case, the splits are executed sequentially but each split
        #     uses several optuna instances
        # (b) we parallelize the Optuna execution per split, as shown below.
        # (c) In addition to (b) we could start multiple optuna optimizations
        #     per split and so achieve an even faster execution of trials.
        #     One could also only do (c) with increased parallelization to
        #     achieve the same effect as (b)+(c).

        # same config parameters also used for parallelization of bag trainings
        optuna_execution = self.experiment.ml_config["config"]["execution_type"]
        max_workers = self.experiment.ml_config["config"]["num_workers"]

        if optuna_execution == "sequential":
            print("Sequential execution of Optuna optimizations for individual HPs")
            for split in splits:
                self.optimize_splits(split)
                print(f"Optimization of split:{split.keys()} completed")
        elif optuna_execution == "parallel":
            print("Parallel execution of Optuna optimizations for individual HPs")
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = []
                for split in splits:
                    try:
                        future = executor.submit(self.optimize_splits, split)
                        futures.append(future)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while submitting task: {e}")
                for future in concurrent.futures.as_completed(futures):
                    try:
                        _ = future.result()
                        print("Optimization of single split completed")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while executing task: {e}")
        else:
            raise ValueError("Execution type not supported")

    def optimize_splits(self, splits):
        """Optimize splits.

        Works if splits contain several splits as well as
        when splits only contains a single split
        """
        # define study name by joined keys of splits
        study_name = "_".join([str(key) for key in splits.keys()])
        # set up Optuna study
        objective = ObjectiveOptuna(
            experiment=self.experiment, data_splits=splits, study_name=study_name
        )

        # multivariate sampler with group option
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True)

        # create study with unique name and database
        db_path = self.path_optuna.joinpath(study_name + ".db")
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
        study = optuna.create_study(
            study_name=study_name,
            direction=optuna_direction(self.experiment.config["target_metric"]),
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        study.optimize(
            objective,
            n_jobs=1,
            n_trials=self.experiment.ml_config["config"]["HPO_trials"],
        )

        print()
        best_parameters = study.best_params
        print("best parameters:", best_parameters)
        print("Experiment completed")

    def predict(self, dataset: pd.DataFrame):
        """Predict on new dataset."""
        # this is old and not working code
        # model = self.experiment.models["model_0"]
        # return model.predict(dataset[self.experiment.feature_columns])

    def predict_proba(self, dataset: pd.DataFrame):
        """Predict_proba on new dataset."""
        # this is old and not working code
        # if self.experiment.ml_type == "classification":
        #    self.model = self.experiment.models["model_0"]
        #    return self.model.predict_proba(dataset[self.experiment.feature_columns])
        # else:
        #    raise ValueError("predict_proba only supported for classifications")


class ObjectiveOptuna:
    """Callable optuna objective.

    A single solution for global and individual HP optimizations.
    """

    def __init__(
        self,
        experiment,
        data_splits,
        study_name,
    ):
        self.experiment = experiment
        self.data_splits = data_splits
        self.study_name = study_name
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
        (c) global parameters that have to be translated
            in fixed model specific parameters
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

        # get model parameters
        optuna_model_settings = None  # use default
        settings_default = parameters_inventory[ml_model_type]["default"]

        if optuna_model_settings is None:
            # use default model parameter settings
            model_params = create_trialparams_from_config(trial, settings_default)
        else:
            # use model parameter settings as provided by config
            model_params = create_trialparams_from_config(trial, optuna_model_settings)

        # overwrite model parameters specified by global settings
        fixed_global_parameters = {
            "ml_jobs": self.ml_jobs,
            "ml_seed": self.ml_seed,
        }
        translate = parameters_inventory[ml_model_type]["translate"]
        for key, value in fixed_global_parameters.items():
            model_params[translate[key]] = value

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
        # create bag with all provided trainings
        bag_trainings = TrainingsBag(
            trainings=trainings,
            execution_type=self.execution_type,
            num_workers=self.num_workers,
            target_metric=self.experiment.config["target_metric"],
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in bag
        bag_trainings.run_trainings()

        # save bag
        path_save = self.experiment.path_study.joinpath(
            self.experiment.path_sequence_item,
            "trials",
            f"study{self.study_name}trial{trial.number}_bag.pkl",
        )
        bag_trainings.to_pickle(path_save)

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
    # same config parameters (execution type, num_workers) also used for
    # parallelization of optuna optimizations of individual inner loop trainings
    execution_type: str = field(validator=[validators.instance_of(str)])
    num_workers: int = field(validator=[validators.instance_of(int)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    # path: Path = field(default=Path(), validator=[validators.instance_of(Path)])
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
                futures = []
                for i in self.trainings:
                    try:
                        future = executor.submit(i.run_training())
                        futures.append(future)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while submitting task: {e}")
                for future in concurrent.futures.as_completed(futures):
                    try:
                        _ = future.result()
                        print("Inner parallel training completed")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while executing task: {e}")
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
        """Save Bag using pickle."""
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path):
        """Load Bag from pickle file."""
        with open(path, "rb") as file:
            return pickle.load(file)
