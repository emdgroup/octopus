"""OctoFull Module."""

import concurrent.futures
import json
import pickle
import shutil
import warnings
from pathlib import Path
from statistics import mean
from typing import List

import optuna
import pandas as pd
from attrs import define, field, validators
from optuna.samplers._tpe.sampler import ExperimentalWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from octopus.experiment import OctoExperiment
from octopus.models.config import model_inventory
from octopus.models.parameters import parameters_inventory
from octopus.models.utils import create_trialparams_from_config
from octopus.modules.utils import optuna_direction

# from sklearn.inspection import permutation_importance
from octopus.utils import DataSplit

# ignore three Optuna experimental warnings
# !may be specific to optuna version due to line number
for line in [319, 330, 338]:
    warnings.filterwarnings(
        "ignore",
        category=ExperimentalWarning,
        module="optuna.samplers._tpe.sampler",
        lineno=line,
    )
# TOBEDONE BASE
# - check module type
# - any issues due to missing .copy() statements???
# - autosk with serial processing of outer folds does not use significant CPU??
# - check that openblas settings are correct and suggest solutions

# TOBEDONE OCTOFULL
# - (1) make bag compatible with sklearn
#   +very difficult as sklearn differentiates between regression, classification
#   RegressionBag, ClassBag
#   +we also want to include T2E
# - (2) bag feature importances (standard, permutation, shapley)
# - (3) training feature importances (standard, permutation, shapley)
# - (4) return selected features
# - (5) basic analytics class
# - (6) implement survival model
# - (7) Make use of default model parameters, see autosk, optuna
# - (8) octofull module is big and should be directory
# - Performance evaluation generalize: ensemble_hard, ensemble_soft

# - automatically remove features with a single value! and provide user feedback
# - deepchecks - https://docs.deepchecks.com/0.18/tabular/auto_checks/data_integrity/index.html
# - outer parallelizaion can lead to very differing execution times per experiment!
# - check that for classification only classification modules are used
# - sequence config -- module is fixed
# - attach results (best_bag) to experiment
# - improve create_best_bags - use a direct way, from returned best trial or optuna.db
# - xgoost class weights need to be set in training! How to solve that?
# - check disk space and inform about disk space requirements


# TOBEDONE OPTUNA
# - Enqueue trials - how to implement that?
# - Bring optuna parallelization to the next level (process based!):
#   (a) multiple parallel optuna optimization instances per split in
#       run_individualhp_optimization for individual split HP optimizations
#   (b) multiple parallel executions of self.run_individualhp_optimization()
#       to parallelize the optuna study with global HPs
# - pruning Trials (parallel execution,check first for pruning)


# TOBEDONE TRAINING
# - variance selection threshold could be a HP, scaling before
# - all multi target models are done separately, shap and permutation
#   importance may not work anyways
# - add outlier removal
# - add scaling, as a HP?
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
    def path_results(self) -> Path:
        """Results path."""
        return self.path_module.joinpath("results")

    def __attrs_post_init__(self):
        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seed=self.experiment.ml_config["datasplit_seed_inner"],
            num_folds=self.experiment.ml_config["n_folds_inner"],
            stratification_col=self.experiment.stratification_column,
        ).get_datasplits()
        # if we don't want to resume optimization:
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_optuna, self.path_trials, self.path_results]:
            if not self.experiment.ml_config["resume_optimization"]:
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

        # assuming n_jobs=1 for every model
        if self.experiment.ml_config["inner_parallelization"] is True:
            num_requested_cpus = (
                self.experiment.ml_config["n_workers"]
                * self.experiment.ml_config["n_jobs"]
            )
        else:
            num_requested_cpus = self.experiment.ml_config["n_jobs"]

        print(f"Number of requested CPUs for this experiment: {num_requested_cpus}")
        print()

    def run_experiment(self):
        """Run experiment."""
        if self.experiment.ml_config["global_hyperparameter"]:
            self.run_globalhp_optimization()
        else:
            self.run_individualhp_optimization()

        # create best bag in results directory
        # - attach best bag to experiment
        # - attach best bag scores to experiment
        self.create_best_bag()

        return self.experiment

    def create_best_bag(self):
        """Create best bag from bags found in results.

        This code here is only meant to show the desired functionality
        and needs to be improved.
        It shows an indirect way of creating the best bag. It
        would be preferable to access the optuna results and
        then create the best bag from them.
        """
        path_bags = list(self.path_results.rglob("*.pkl"))

        if len(path_bags) == 1:
            # only single bag found - copy to best bag
            shutil.copy(path_bags[0], self.path_results.joinpath("best_bag.pkl"))
            file = path_bags[0]
            if file.is_file():
                best_bag = TrainingsBag.from_pickle(file)
        elif len(path_bags) > 1:
            # collect all trainings from bags
            trainings = list()
            for file in path_bags:
                if file.is_file():
                    bag = TrainingsBag.from_pickle(file)
                    trainings.extend(bag.trainings)
            # create best bag
            best_bag = TrainingsBag(
                trainings=trainings,
                parallel_execution=self.experiment.ml_config["inner_parallelization"],
                num_workers=self.experiment.ml_config["n_workers"],
                target_metric=self.experiment.config["target_metric"],
                row_column=self.experiment.row_column,
            )
            # save best bag
            best_bag.to_pickle(self.path_results.joinpath("best_bag.pkl"))
        else:
            raise ValueError("No bags founds in results directory")

        # save performance values of best bag
        best_bag_scores = best_bag.get_scores()
        # show and save test results, MAE
        print(
            f"Experiment: {self.experiment.id} "
            f"Test (ensembled predictions) {self.experiment.config['target_metric']}:"
            f"{best_bag_scores['test_pool_hard']}"
        )

        with open(
            self.path_results.joinpath("best_bag_scores.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(best_bag_scores, f)

        # save best bag to the experiment
        self.experiment.models["best"] = best_bag

        # save best bag scores to the experiment
        self.experiment.scores = best_bag_scores

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

        if self.experiment.ml_config["inner_parallelization"]:
            print("Parallel execution of Optuna optimizations for individual HPs")
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.experiment.ml_config["n_workers"]
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
            print("Sequential execution of Optuna optimizations for individual HPs")
            for split in splits:
                self.optimize_splits(split)
                print(f"Optimization of split:{split.keys()} completed")

    def optimize_splits(self, splits):
        """Optimize splits.

        Works if splits contain several splits as well as
        when splits only contains a single split
        """
        # define study name by joined keys of splits
        study_name = "optuna_" + str(sum([int(key) for key in splits.keys()]))

        # set up Optuna study
        objective = ObjectiveOptuna(
            experiment=self.experiment,
            data_splits=splits,
            study_name=study_name,
            save_trials=self.experiment.ml_config["save_trials"],
        )

        # multivariate sampler with group option
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=self.experiment.ml_config["optuna_seed"],
            n_startup_trials=self.experiment.ml_config["n_optuna_startup_trials"],
        )

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
            n_trials=self.experiment.ml_config["n_trials"],
        )

        # copy bag of best trial to results
        # not needed anymore
        # source = self.experiment.path_study.joinpath(
        #    self.experiment.path_sequence_item,
        #    "trials",
        #    f"study{study_name}trial{study.best_trial.number}_bag.pkl",
        # )
        # shutil.copy(source, self.path_results / source.name)

        # display results
        print()
        print("Optimization results: ")
        # print("Best trial:", study.best_trial) #full info
        print("Best trial number:", study.best_trial.number)
        print("Best target value:", study.best_value)
        user_attrs = study.best_trial.user_attrs
        performance_info = {
            key: v for key, v in user_attrs.items() if key not in ["config_training"]
        }
        print("Best parameters:", user_attrs["config_training"])
        print("Performance Info:", performance_info)
        print("Optimization completed")

        # create best bag from optuna info
        best_trainings = list()
        for key, split in splits.items():
            best_trainings.append(
                Training(
                    training_id=self.experiment.id + "_" + str(key),
                    ml_type=self.experiment.ml_type,
                    target_assignments=self.experiment.target_assignments,
                    feature_columns=self.experiment.feature_columns,
                    row_column=self.experiment.row_column,
                    data_train=split["train"],  # inner datasplit, train
                    data_dev=split["test"],  # inner datasplit, dev
                    data_test=self.experiment.data_test,
                    config_training=user_attrs["config_training"],
                    target_metric=self.experiment.config["target_metric"],
                )
            )
        # create bag with all provided trainings
        best_bag = TrainingsBag(
            trainings=best_trainings,
            parallel_execution=self.experiment.ml_config["inner_parallelization"],
            num_workers=self.experiment.ml_config["n_workers"],
            target_metric=self.experiment.config["target_metric"],
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in best_bag
        best_bag.fit()
        # save best bag
        best_bag.to_pickle(
            self.path_results.joinpath(
                f"study{study_name}trial{study.best_trial.number}_bag.pkl"
            )
        )

        return True

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
        save_trials,
    ):
        self.experiment = experiment
        self.data_splits = data_splits
        self.study_name = study_name
        self.save_trials = save_trials
        # parameters potentially used for optimizations
        self.ml_model_types = self.experiment.ml_config["models"]
        self.dim_red_methods = self.experiment.ml_config["dim_red_methods"]
        self.max_outl = self.experiment.ml_config["max_outl"]
        # fixed parameters
        self.ml_seed = self.experiment.ml_config["model_seed"]
        self.ml_jobs = self.experiment.ml_config["n_jobs"]
        # training parameters
        self.parallel_execution = self.experiment.ml_config["inner_parallelization"]
        self.num_workers = self.experiment.ml_config["n_workers"]

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
            "n_jobs": self.ml_jobs,
            "model_seed": self.ml_seed,
        }
        translate = parameters_inventory[ml_model_type]["translate"]
        for key, value in fixed_global_parameters.items():
            if translate[key] != "NA":  # NA=ignore
                model_params[translate[key]] = value

        config_training = {
            "dim_reduction": dim_reduction,
            "outl_reduction": num_outl,
            "ml_model_type": ml_model_type,
            "ml_model_params": model_params,
        }

        # create trainings
        trainings = list()
        for key, split in self.data_splits.items():
            trainings.append(
                Training(
                    training_id=self.experiment.id + "_" + str(key),
                    ml_type=self.experiment.ml_type,
                    target_assignments=self.experiment.target_assignments,
                    feature_columns=self.experiment.feature_columns,
                    row_column=self.experiment.row_column,
                    data_train=split["train"],  # inner datasplit, train
                    data_dev=split["test"],  # inner datasplit, dev
                    data_test=self.experiment.data_test,
                    config_training=config_training,
                    target_metric=self.experiment.config["target_metric"],
                )
            )
        # create bag with all provided trainings
        bag_trainings = TrainingsBag(
            trainings=trainings,
            parallel_execution=self.parallel_execution,
            num_workers=self.num_workers,
            target_metric=self.experiment.config["target_metric"],
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in bag
        bag_trainings.fit()

        # save bag if desired
        if self.save_trials:
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

        # add config_training to user attributes
        trial.set_user_attr("config_training", config_training)

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
    target_metric: str = field(validator=[validators.instance_of(str)])
    # configuration for training
    config_training: dict = field(validator=[validators.instance_of(dict)])
    # training output
    model = field(default=None)
    predictions: dict = field(default=dict(), validator=[validators.instance_of(dict)])
    # scaler
    scaler = field(init=False)

    @property
    def dim_reduction(self) -> str:
        """Dimension reduction method."""
        return self.config_training["dim_reduction"]

    @property
    def outl_reduction(self) -> int:
        """Parameter outlier reduction method."""
        return self.config_training["outl_reduction"]

    @property
    def ml_model_type(self) -> str:
        """Dimension reduction method."""
        return self.config_training["ml_model_type"]

    @property
    def ml_model_params(self) -> dict:
        """Dimension reduction method."""
        return self.config_training["ml_model_params"]

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

    def __attrs_post_init__(self):
        self.scaler = MinMaxScaler()

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

    def fit(self):
        """Run trainings."""
        # missing:
        # (1) missing: outlier removal
        # (2) scaling
        # (3) missinf dim reduction

        # scaling (!after outlier removal)
        # x_train_scaled = self.scaler.fit_transform(self.x_train)
        # x_dev_scaled = self.scaler.transform(self.x_dev)
        # x_test_scaled = self.scaler.transform(self.x_test)
        self.model = model_inventory[self.ml_model_type](**self.ml_model_params)

        if len(self.target_assignments) == 1:
            # standard sklearn single target models
            self.model.fit(
                # x_train_scaled,
                self.x_train,
                self.y_train.squeeze(axis=1),
            )
        else:
            # multi target models, incl. time2event
            # self.model.fit(x_train_scaled, self.y_train)
            self.model.fit(self.x_train, self.y_train)

        self.predictions["train"] = pd.DataFrame()
        self.predictions["train"][self.row_column] = self.data_train[self.row_column]
        self.predictions["train"]["target"] = self.y_train.squeeze(axis=1)
        # self.predictions["train"]["prediction"] = self.model.predict(x_train_scaled)
        self.predictions["train"]["prediction"] = self.model.predict(self.x_train)

        self.predictions["dev"] = pd.DataFrame()
        self.predictions["dev"][self.row_column] = self.data_dev[self.row_column]
        self.predictions["dev"]["target"] = self.y_dev.squeeze(axis=1)
        # self.predictions["dev"]["prediction"] = self.model.predict(x_dev_scaled)
        self.predictions["dev"]["prediction"] = self.model.predict(self.x_dev)

        self.predictions["test"] = pd.DataFrame()
        self.predictions["test"][self.row_column] = self.data_test[self.row_column]
        self.predictions["test"]["target"] = self.y_test.squeeze(axis=1)
        # self.predictions["test"]["prediction"] = self.model.predict(x_test_scaled)
        self.predictions["test"]["prediction"] = self.model.predict(self.x_test)

        if self.ml_type == "classification":
            columns = [int(x) for x in self.model.classes_]  # column names --> int
            # self.predictions["train"][columns] = self.model.predict_proba(
            #    x_train_scaled
            # )
            self.predictions["train"][columns] = self.model.predict_proba(self.x_train)
            # self.predictions["dev"][columns] = self.model.predict_proba(x_dev_scaled)
            self.predictions["dev"][columns] = self.model.predict_proba(self.x_dev)
            # self.predictions["test"][columns] =self.model.predict_proba(x_test_scaled)
            self.predictions["test"][columns] = self.model.predict_proba(self.x_test)

        # missing: other feature importance methods
        # result = permutation_importance(
        #    self.model, X=x_dev_scaled, y=self.y_dev, n_repeats=10, random_state=0
        # )
        # print(result)
        return self

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
    parallel_execution: bool = field(validator=[validators.instance_of(bool)])
    num_workers: int = field(validator=[validators.instance_of(int)])
    target_metric: str = field(validator=[validators.instance_of(str)])
    row_column: str = field(validator=[validators.instance_of(str)])
    # path: Path = field(default=Path(), validator=[validators.instance_of(Path)])
    train_status: bool = field(default=False)

    def fit(self):
        """Run all available trainings."""
        if self.parallel_execution is True:
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
            ) as executor:
                futures = []
                train_results = []
                for i in self.trainings:
                    try:
                        future = executor.submit(i.fit)
                        futures.append(future)
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while submitting task: {e}")
                for future in concurrent.futures.as_completed(futures):
                    try:
                        train_results.append(future.result())
                        print("Inner parallel training completed")
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Exception occurred while executing task: {e}")
                        print(f"Exception: {type(e).__name__}")
                # replace trainings with processed trainings
                # order in self.trainings may change!
                self.trainings = train_results

        else:
            for training in self.trainings:
                training.fit()
                print("Inner sequential training completed")

        self.train_status = True

    def get_scores(self):
        """Get scores."""
        if not self.train_status:
            print("Running trainings first to be able to get scores")
            self.fit()

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


@define
class OctopusFullConfig:
    """OctopusLightConfig."""

    models: List = field(
        # validator=[validators.in_(["ExtraTreesRegressor", "RandomForestRegressor"])],
    )
    """Models for ML."""

    module: List = field(default="octofull")
    """Models for ML."""

    description: str = field(validator=[validators.instance_of(str)], default=None)
    """Description."""
    # datasplit
    n_folds_inner: int = field(validator=[validators.instance_of(int)], default=5)
    """Number of inner folds."""

    datasplit_seed_inner: int = field(
        validator=[validators.instance_of(int)], default=0
    )
    """Data split seed for inner loops."""
    # model training

    model_seed: int = field(validator=[validators.instance_of(int)], default=0)
    """Model seed."""

    n_jobs: int = field(validator=[validators.instance_of(int)], default=1)
    """Number of parallel jobs."""

    dim_red_methods: List = field(default=[""])
    """Methods for dimension reduction."""

    max_outl: int = field(validator=[validators.instance_of(int)], default=5)
    """?"""
    # parallelization
    inner_parallelization: bool = field(
        validator=[validators.instance_of(bool)], default=False
    )

    n_workers: int = field(validator=[validators.instance_of(int)], default=None)
    """Number of workers."""
    # hyperparamter optimization
    optuna_seed: int = field(validator=[validators.instance_of(int)], default=None)
    """Seed for Optuna TPESampler, default=no seed"""

    n_optuna_startup_trials: int = field(
        validator=[validators.instance_of(int)], default=10
    )
    """Number of Optuna startup trials (random sampler)"""

    global_hyperparameter: bool = field(
        validator=[validators.in_([True, False])], default=True
    )
    """Selection of hyperparameter set."""

    n_trials: int = field(validator=[validators.instance_of(int)], default=100)
    """Number of Optuna trials."""

    hyperparameter: dict = field(validator=[validators.instance_of(dict)], default={})
    """Bring own hyperparameter space."""

    max_features: int = field(validator=[validators.instance_of(int)], default=-1)
    """Maximum features."""

    save_trials: bool = field(validator=[validators.instance_of(bool)], default=False)
    """Save trials (bags)."""

    resume_optimization: bool = field(
        validator=[validators.instance_of(bool)], default=False
    )
    """Resume HPO, use existing optuna.db, don't delete optuna.de"""

    def __attrs_post_init__(self):
        # set default of n_workers to n_folds_inner
        if self.n_workers is None:
            self.n_workers = self.n_folds_inner
        if self.n_workers != self.n_folds_inner:
            print(
                f"Octofull Warning: n_workers ({self.n_workers}) "
                f"does not match n_folds_inner ({self.n_folds_inner})",
            )
