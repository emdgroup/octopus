"""OctoFull Module."""

import concurrent.futures
import json
import shutil
import warnings
from pathlib import Path

import optuna
import pandas as pd
from attrs import define, field, validators
from optuna.samplers._tpe.sampler import ExperimentalWarning

from octopus.experiment import OctoExperiment
from octopus.models.parameters import parameters_inventory
from octopus.models.utils import create_trialparams_from_config
from octopus.modules.octo.bags import TrainingsBag
from octopus.modules.octo.trainings import Training
from octopus.modules.utils import optuna_direction
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
# - autosk with serial processing of outer folds does not use significant CPU??
# - check that openblas settings are correct and suggest solutions

# TOBEDONE OCTOFULL
# - (1) return selected features, based on shap! and dev dataset
#       take internal if available, otherwise take shapley
#       test if internal is faster than shapley
# - (2) modify feat_importance functions to return fi for dev
# - (3) add feat num constraint
# - (4) add T2E model
# - (5) create predict/predict_proba function for bag
#       Does it work with shap and permutation feature importance?
# - (6) TestFI: Apply shape and permutation feature importance to bag to test
#       compare to fis from individual trainings

# - (7) is there a good way to determin which shap values are relevant, stats test?
# - (8) make bag compatible with sklearn
#   +very difficult as sklearn differentiates between regression, classification
#   RegressionBag, ClassBag
#   +we also want to include T2E
#   + check if a single predict function is sufficient for shap/permutation importance
# - (9) basic analytics class
# - (10) Make use of default model parameters, see autosk, optuna
# - Performance evaluation generalize: ensemble_hard, ensemble_soft
# - automatically remove features with a single value! and provide user feedback
# - deepchecks - https://docs.deepchecks.com/0.18/tabular/auto_checks/data_integrity/index.html
# - outer parallelizaion can lead to very differing execution times per experiment!
# - check that for classification only classification modules are used
# - sequence config -- module is fixed
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

        # save selected features to experiment
        self.experiment.selected_features = best_bag.get_selected_features()
        print("Number of original features:", len(self.experiment.feature_columns))
        print("Number of selected features:", len(self.experiment.selected_features))

        # save feature importances to experiment
        self.experiment.feature_importances = best_bag.get_feature_importances()

        # save test predictions to experiment
        self.experiment.test_predictions = best_bag.get_test_predictions()

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
