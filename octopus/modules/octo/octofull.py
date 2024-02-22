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
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.objective_optuna import ObjectiveOptuna
from octopus.modules.octo.training import Training
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
# - (1) include data preprocessing
# - (2) Ensemble selection
# - (3) feature counts in bag
# - (1) num feat constraint (automatic parameters)
#       + scaling factor from first random optuna runs
#       + max_feature from dataset size
# - (2) add T2E model
# - (3) create predict/predict_proba function for bag
#       Does it work with shap and permutation feature importance?
# - (4) TestFI: Apply shape and permutation feature importance to bag to test
#       compare to fis from individual trainings

# - (5) is there a good way to determine which shap values are relevant, stats test?
# - (6) make bag compatible with sklearn
#   +very difficult as sklearn differentiates between regression, classification
#   RegressionBag, ClassBag
#   +we also want to include T2E
#   + check if a single predict function is sufficient for shap/permutation importance
# - (7) basic analytics class
# - (8) Make use of default model parameters, see autosk, optuna
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
        self._check_resources()

    def _check_resources(self):
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
            self._run_globalhp_optimization()
        else:
            self._run_individualhp_optimization()

        # create best bag in results directory
        # - attach best bag to experiment
        # - attach best bag scores to experiment
        self._create_best_bag()

        return self.experiment

    def _create_best_bag(self):
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
                best_bag = Bag.from_pickle(file)
        elif len(path_bags) > 1:
            # collect all trainings from bags
            trainings = list()
            for file in path_bags:
                if file.is_file():
                    bag = Bag.from_pickle(file)
                    trainings.extend(bag.trainings)
            # create best bag
            best_bag = Bag(
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
        self.experiment.selected_features = best_bag.features_used
        print("Number of original features:", len(self.experiment.feature_columns))
        print("Number of selected features:", len(self.experiment.selected_features))

        # save feature importances to experiment
        self.experiment.feature_importances = best_bag.get_feature_importances()

        # save test predictions to experiment
        self.experiment.predictions = best_bag.get_predictions()

    def _run_globalhp_optimization(self):
        """Optimization run with a global HP set over all inner folds."""
        print("Running Optuna Optimization with a global HP set")

        # run Optuna study with a global HP set
        self._optimize_splits(self.data_splits)

    def _run_individualhp_optimization(self):
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
                        future = executor.submit(self._optimize_splits, split)
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
                self._optimize_splits(split)
                print(f"Optimization of split:{split.keys()} completed")

    def _optimize_splits(self, splits):
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
            direction="minimize",  # metric adjustment in optuna objective
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
        best_bag = Bag(
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
