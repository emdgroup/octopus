"""OctoFull core function."""

# import concurrent.futures
import copy
import json
import shutil
import warnings
from pathlib import Path

import optuna
from attrs import define, field, validators
from joblib import Parallel, delayed
from optuna.samplers._tpe.sampler import ExperimentalWarning

from octopus.experiment import OctoExperiment
from octopus.modules.octo.bag import Bag
from octopus.modules.octo.enssel import EnSel
from octopus.modules.octo.objective_optuna import ObjectiveOptuna
from octopus.modules.octo.training import Training
from octopus.results import ModuleResults
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

# TOBEDONE ENSEMBLING
# - simplify and centralise score calculations (bag, enssel)
# - add hard pooling to ensemble prediction


# TOBEDONE BASE
# - check module type
# - autosk with serial processing of outer folds does not use significant CPU??
# - check that openblas settings are correct and suggest solutions

# TOBEDONE OCTOFULL
# - (0) simplement equence branching by spedifying in every module where the input
#       data comes from.
# - (0) logisticregression model -- feature importance via coeff
#       + scaling needed
# - (0) MRMR as part of Octo optimization
# - (0) Octo selected_features based on "ensel"
# - (0) Parallelize pfi calculation in training!!
# - (0) Optuna trials should be >> than optuna_start_trials
# - (0) RFE - better test results than octo (datasplit difference?)
# - (0) ensemble or pooling, some metrics (ACC) require int, others not (MSE)
# - (0) Cleanup: get rid of individual HP
# - (0) Cleanup: add more info to metrics (prict/proba, scoring,
#       scoring_string sklearn, input type (int, float))
# - (0) Cleanup: Optuna, use score consistently everywhere to remove complexity
# - (0) Improvement high priority: IMPUTATION
# - (0) Improvement high priority: sequence handover
# - (0) selected features is currently only taken from best bag, and not ensel
# - (0) fix sklearn due to new ModelResults
# - (0) ensemble selection - use training weight
#       training weight needs to be considered in bag fi, score, predict
# - (1) ensemble selection - missing
#       - max_n_iterations need to become configurable
#       - save best bag scores to the experiment
#       - calculate and save specified feature importances of best bag
#       - save selected features to experiment
#       - save test predictions to experiment
# - (1) complete ensemble selection -- important for mrmr!, feature counts, etc..
# - (2) Clean up fi code and, remove duplicates and put into one place!
# - (3) predict group_pfi --- now based on feature_groups (may not use rdc) for
#        group selection
# - (4) MRMR -- qubim either uses F1, Smolonogov-stats or Shapley as input
#       https://github.com/smazzanti/mrmr/blob/main/mrmr/pandas.py
#       use this implementation?
# - (5) Feature Counts in Bag - is everything available?
# - (6) Revise sequence handover. Handover a dict that may contain
#       (a) selected feature (b) previous fis (c) changed dataset
# - (7) use numba to speed up rdc
# - (8) hierarchical clustering for feature auto-groups
# - (9) !Calculate performance of survival models: CI, CI_uno,  IBS, dynAUC,
# - (10) rename ensemble test?
# - (11) include data preprocessing
# - (12) num feat constraint (automatic parameters)
#       + scaling factor from first random optuna runs
#       + max_feature from dataset size
# - (13) is there a good way to determine which shap values are relevant, stats test?
# - (14) make bag compatible with sklearn
#   +very difficult as sklearn differentiates between regression, classification
#   RegressionBag, ClassBag
#   +we also want to include T2E
#   + check if a single predict function is sufficient for shap/permutation importance
# - (14) Make use of default model parameters, see autosk, optuna -- meaningful?
# - predictions, replace "ensemble_test" with
#   experiment_id+ sequence_id + ensemble + [test]
# - Performance evaluation generalize: ensemble_hard, ensemble_soft
# - deepchecks - https://docs.deepchecks.com/0.18/tabular/auto_checks/data_integrity/index.html
# - outer parallelizaion can lead to very differing execution times per experiment!
# - check that for classification only classification modules are used
# - sequence config -- module is fixed
# - improve create_best_bags - use a direct way, from returned best trial or optuna.db
# - xgoost class weights need to be set in training! How to solve that?
# - check disk space and inform about disk space requirements
# - Shap FI is too complicated for sksurv models - strangely, it takes very long


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
# - add scaling, as a HP?
# - include dimensionality reduction
# - include outlier elimination

# TOBEDONE FEATURE_IMPORTANCE
# - (1) grouped permutation importance, see predict class
# - (2) use feature importance counts
# - (3) !!!! Average FIs differently -- see bag, no groupby.mean()
# - check alibi package
# - separate fi code from training class
# - group identification (experiment.py) - add hirarchical clustering
# - create new module that filters out correlated variables
# - crate new module that replaces groups with PCA 1st component
# - https://arxiv.org/pdf/2312.10858
# - see alibi package, ALE, https://github.com/SeldonIO/alibi/tree/master
# - add kernel shape fi


@define
class OctoCore:
    """Manages and executes machine learning experiments.

    This class integrates all components necessary for conducting
    experiments using OctoExperiment configurations.
    It supports operations such as data splitting, path management,
    model optimization with Optuna, and results handling.
    The class is designed to work seamlessly with the defined experiment
    configurations and ensures robust handling of experiment resources,
    directories, and optimization processes.

    Attributes:
        experiment (OctoExperiment): Configuration and data container
            for the experiment.
        data_splits (dict): Stores training and validation data splits.
        paths_optuna_db (dict): Stores file paths to Optuna databases
            for each experiment.
        top_trials (list): Keeps track of the best performing trials.

    Raises:
        ValueError: Thrown when encountering invalid operations or unsupported
            configurations during the experiment's execution.

    Usage:
        An instance of this class is initialized with an OctoExperiment
        object and utilizes its methods to run comprehensive machine
        learning experiments. This includes preparing data, optimizing
        model parameters, and evaluating results. Proper error handling
        is incorporated to manage any discrepancies during the experiment phases.
    """

    experiment: OctoExperiment = field(
        validator=[validators.instance_of(OctoExperiment)]
    )
    # model = field(default=None)
    data_splits: dict = field(init=False, validator=[validators.instance_of(dict)])

    paths_optuna_db: dict = field(init=False, validator=[validators.instance_of(dict)])

    top_trials: list = field(init=False, validator=[validators.instance_of(list)])

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
        # initialization here due to "Python immutable default"
        self.paths_optuna_db = dict()
        self.top_trials = []

        # create datasplit during init
        self.data_splits = DataSplit(
            dataset=self.experiment.data_traindev,
            datasplit_col=self.experiment.datasplit_column,
            seed=self.experiment.ml_config.datasplit_seed_inner,
            num_folds=self.experiment.ml_config.n_folds_inner,
            stratification_col=self.experiment.stratification_column,
        ).get_datasplits()
        # if we don't want to resume optimization:
        # delete directories /trials /optuna /results to ensure clean state
        # of module when restarted, required for parallel optuna runs
        # as optuna.create(...,load_if_exists=True)
        # create directory if it does not exist
        for directory in [self.path_optuna, self.path_trials, self.path_results]:
            if not self.experiment.ml_config.resume_optimization:
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
        if self.experiment.ml_config.inner_parallelization is True:
            num_requested_cpus = (
                self.experiment.ml_config.n_workers * self.experiment.ml_config.n_jobs
            )
        else:
            num_requested_cpus = self.experiment.ml_config.n_jobs

        print(f"Number of requested CPUs for this experiment: {num_requested_cpus}")
        print()

    def run_experiment(self):
        """Run experiment."""
        # (1) model training and optimization
        if self.experiment.ml_config.global_hyperparameter:
            self._run_globalhp_optimization()
        else:
            self._run_individualhp_optimization()

        # create best bag in results directory
        # - attach best bag to experiment
        # - attach best bag scores to experiment
        self._create_best_bag()

        # (2) ensemble selection, only globalhp scenario is supported
        if (
            self.experiment.ml_config.global_hyperparameter
            & self.experiment.ml_config.ensemble_selection
        ):
            self._run_ensemble_selection()

        return self.experiment

    def _run_ensemble_selection(self):
        """Run ensemble selection."""
        ensel = EnSel(
            target_metric=self.experiment.configs.study.target_metric,
            path_trials=self.path_trials,
            max_n_iterations=100,
            row_column=self.experiment.row_column,
            target_assignments=self.experiment.target_assignments,
        )
        ensemble_paths_dict = ensel.optimized_ensemble
        self._create_ensemble_bag(ensemble_paths_dict)

    def _create_ensemble_bag(self, ensemble_paths_dict):
        """Create ensemble bag from a ensemble path dict."""
        if len(ensemble_paths_dict) == 0:
            raise ValueError("Valid ensemble information need to be provided")

        # extract trainings
        # here, we don't use the weight info
        # this requires more work for scores and feature importances
        trainings = list()
        train_id = 0
        for path, weight in ensemble_paths_dict.items():
            bag = Bag.from_pickle(path)
            for training in bag.trainings:
                # training.training_weight - tobedone
                for _ in range(int(weight)):
                    train_cp = copy.deepcopy(training)
                    train_cp.training_id = self.experiment.id + "_" + str(train_id)
                    train_cp.training_weight = 1
                    train_id += 1
                    trainings.append(train_cp)

        # create ensemble bag
        ensel_bag = Bag(
            bag_id=self.experiment.id + "_ensel",
            trainings=trainings,
            train_status=True,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.experiment.ml_config.inner_parallelization,
            num_workers=self.experiment.ml_config.n_workers,
            target_metric=self.experiment.configs.study.target_metric,
            row_column=self.experiment.row_column,
        )
        # save ensel bag
        ensel_bag.to_pickle(self.path_results.joinpath("ensel_bag.pkl"))

        # save performance values of best bag
        ensel_scores = ensel_bag.get_scores()
        # show and save test results
        print("Ensemble selection performance")
        print(
            f"Experiment: {self.experiment.id} "
            f"{self.experiment.configs.study.target_metric} "
            f"(ensembled, hard vote):"  # noqa E501
            f"Dev {ensel_scores['dev_pool_hard']:.3f}, "
            f"Test {ensel_scores['test_pool_hard']:.3f}"
        )
        if self.experiment.ml_type == "classification":
            print(
                f"Experiment: {self.experiment.id} "
                f"{self.experiment.configs.study.target_metric} "
                f"(ensembled, soft vote):"  # noqa E501
                f"Dev {ensel_scores['dev_pool_soft']:.3f}, "
                f"Test {ensel_scores['test_pool_soft']:.3f}"
            )

        with open(
            self.path_results.joinpath("ensel_scores_scores.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ensel_scores, f)

        # calculate feature importances of best bag
        # fi_methods = self.experiment.ml_config.fi_methods_bestbag
        fi_methods = []  # disable calculation of pfi for ensel_bag
        ensel_bag_fi = ensel_bag.get_feature_importances(fi_methods)

        # calculate selected features
        selected_features = ensel_bag.get_selected_features(fi_methods)

        # save best bag and results to experiment
        self.experiment.results["ensel"] = ModuleResults(
            id="ensel",
            model=ensel_bag,
            scores=ensel_scores,
            feature_importances=ensel_bag_fi,
            predictions=ensel_bag.get_predictions(),
            selected_features=selected_features,
        )

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
                bag_id=self.experiment.id + "_best",
                trainings=trainings,
                target_assignments=self.experiment.target_assignments,
                parallel_execution=self.experiment.ml_config.inner_parallelization,
                num_workers=self.experiment.ml_config.n_workers,
                target_metric=self.experiment.config.target_metric,
                row_column=self.experiment.row_column,
            )
            # save best bag
            best_bag.to_pickle(self.path_results.joinpath("best_bag.pkl"))
        else:
            raise ValueError("No bags founds in results directory")

        # save performance values of best bag
        best_bag_scores = best_bag.get_scores()
        # show and save test results
        print(
            f"Experiment: {self.experiment.id} "
            f"{self.experiment.configs.study.target_metric} "
            f"(ensembled, hard vote):"  # noqa E501
            f"Dev {best_bag_scores['dev_pool_hard']:.3f}, "
            f"Test {best_bag_scores['test_pool_hard']:.3f}"
        )
        if self.experiment.ml_type == "classification":
            print(
                f"Experiment: {self.experiment.id} "
                f"{self.experiment.configs.study.target_metric} "
                f"(ensembled, soft vote):"  # noqa E501
                f"Dev {best_bag_scores['dev_pool_soft']:.3f}, "
                f"Test {best_bag_scores['test_pool_soft']:.3f}"
            )

        with open(
            self.path_results.joinpath("best_bag_scores.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(best_bag_scores, f)

        # calculate feature importances of best bag
        fi_methods = self.experiment.ml_config.fi_methods_bestbag
        best_bag_fi = best_bag.get_feature_importances(fi_methods)

        # calculate selected features
        selected_features = best_bag.get_selected_features(fi_methods)

        # save best bag and results to experiment
        self.experiment.results["best"] = ModuleResults(
            id="best",
            model=best_bag,
            scores=best_bag_scores,
            feature_importances=best_bag_fi,
            selected_features=selected_features,
            predictions=best_bag.get_predictions(),
        )

        # save selected features to experiment
        print("Number of original features:", len(self.experiment.feature_columns))
        self.experiment.selected_features = selected_features
        print("Number of selected features:", len(self.experiment.selected_features))
        if len(self.experiment.selected_features) == 0:
            print(
                "Warning: No feature importance method specified, "
                "or specified method is not applicable to model."
            )

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

        if self.experiment.ml_config.inner_parallelization:
            # (A) joblib parallelization, compatible with xgboost
            def optimize_split(split, split_index):
                try:
                    self._optimize_splits(split)
                    print(f"Optimization of split {split_index} completed")
                except Exception as e:  # pylint: disable=broad-except
                    print(
                        f"Exception occurred while optimizing split {split_index}: {e}"
                    )
                    print(f"Exception type: {type(e).__name__}")

            print("Parallel execution of Optuna optimizations for individual HPs")
            with Parallel(n_jobs=self.experiment.ml_config.n_workers) as parallel:
                parallel(
                    delayed(optimize_split)(split, idx)
                    for idx, split in enumerate(splits)
                )

            # (B) Alternative with xbgoost issue, issue46
            #    print("Parallel execution of Optuna optimizations for individual HPs")
            #    # max_tasks_per_child=1 requires Python3.11
            #    with concurrent.futures.ProcessPoolExecutor(
            #        max_workers=self.experiment.ml_config["n_workers"]
            #    ) as executor:
            #        futures = []
            #        for split in splits:
            #            try:
            #                future = executor.submit(self._optimize_splits, split)
            #                futures.append(future)
            #            except Exception as e:  # pylint: disable=broad-except
            #                print(f"Exception occurred while submitting task: {e}")
            #        for future in concurrent.futures.as_completed(futures):
            #            try:
            #                _ = future.result()
            #                print("Optimization of single split completed")
            #            except Exception as e:  # pylint: disable=broad-except
            #                print(f"Exception occurred while executing task: {e}")

        else:
            print("Sequential execution of Optuna optimizations for individual HPs")
            for split in splits:
                try:
                    self._optimize_splits(split)
                    print(f"Optimization of split:{split.keys()} completed")
                except Exception as e:  # pylint: disable=broad-except
                    print(
                        f"Error during optimizatio of split:{split.keys()}: {e},"
                        f" type: {type(e).__name__}"
                    )

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
            top_trials=self.top_trials,
        )

        # multivariate sampler with group option
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=self.experiment.ml_config.optuna_seed,
            n_startup_trials=self.experiment.ml_config.n_optuna_startup_trials,
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
        # store optuna db path
        self.paths_optuna_db[study_name] = db_path

        study.optimize(
            objective,
            n_jobs=1,
            n_trials=self.experiment.ml_config.n_trials,
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
                    target_metric=self.experiment.configs.study.target_metric,
                    max_features=self.experiment.ml_config.max_features,
                    feature_groups=self.experiment.feature_groups,
                )
            )
        # create bag with all provided trainings
        best_bag = Bag(
            bag_id=self.experiment.id + "_best",
            trainings=best_trainings,
            target_assignments=self.experiment.target_assignments,
            parallel_execution=self.experiment.ml_config.inner_parallelization,
            num_workers=self.experiment.ml_config.n_workers,
            target_metric=self.experiment.configs.study.target_metric,
            row_column=self.experiment.row_column,
            # path?
        )

        # train all models in best_bag
        best_bag.fit()
        # save best bag
        best_bag.to_pickle(
            self.path_results.joinpath(
                f"{study_name}_trial{study.best_trial.number}_bag.pkl"
            )
        )

        return True
