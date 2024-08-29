"""OctoManager."""

# import concurrent.futures
import copy
import math
from os import cpu_count
from pathlib import Path

from attrs import define, field, validators
from joblib import Parallel, delayed

from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.modules import modules_inventory


@define
class OctoManager:
    """OctoManager."""

    base_experiments: list = field(
        validator=[validators.instance_of(list)],
    )
    configs: OctoConfig = field(
        validator=[validators.instance_of(OctoConfig)],
    )

    # def __attrs_post_init__(self):
    # set defaults in cfg_manager
    # self.configs.manager.setdefault("run_single_experiment_num", -1)
    # self.configs.manager.setdefault("outer_parallelization", False)

    def run_outer_experiments(self):
        """Run outer experiments."""
        print("Preparing execution of experiments.......")
        print("Outer parallelization:", self.configs.manager.outer_parallelization)
        single_exp = self.configs.manager.run_single_experiment_num
        if single_exp == -1:
            print("Run all experiments")
        else:
            print("Run single experiment:", single_exp)
        print()
        print("Parallel execution info")
        print("Number of outer folds: ", self.configs.study.n_folds_outer)
        print("Number of logical CPUs:", cpu_count())
        num_workers = min([self.configs.study.n_folds_outer, cpu_count()])
        print("Number of outer fold workers:", num_workers)
        print()

        if len(self.base_experiments) == 0:
            raise ValueError("No experiments defined")

        if single_exp != -1:
            print("Only running experiment:", single_exp)
            self.create_execute_mlmodules(self.base_experiments[single_exp])

        elif self.configs.manager.outer_parallelization is False:  # sequential
            for cnt, base_experiment in enumerate(self.base_experiments):
                print("#### Outerfold:", cnt)
                self.create_execute_mlmodules(base_experiment)
                print()
        # tobedone: suppress output
        elif self.configs.manager.outer_parallelization is True:  # parallel
            # (A) code using joblib
            def execute_task(base_experiment, index):
                try:
                    self.create_execute_mlmodules(base_experiment)
                    print(f"Outer fold {index} completed")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"Exception occurred while executing task{index}: {e}")

            with Parallel(n_jobs=num_workers) as parallel:
                parallel(
                    delayed(execute_task)(base_experiment, index)
                    for index, base_experiment in enumerate(self.base_experiments)
                )

            # (B) code using ProcessPoolExecutor
            # max_tasks_per_child=1 requires Python3.11
            # with concurrent.futures.ProcessPoolExecutor(
            #    max_workers=num_workers,
            # ) as executor:
            #    futures = []
            #    for i in self.base_experiments:
            #        try:
            #            future = executor.submit(self.create_execute_mlmodules, i)
            #            futures.append(future)
            #        except Exception as e:  # pylint: disable=broad-except
            #            print(f"Exception occurred while submitting task: {e}")
            #    for future in concurrent.futures.as_completed(futures):
            #        try:
            #            _ = future.result()
            #            print("Outer fold completed")
            #        except Exception as e:  # pylint: disable=broad-except
            #            print(f"Exception occurred while executing task: {e}")

    def create_execute_mlmodules(self, base_experiment: OctoExperiment):
        """Create and execute ml modules."""
        selected_features = []
        prior_feature_importances = {}
        for cnt, element in enumerate(self.configs.sequence.sequence_items):
            print("------------------------------------------")
            print("Step:", cnt)
            print("Module:", element.module)
            print("Description:", element.description)
            print("Load existing sequence item:", element.load_sequence_item)

            # sequence item is created and not load
            if not element.load_sequence_item:
                # add config to experiment
                experiment = copy.deepcopy(base_experiment)
                experiment.ml_module = element.module
                experiment.ml_config = element
                experiment.id = experiment.id + "_" + str(cnt)
                experiment.sequence_item_id = cnt
                experiment.path_sequence_item = Path(
                    f"experiment{experiment.experiment_id}", f"sequence{cnt}"
                )

                # calculating number of CPUs available to every experiment
                if self.configs.manager.outer_parallelization:
                    experiment.num_assigned_cpus = math.floor(
                        cpu_count() / self.configs.study.n_folds_outer
                    )
                else:
                    experiment.num_assigned_cpus = cpu_count()
                if self.configs.manager.run_single_experiment_num != -1:
                    experiment.num_assigned_cpus = cpu_count()

                # create directory for sequence item
                path_study_sequence = experiment.path_study.joinpath(
                    experiment.path_sequence_item
                )
                path_study_sequence.mkdir(parents=True, exist_ok=True)
                print("Running experiment: ", experiment.id)
                # save experiment before running experiment
                path_save = path_study_sequence.joinpath(
                    f"exp{experiment.experiment_id}_{experiment.sequence_item_id}.pkl"
                )

                # update features with selected features from previous run
                if cnt > 0:
                    experiment.feature_columns = selected_features
                    experiment.prior_feature_importances = prior_feature_importances

                # update feature groups as feature_columns may have changed
                experiment.calculate_feature_groups()

                # get desired module and intitialze with experiment
                if experiment.ml_module in modules_inventory:
                    module = modules_inventory[experiment.ml_module](experiment)
                else:
                    raise ValueError(f"ml_module {experiment.ml_module} not supported")

                # save experiment before running module
                experiment.to_pickle(path_save)

                # run module and overwrite experiment
                experiment = module.run_experiment()

                # extract selected features and feature importances after running module
                selected_features = experiment.selected_features
                prior_feature_importances = experiment.extract_fi_from_results()

                # save experiment after experiment has been completed
                experiment.to_pickle(path_save)

            # existing sequence item (experiment) is loaded
            else:
                path_study_sequence = base_experiment.path_study.joinpath(
                    f"experiment{base_experiment.experiment_id}", f"sequence{cnt}"
                )

                path_load = path_study_sequence.joinpath(
                    f"exp{base_experiment.experiment_id}_{cnt}.pkl"
                )

                if not path_load.exists():
                    raise FileNotFoundError("Sequence item to be loaded does not exist")

                experiment = OctoExperiment.from_pickle(path_load)
                print("Step loaded from: ", path_load)

                # extract selected features and feature importances from loaded module
                selected_features = experiment.selected_features
                prior_feature_importances = experiment.extract_fi_from_results()
