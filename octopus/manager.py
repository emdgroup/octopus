"""OctoManager."""
import concurrent.futures
import copy
from os import cpu_count
from pathlib import Path

from attrs import define, field, validators

from octopus.config import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.modules.config import modules_inventory


@define
class OctoManager:
    """OctoManager."""

    base_experiments: list = field(
        validator=[validators.instance_of(list)],
    )
    oconfig: OctoConfig = field(
        validator=[validators.instance_of(OctoConfig)],
    )

    def run_outer_experiments(self):
        """Run outer experiments."""
        print("Preparing execution of experiments.......")
        print("Execution:", self.oconfig.cfg_manager["ml_execution"])
        print("Only first experiment:", self.oconfig.cfg_manager["ml_only_first"])
        print()
        print("Parallel execution info")
        print("Number of outer folds: ", self.oconfig.k_outer)
        print("Number of logical CPUs:", cpu_count())
        num_workers = min([self.oconfig.k_outer, cpu_count()])
        print("Number of outer fold workers:", num_workers)
        print()

        if len(self.base_experiments) == 0:
            raise ValueError("No experiments defined")

        if self.oconfig.cfg_manager["ml_only_first"] is True:
            print("Only running first experiment")
            self.create_execute_mlmodules(self.base_experiments[0])

        elif self.oconfig.cfg_manager["ml_execution"] == "sequential":
            for cnt, base_experiment in enumerate(self.base_experiments):
                print("#### Outerfold:", cnt)
                self.create_execute_mlmodules(base_experiment)
                print()
        # tobedone: suppress output
        # tobedone: show which outer fold has been completed
        elif self.oconfig.cfg_manager["ml_execution"] == "parallel":
            # max_tasks_per_child=1 requires Python3.11
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
            ) as executor:
                futures = [
                    executor.submit(self.create_execute_mlmodules, i)
                    for i in self.base_experiments
                ]
                for _ in concurrent.futures.as_completed(futures):
                    print("Outer fold completed")
        else:
            raise ValueError("Execution type not supported")

    def create_execute_mlmodules(self, base_experiment: OctoExperiment):
        """Create and execute ml modules."""
        selected_features = []
        for cnt, element in enumerate(self.oconfig.cfg_sequence):
            print("ml step:", cnt)
            print("ml_module:", element["ml_module"])
            print("description:", element["description"])

            # add config to experiment
            experiment = copy.deepcopy(base_experiment)
            experiment.ml_module = element["ml_module"]
            experiment.ml_config = element
            experiment.id = experiment.id + "_" + str(cnt)
            experiment.sequence_item_id = cnt
            experiment.path_sequence_item = Path(
                f"experiment{experiment.experiment_id}", f"sequence{cnt}"
            )

            # create directory for sequence item
            path_study_sequence = experiment.path_study.joinpath(
                experiment.path_sequence_item
            )
            path_study_sequence.mkdir(
                parents=True, exist_ok=not self.oconfig.production_mode
            )
            print("Running experiment: ", experiment.id)
            # save experiment before running experiment
            path_save = path_study_sequence.joinpath(
                f"exp{experiment.experiment_id}_{experiment.sequence_item_id}.pkl"
            )
            experiment.to_pickle(path_save)

            # update features with selected features from previous run
            if cnt > 0:
                experiment.features = selected_features

            # get desired module and intitialze with experiment
            if experiment.ml_module in modules_inventory:
                module = modules_inventory[experiment.ml_module](experiment)
            else:
                raise ValueError(f"ml_module {experiment.ml_module} not supported")

            # run module and overwrite experiment
            experiment = module.run_experiment()

            # extract selected features after running module
            selected_features = experiment.selected_features

            # save experiment
            experiment.to_pickle(path_save)
