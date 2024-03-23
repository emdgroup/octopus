"""OctoML module."""

import shutil
import sys
from pathlib import Path

from attrs import asdict, define, field, validators

from octopus.config import OctoConfig
from octopus.data import OctoData
from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager
from octopus.utils import DataSplit


@define
class OctoML:
    """OctoML."""

    odata: OctoData = field(validator=[validators.instance_of(OctoData)])
    oconfig: OctoConfig = field(validator=[validators.instance_of(OctoConfig)])
    experiments: list = field(init=False)
    manager: OctoManager = field(init=False, default=None)

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.experiments = list()

    def create_outer_experiments(self):
        """Create outer experiments."""
        # create study folder and check if it exists in prod mode
        path_study = Path(self.oconfig.output_path, self.oconfig.study_name)

        # delete if study exists
        if path_study.exists():
            # offer exit option in prod mode
            if self.oconfig.production_mode:
                confirmation = input(
                    "Study exists, do you want to continue (resume)? (yes/no): "
                )
                if confirmation.lower() == "yes":
                    print("Continuing...")
                else:
                    print("Exiting...")
                    sys.exit()
            if self.oconfig.start_with_empty_study:
                shutil.rmtree(path_study)

        path_study.mkdir(parents=True, exist_ok=True)
        print("Path to study:", path_study)

        # create subfolders
        for subdir in ["data", "config"]:
            path_sub = path_study.joinpath(subdir)
            path_sub.mkdir(parents=False, exist_ok=True)

        # save files
        if subdir == "data":
            self.odata.to_pickle(path_sub.joinpath("data.pkl"))
            self.odata.save(path_sub)
        elif subdir == "config":
            self.oconfig.to_json(path_sub.joinpath("config.json"))  # human readable
            self.oconfig.to_pickle(path_sub.joinpath("config.dill"))

        # restrict dataset to relevant columns ("need to know basis")
        target_cols = list(self.odata.target_columns.keys())
        stratification_col = "".join(list(self.odata.stratification_column.keys()))
        sample_col = self.odata.sample_id
        row_col = self.odata.row_id
        feature_cols = self.odata.features
        target_assignments = self.odata.target_asignments
        relevant_cols = list(
            set(
                feature_cols
                + target_cols
                + [sample_col, row_col, stratification_col]
                + ["group_features", "group_sample_and_features"]
            )
        )
        data_clean_df = self.odata.data[relevant_cols]

        # select datasplit column
        datasplit_col = self.odata.datasplit_type
        if datasplit_col == "sample":
            datasplit_col = self.odata.sample_id

        data_splits = DataSplit(
            dataset=data_clean_df,
            datasplit_col=datasplit_col,
            seed=self.oconfig.datasplit_seed_outer,
            num_folds=self.oconfig.n_folds_outer,
            stratification_col=stratification_col,
        ).get_datasplits()

        for key, value in data_splits.items():
            # create path for experiment
            path_experiment = Path(f"experiment{key}")
            path_study.joinpath(path_experiment).mkdir(parents=True, exist_ok=True)
            self.experiments.append(
                OctoExperiment(
                    id=str(key),
                    experiment_id=int(key),
                    sequence_item_id=-1,  # indicating base experiment
                    path_sequence_item=path_experiment,  # indicating base experiment
                    config=asdict(self.oconfig),
                    datasplit_column=datasplit_col,
                    row_column=row_col,
                    feature_columns=feature_cols,
                    stratification_column=stratification_col,
                    target_assignments=target_assignments,
                    data_traindev=value["train"],
                    data_test=value["test"],
                )
            )

        print()
        print()

    def run_outer_experiments(self):
        """Run outer experiments."""
        # send self.experiments() to manager
        self.manager = OctoManager(self.experiments, self.oconfig)
        self.manager.run_outer_experiments()
