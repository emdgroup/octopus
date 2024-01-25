"""OctoML module."""
from pathlib import Path

import pandas as pd
from attrs import asdict, define, field, validators
from sklearn.model_selection import KFold, StratifiedKFold

from octopus.config import OctoConfig
from octopus.data import OctoData
from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager


@define
class OctoML:
    """OctoML."""

    odata: OctoData = field(validator=[validators.instance_of(OctoData)])
    oconfig: OctoConfig = field(validator=[validators.instance_of(OctoConfig)])
    experiments: list = field(init=False, default=[])
    manager: OctoManager = field(init=False, default=None)

    def create_outer_experiments(self):
        """Create outer experiments."""
        # create study folder
        path_study = Path(self.oconfig.output_path, self.oconfig.study_name)
        path_study.mkdir(parents=True, exist_ok=not self.oconfig.production_mode)
        print("Path to study:", path_study)

        # create subfolders
        for subdir in ["data", "config", "tmp"]:
            path_sub = path_study.joinpath(subdir)
            path_sub.mkdir(parents=False, exist_ok=not self.oconfig.production_mode)

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
                + [sample_col, row_col, "group_features", "group_sample_and_features"]
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
            num_folds=self.oconfig.k_outer,
            stratification_col=stratification_col,
        ).get_datasplits()

        for key, value in data_splits.items():
            # create path for experiment
            path_experiment = Path(f"experiment{key}")
            path_study.joinpath(path_experiment).mkdir(
                parents=True, exist_ok=not self.oconfig.production_mode
            )
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


@define
class DataSplit:
    """Data Split.

    We don't use groupKFold as it does not offer the shuffle option.
    StratifiedGroupKfold would work but is not available for sklearn 0.24.3
    which is required for Auto-Sklearn 0.15.
    stratification_col: contains the group info used for stratification
    datasplit_col: contains group info on samples. Each group goes either
    into the training or the test dataset.
    """

    datasplit_col: str = field(validator=[validators.instance_of(str)])
    seed: int = field(validator=[validators.instance_of(int)])
    num_folds: int = field(validator=[validators.instance_of(int)])
    dataset: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    stratification_col: str = field(default="", validator=[validators.instance_of(str)])

    def __attrs_post_init__(self):
        # reset index
        self.dataset.reset_index(drop=True, inplace=True)

    def get_datasplits(self):
        """Get datasplits."""
        # Allow for grouped rows as defined in datasplit_col
        # The split is done on dataset_unique, with an reset index.
        # This ensures that we split by group.
        dataset_unique = self.dataset.drop_duplicates(
            subset=self.datasplit_col, keep="first", inplace=False
        )
        dataset_unique.reset_index(drop=True, inplace=True)

        print(
            f"""Number of unique groups (as in column: {self.datasplit_col}):"""
            f"""{len(dataset_unique)}"""
        )
        print("Number of rows in dataset:", len(self.dataset))
        print()
        print("Creating data splits....")

        # StratifiedKfold or Kfold
        if self.stratification_col:
            print("Data split: stratified KFold")
            kf = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            if dataset_unique[self.stratification_col].dtype.kind not in "iub":
                raise ValueError(
                    "Stratification column is of wrong type (reg.: bool,int)"
                )

            stratification_target = dataset_unique[self.stratification_col].astype(int)
        else:
            print("Data split: KFold (unstratified)")
            kf = KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            stratification_target = None

        data_splits = dict()
        all_test_indices = list()
        all_test_groups = list()
        print("Number of splits:", self.num_folds)
        # split based on dataset_unique
        for num_split, (train_ind, test_ind) in enumerate(
            kf.split(dataset_unique, stratification_target)
        ):
            print("##### split number:", num_split)
            # train and test groups
            groups_train = set(dataset_unique.iloc[train_ind][self.datasplit_col])
            groups_test = set(dataset_unique.iloc[test_ind][self.datasplit_col])
            assert groups_train.intersection(groups_test) == set()
            all_test_groups.extend(list(groups_test))

            # take groups and partition self.dataset based on groups
            # This makes sure that samples of the same group are in
            # the same partition. Stratification may not be optimal.
            partition_train = self.dataset[
                self.dataset[self.datasplit_col].isin(groups_train)
            ]
            partition_test = self.dataset[
                self.dataset[self.datasplit_col].isin(groups_test)
            ]
            assert (
                set(partition_train.index).intersection(partition_test.index) == set()
            )
            all_test_indices.extend(partition_test.index.tolist())

            # reset partition indices
            partition_train.reset_index(drop=True, inplace=True)
            partition_test.reset_index(drop=True, inplace=True)

            print("train, number of rows:", len(partition_train))
            print("train, number of groups:", len(set(groups_train)))
            print("test, number of rows:", len(partition_test))
            print("test, number of groups:", len(set(groups_test)))

            data_splits[num_split] = {
                "test": partition_test,
                "train": partition_train,
            }

        # checking datasplit groups
        assert len(all_test_groups) == len(set(all_test_groups))
        assert (
            len(
                set(self.dataset[self.datasplit_col]).symmetric_difference(
                    set(all_test_groups)
                )
            )
            == 0
        )

        # checking datasplit indices
        assert len(all_test_indices) == len(set(all_test_indices))
        assert len(self.dataset) == len(all_test_indices)

        return data_splits
