"""Utils."""

import random
from typing import Optional

import numpy as np
import pandas as pd
from attrs import Factory, define, field, validators
from sklearn.model_selection import KFold, StratifiedKFold


@define
class DataSplit:
    """Data Split.

    We don't use groupKFold as it does not offer the shuffle option.
    The StratifiedGroupKfold might work as an alternative (check examples).
    StratifiedGroupKfold is not available for sklearn 0.24.3
    which is required for Auto-Sklearn 0.15.
    stratification_col: contains the group info used for stratification
    datasplit_col: contains group info on samples. Each group goes either
    into the training or the test dataset.
    """

    datasplit_col: str = field(validator=[validators.instance_of(str)])
    seed: int = field(validator=[validators.instance_of(int)])
    num_folds: int = field(validator=[validators.instance_of(int)])
    dataset: pd.DataFrame = field(validator=[validators.instance_of(pd.DataFrame)])
    stratification_col: Optional[str] = field(
        default=Factory(lambda: None),
        validator=validators.optional(validators.instance_of(str)),
    )

    def __attrs_post_init__(self):
        # reset index
        self.dataset.reset_index(drop=True, inplace=True)
        print(self.dataset)

    def get_datasplits(self):
        """Get datasplits."""
        # set seeds for reproducibility
        random.seed(0)
        np.random.seed(0)

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
