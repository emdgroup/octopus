"""OctoML module."""
from pathlib import Path

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
        path_study.mkdir(parents=True, exist_ok=False)
        print("Path to study:", path_study)

        # create subfolders
        for subdir in ["data", "config", "tmp", "experiments"]:
            path_sub = path_study.joinpath(subdir)
            path_sub.mkdir(parents=False, exist_ok=False)

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

        # allow for multiple rows per sample_ID
        data_clean_unique_df = data_clean_df.drop_duplicates(
            subset=datasplit_col, keep="first", inplace=False
        )
        print(
            f"""Number of unique sample groups ({datasplit_col}):
            {len(data_clean_unique_df)}"""
        )
        print("")
        print("Creating outer folds....")

        if stratification_col:
            print("Data split: stratified KFold")
            split_outer = StratifiedKFold(
                n_splits=self.oconfig.k_outer,
                shuffle=True,
                random_state=self.oconfig.datasplit_seed_outer,
            )
            stratification_target = data_clean_unique_df[stratification_col].astype(int)
        else:
            print("Data split: KFold (unstratified)")
            split_outer = KFold(
                n_splits=self.oconfig.k_outer,
                shuffle=True,
                random_state=self.oconfig.datasplit_seed_outer,
            )
            stratification_target = None

        print("Number of outer base experiments:", self.oconfig.k_outer)
        for num_out, (traindev_ind, test_ind) in enumerate(
            split_outer.split(data_clean_unique_df, stratification_target)
        ):
            print("##### outer fold:", num_out)

            # fold traindev
            subjid_traindev = data_clean_unique_df.iloc[traindev_ind][
                sample_col
            ]  # get all subjects in this fold
            fold_traindev = data_clean_df[
                data_clean_df[sample_col].isin(subjid_traindev)
            ]  # put all all data for those subjects into fold_train
            fold_traindev.reset_index(inplace=True, drop=True)
            print("traindev: number of samples", len(subjid_traindev))

            # fold test
            subjid_test = data_clean_unique_df.iloc[test_ind][
                sample_col
            ]  # get all subjects in this fold
            fold_test = data_clean_df[
                data_clean_df[sample_col].isin(subjid_test)
            ]  # put all all data for those subjects into fold_train
            fold_test.reset_index(inplace=True, drop=True)
            print("test: number of samples", len(subjid_test))

            self.experiments.append(
                OctoExperiment(
                    id=str(num_out),
                    config=asdict(self.oconfig),
                    sample_column=sample_col,
                    row_column=row_col,
                    feature_columns=feature_cols,
                    stratification_column=stratification_col,
                    target_assignments=target_assignments,
                    data_traindev=fold_traindev,
                    data_test=fold_test,
                )
            )

        print()
        print()

    def run_outer_experiments(self):
        """Run outer experiments."""
        # send self.experiments() to manager
        self.manager = OctoManager(self.experiments, self.oconfig)
        self.manager.run_outer_experiments()
