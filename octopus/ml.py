"""OctoML module."""

import shutil
import sys
from pathlib import Path

import miceforest as mf
import pandas as pd
from attrs import define, field, validators

from octopus import OctoData
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.config.core import OctoConfig
from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager
from octopus.utils import DataSplit


@define
class OctoML:
    """OctoML class responsible for managing experiments.

    Attributes:
        data (OctoData): The data used in the experiments.
        config_study (ConfigStudy): Configuration for the study.
        config_manager (ConfigManager): Configuration for the manager.
        config_sequence (ConfigSequence): Configuration for the sequence.
        configs (OctoConfig): The configuration settings for the experiments.
        experiments (List): A list to store experiment details.
        manager (Optional[OctoManager]): An optional manager for the experiments.
    """

    data: OctoData = field(validator=[validators.instance_of(OctoData)])
    config_study: ConfigStudy = field(validator=[validators.instance_of(ConfigStudy)])
    config_manager: ConfigManager = field(
        validator=[validators.instance_of(ConfigManager)]
    )
    config_sequence: ConfigSequence = field(
        validator=[validators.instance_of(ConfigSequence)]
    )
    configs: OctoConfig = field(default=None)
    experiments: list = field(init=False)
    manager: OctoManager = field(init=False, default=None)

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.experiments = []
        self.configs = OctoConfig(
            study=self.config_study,
            manager=self.config_manager,
            sequence=self.config_sequence,
        )

    def create_outer_experiments(self) -> None:
        """Create the outer experiments structure.

        This method creates the study folder and required subfolders. It handles
        existing study paths based on the configuration settings.
        """
        # create study path
        path_study = Path(self.configs.study.path, self.configs.study.name)

        # Handle existing study path
        self._handle_existing_study_path(path_study)

        # Create study folder and subfolders
        self._create_folders(path_study)

        # Save data and config files
        self._save_files(path_study)

        # get clean dataset only with relevant columns for ML
        data_clean_df = self._get_dataset_with_relevant_columns()

        # get datasplit column
        datasplit_col = (
            self.data.sample_id
            if self.data.datasplit_type == "sample"
            else self.data.datasplit_type
        )

        # create datasplits for outer experiments
        # why is the stratification col here a string and before a List
        data_splits = DataSplit(
            dataset=data_clean_df,
            datasplit_col=datasplit_col,
            seed=self.configs.study.datasplit_seed_outer,
            num_folds=self.configs.study.n_folds_outer,
            stratification_col="".join(self.data.stratification_column),
        ).get_datasplits()

        # create experiments from the datasplit
        self._create_experiments(path_study, data_splits, datasplit_col)

    def _handle_existing_study_path(self, path_study: Path) -> None:
        """Handle the existing study path.

        Args:
            path_study: The path to the study directory.
        """
        if path_study.exists():
            if not self.configs.study.silently_overwrite_study:
                confirmation = input(
                    "Study exists, do you want to continue (resume)? (yes/no): "
                )
                if confirmation.strip().lower() != "yes":
                    print("Exiting...")
                    sys.exit()
                print("Continuing...")

            if self.configs.study.start_with_empty_study:
                shutil.rmtree(path_study)

    def _create_folders(self, path_study: Path) -> None:
        """Create study folder and subdirectories.

        Args:
            path_study: The path to the study directory.
        """
        # Create main study directory
        path_study.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["data", "config"]:
            path_sub = path_study / subdir
            path_sub.mkdir(parents=False, exist_ok=True)

    def _save_files(self, path_study: Path) -> None:
        """Save data and config files to the respective subdirectories.

        Args:
            path_study: The path to the study directory.
        """
        # Save data files
        data_path = path_study / "data"
        self.data.to_pickle(data_path / "data.pkl")

        # Save config files
        config_path = path_study / "config"
        # Uncomment if JSON is needed
        # self.config.to_json(config_path / "config.json")
        self.configs.to_pickle(config_path / "config.pkl")

    def _get_dataset_with_relevant_columns(self) -> pd.DataFrame:
        """Get the dataset only with relevant columns for the ML.

        Returns:
        DataFrame: DataFrame with the relevant columns.
        """
        relevant_cols = list(
            set(
                self.data.feature_columns
                + self.data.target_columns
                + [
                    self.data.sample_id,
                    self.data.row_id,
                    "group_features",
                    "group_sample_and_features",
                ]
            )
        )
        stratification_col = "".join(self.data.stratification_column)
        if stratification_col != "":
            relevant_cols.append(stratification_col)
            # keep columns unique, if target columns eqals stratification column
            relevant_cols = list(set(relevant_cols))

        return self.data.data[relevant_cols]

    def _impute_dataset(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list
    ) -> tuple[pd.DataFrame, pd.DataFrame, mf.ImputationKernel]:
        """Impute training and test datasets using mice-forest."""
        # Check for missing values in the specified feature columns
        train_has_missing = train_df[feature_columns].isna().any().any()
        test_has_missing = test_df[feature_columns].isna().any().any()

        print("Missing value in train:", train_has_missing)
        print("Missing value in test:", test_has_missing)

        if not train_has_missing and not test_has_missing:
            # No missing values in both datasets; return them unchanged
            return train_df, test_df, None
        else:
            # Set parameters for imputation
            print("Imputing datasets...")
            num_iterations = 10  # Number of MICE iterations

            # Fit the imputation model on the training data
            kernel = mf.ImputationKernel(
                train_df[feature_columns],
                variable_schema=feature_columns,  # train on all columns
                save_all_iterations_data=True,
                random_state=42,  # For reproducibility
            )

            # Run the MICE algorithm for the specified number of iterations
            kernel.mice(num_iterations)

            # Transform the training data
            imputed_train_features = kernel.complete_data(dataset=0)

            # Replace the original feature columns with the imputed values
            imputed_train_df = train_df.copy()
            imputed_train_df[feature_columns] = imputed_train_features[feature_columns]

            if test_has_missing:
                # Impute the test data using the model fitted on training data
                imputed_test = kernel.impute_new_data(
                    test_df[feature_columns],
                    datasets=[0],  # Use dataset index 0
                )
                imputed_test_features = imputed_test.complete_data(dataset=0)

                # Replace the original feature columns with the imputed values
                imputed_test_df = test_df.copy()
                imputed_test_df[feature_columns] = imputed_test_features[
                    feature_columns
                ]
            else:
                # No missing values in test dataset; use it as is
                imputed_test_df = test_df.copy()

            return imputed_train_df, imputed_test_df, kernel

    def _create_experiments(
        self, path_study: Path, data_splits: dict, datasplit_col: str
    ) -> None:
        """Create the experiments based on the data splits.

        Args:
            path_study: The path to the study folder.
            data_splits: The dictionary containing data splits.
            datasplit_col: The column used for data splitting.
        """
        for key, value in data_splits.items():
            path_experiment = Path(f"experiment{key}")
            path_study.joinpath(path_experiment).mkdir(parents=True, exist_ok=True)
            # impute datasets
            feature_columns = self.data.feature_columns
            traindev_df, test_df, kernel = self._impute_dataset(
                value["train"], value["test"], feature_columns
            )

            self.experiments.append(
                OctoExperiment(
                    id=str(key),
                    experiment_id=int(key),
                    sequence_item_id=-1,  # indicating base experiment
                    path_sequence_item=path_experiment,  # indicating base experiment
                    configs=self.configs,
                    datasplit_column=datasplit_col,
                    row_column=self.data.row_id,
                    feature_columns=feature_columns,
                    stratification_column="".join(self.data.stratification_column),
                    target_assignments=self.data.target_assignments,
                    data_traindev=traindev_df,
                    data_test=test_df,
                    imputation_kernel=kernel,
                )
            )

    def run_outer_experiments(self):
        """Run outer experiments."""
        # send self.experiments() to manager
        self.manager = OctoManager(self.experiments, self.configs)
        self.manager.run_outer_experiments()
