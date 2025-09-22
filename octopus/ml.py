"""OctoML module."""

import shutil
import sys
from pathlib import Path

import pandas as pd
from attrs import Factory, asdict, define, field, validators

from octopus import OctoData
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.config.core import OctoConfig
from octopus.data.imputer import impute_mice, impute_simple
from octopus.experiment import OctoExperiment
from octopus.manager import OctoManager
from octopus.utils import DataSplit

from .logger import LogGroup, get_logger

logger = get_logger()


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
    config_manager: ConfigManager = field(validator=[validators.instance_of(ConfigManager)])
    config_sequence: ConfigSequence = field(validator=[validators.instance_of(ConfigSequence)])
    configs: OctoConfig = field(default=None)
    experiments: list = field(default=Factory(list))
    manager: OctoManager = field(init=False, default=None)

    def __attrs_post_init__(self):
        # initialization here due to "Python immutable default"
        self.configs = OctoConfig(
            study=self.config_study,
            manager=self.config_manager,
            sequence=self.config_sequence,
        )
        # initialize ray

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

        # save data report
        self._save_data_report(path_study)

        # check data for critical issues
        self._check_for_data_issues()

        # get clean dataset only with relevant columns for ML
        # data_clean_df = self._get_dataset_with_relevant_columns()
        data_clean_df = self.data.data[self.data.relevant_columns]
        # get datasplit column
        datasplit_col = self.data.sample_id if self.data.datasplit_type == "sample" else self.data.datasplit_type

        # create datasplits for outer experiments
        data_splits = DataSplit(
            dataset=data_clean_df,
            datasplit_col=datasplit_col,
            seeds=[self.configs.study.datasplit_seed_outer],
            num_folds=self.configs.study.n_folds_outer,
            stratification_col=self.data.stratification_column,
        ).get_datasplits()

        # create experiments from the datasplit
        self._create_experiments(path_study, data_splits, datasplit_col)

    def _save_data_report(self, path_study: Path) -> None:
        """Save data report."""
        report_df = self.data.report
        report_df.to_csv(path_study.joinpath("data", "data_health_report.csv"), sep=";")

    def _check_for_data_issues(self):
        """Check for data issues."""
        logger.set_log_group(LogGroup.DATA_HEALTH_REPORT)

        critical_issues = False
        warning_issues = False

        df_sorted = self.data.report.sort_values(
            "Severity",
            key=lambda x: pd.Categorical(x, categories=["Info", "Warning", "Critical"], ordered=True),
        )

        for _, issue in df_sorted.iterrows():
            if issue.Severity == "Info":
                logger.info(f"{issue.Category} - {issue['Issue Type']}")
            elif issue.Severity == "Warning":
                logger.warning(f"{issue.Category} - {issue['Issue Type']}")
                warning_issues = True
            elif issue.Severity == "Critical":
                logger.critical(f"{issue.Category} - {issue['Issue Type']}")
                critical_issues = True

        if critical_issues:
            raise Exception(
                f"Critical data issues have been detected. "
                f"Please check the details in the following file: "
                f"{Path(self.configs.study.path, self.configs.study.name)}"
                f"/data/data_health_report.csv"
            )

        if warning_issues:
            if not self.config_study.ignore_data_health_warning:
                raise Exception(
                    f"Data issues have been detected. "
                    f"Please check the details in the following file: "
                    f"{Path(self.configs.study.path, self.configs.study.name)}"
                    f"/data/data_health_report.csv\n\n"
                    f"To proceed despite these warnings"
                    f", set `ignore_data_health_warning` "
                    f"to True in `ConfigStudy`."
                )

    def _handle_existing_study_path(self, path_study: Path) -> None:
        """Handle the existing study path.

        Args:
            path_study: The path to the study directory.
        """
        if path_study.exists():
            if not self.configs.study.silently_overwrite_study:
                confirmation = input("Study exists, do you want to continue? (yes/no): ")
                if confirmation.strip().lower() != "yes":
                    print("Exiting...")
                    sys.exit()
                print("Continuing...")

            if self.configs.study.start_with_empty_study:
                print("Overwriting existing study....")
                shutil.rmtree(path_study)
            else:
                print("Resume existing study....")

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
        self.data.save_attributes_to_parquet(data_path / "data.parquet")
        self.data.to_pickle(data_path / "data.pkl")

        # Save config files
        config_path = path_study / "config"
        pd.DataFrame(
            [(k, str(v)) for k, v in asdict(self.configs.study).items()],
            columns=["Parameter", "Value"],
        ).to_parquet(config_path.joinpath("config_study.parquet"))

        pd.DataFrame(
            [(k, str(v)) for k, v in asdict(self.configs.manager).items()],
            columns=["Parameter", "Value"],
        ).to_parquet(config_path.joinpath("config_manager.parquet"))

        pd.DataFrame(
            [(k, str(v)) for k, v in asdict(self.configs.sequence).items()],
            columns=["Parameter", "Value"],
        ).to_parquet(config_path.joinpath("config_sequence.parquet"))

        self.configs.to_pickle(config_path / "config.pkl")

    def _impute_dataset(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute dataset if missing values are present.

        Parameters:
            train_df: The training dataset.
            test_df: The testing dataset.
            feature_columns: List of feature column names to impute.

        Returns:
            tuple: A tuple containing the imputed (or original) train and test datasets.
        """
        imputation_method = self.configs.study.imputation_method

        # Check for missing values in the feature columns
        missing_in_train = train_df[feature_columns].isna().any().any()
        missing_in_test = test_df[feature_columns].isna().any().any()

        if not missing_in_train and not missing_in_test:
            # No missing values, return original datasets
            return train_df, test_df

        logger.info("Imputing data .....")
        # Perform imputation if missing values are present
        if imputation_method == "mice":
            logger.info("MICE imputation")
            imputed_train_df, imputed_test_df = impute_mice(train_df, test_df, feature_columns)
        else:
            logger.info("Simple imputation")
            imputed_train_df, imputed_test_df = impute_simple(train_df, test_df, feature_columns, imputation_method)

        # Assert that there are no NaNs in the imputed data
        assert not imputed_train_df[feature_columns].isna().any().any(), "NaNs present in imputed train_df"
        assert not imputed_test_df[feature_columns].isna().any().any(), "NaNs present in imputed test_df"

        return imputed_train_df, imputed_test_df

    def _create_experiments(self, path_study: Path, data_splits: dict, datasplit_col: str) -> None:
        """Create the experiments based on the data splits.

        Args:
            path_study: The path to the study folder.
            data_splits: The dictionary containing data splits.
            datasplit_col: The column used for data splitting.
        """
        for key, value in data_splits.items():
            path_experiment = Path(f"experiment{key}")
            path_study.joinpath(path_experiment).mkdir(parents=True, exist_ok=True)
            # Changed: Imputation takes place in training
            # feature_columns = self.data.feature_columns
            # traindev_df, test_df = self._impute_dataset(value["train"], value["test"], feature_columns)
            self.experiments.append(
                OctoExperiment(
                    id=str(key),
                    experiment_id=int(key),
                    sequence_id=None,  # indicating base experiment
                    input_sequence_id=None,  # indicating base experiment
                    path_sequence_item=None,  # indicating base experiment
                    configs=self.configs,
                    datasplit_column=datasplit_col,
                    row_column=self.data.row_id,
                    feature_columns=self.data.feature_columns,
                    stratification_column=self.data.stratification_column,
                    target_assignments=self.data.target_assignments,
                    data_traindev=value["train"],
                    data_test=value["test"],
                )
            )

    def run_outer_experiments(self):
        """Run outer experiments."""
        # send self.experiments() to manager
        self.manager = OctoManager(self.experiments, self.configs)
        self.manager.run_outer_experiments()

    def run_study(self):
        """Run study."""
        self.create_outer_experiments()
        self.run_outer_experiments()
