"""OctoML module."""

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from attrs import define, field, validators

from octopus import OctoData
from octopus.config import ConfigManager, ConfigSequence, ConfigStudy
from octopus.config.core import OctoConfig
from octopus.data.imputer import impute_mice, impute_simple
from octopus.experiment import OctoExperiment
from octopus.logger import configure_logging
from octopus.manager import OctoManager
from octopus.utils import DataSplit

configure_logging()


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

        # save data report
        self._save_data_report(path_study)

        # check data for critical issues
        self._check_for_data_issues()

        # get clean dataset only with relevant columns for ML
        # data_clean_df = self._get_dataset_with_relevant_columns()
        data_clean_df = self.data.data[self.data.relevant_columns]
        # get datasplit column
        datasplit_col = (
            self.data.sample_id
            if self.data.datasplit_type == "sample"
            else self.data.datasplit_type
        )

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
        report = self.data.report
        print("Report")
        report_df = report.create_df()
        report_df.to_csv(path_study.joinpath("data", "data_health_report.csv"))

    def _check_for_data_issues(self):
        targets = self.data.target_columns
        features = self.data.feature_columns
        stratification = self.data.stratification_column

        report_cols = self.data.report.columns
        report_cols_feat_tar = {
            key: val
            for key, val in report_cols.items()
            if key in targets or key in features
        }
        report_col_stratification = {
            key: val for key, val in report_cols.items() if key == stratification
        }
        error_messages = []
        warning_messages = []

        # Check for NaNs
        for col in report_cols:
            missing_share = report_cols[col].get("missing values share", None)
            if col in targets:
                if missing_share is not None and missing_share > 0:
                    error_messages.append("NaN values detected in target columns.")
            if col == self.data.stratification_column:
                if missing_share is not None and missing_share > 0:
                    error_messages.append(
                        "NaN values detected in stratification column."
                    )
            if col == self.data.row_id:
                if missing_share is not None and missing_share > 0:
                    error_messages.append("NaN values detected in row_id.")
            if col == self.data.sample_id:
                if missing_share is not None and missing_share > 0:
                    error_messages.append("NaN values detected in sample_id.")

            if col in features:
                if missing_share is not None and missing_share > 0.25:  # original: 0.2
                    error_messages.append(
                        "Columns with high missing share detected in feature columns."
                    )
                    break

        # Check for object type columns
        if any(val.get("iu dtype", None) for val in report_col_stratification.values()):
            error_messages.append(
                "Stratification columns must be of type integer or uint."
            )

        # Check for infinity values
        if any(val.get("infinity values share", 0) > 0 for val in report_cols.values()):
            error_messages.append("Inf values in dataset")

        # Check unique row id
        if (
            self.data.row_id in report_cols
            and report_cols[self.data.row_id].get("unique row id", None) is not None
        ):
            error_messages.append("Values in the Row ID must be unique.")

        # Check few integer values
        if any(
            val.get("unique_int_values'", None) is not None
            for val in report_cols_feat_tar.values()
        ):
            warning_messages.append(
                """Some columns have few unique integer values.
                Consider using dummy encoding."""
            )

        # feature-feature corrlation
        if any(
            any(
                key in val
                for key in [
                    "high feature correlation (pearson)",
                    "high feature correlation (spearman)",
                ]
            )
            for val in report_cols_feat_tar.values()
        ):
            warning_messages.append("""Some features are highly correlated.""")

        # Log all warning and errors
        if warning_messages:
            for message in warning_messages:
                logging.warning(message)

        if error_messages:
            for message in error_messages:
                logging.error(message)

        # raise Exception
        if error_messages:
            raise Exception(
                f"Critical data issues have been detected. "
                f"Please check the details in the following file: "
                f"{Path(self.configs.study.path, self.configs.study.name)}"
                f"/data/data_health_report.csv"
            )

        if warning_messages:
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

        print("Imputing data .....")
        # Perform imputation if missing values are present
        if imputation_method == "mice":
            imputed_train_df, imputed_test_df = impute_mice(
                train_df, test_df, feature_columns
            )
        else:
            imputed_train_df, imputed_test_df = impute_simple(
                train_df, test_df, feature_columns, imputation_method
            )

        # Assert that there are no NaNs in the imputed data
        assert (
            not imputed_train_df[feature_columns].isna().any().any()
        ), "NaNs present in imputed train_df"
        assert (
            not imputed_test_df[feature_columns].isna().any().any()
        ), "NaNs present in imputed test_df"

        return imputed_train_df, imputed_test_df

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
            traindev_df, test_df = self._impute_dataset(
                value["train"], value["test"], feature_columns
            )

            self.experiments.append(
                OctoExperiment(
                    id=str(key),
                    experiment_id=int(key),
                    sequence_id=-1,  # indicating base experiment
                    input_sequence_id=-1,  # indicating base experiment
                    path_sequence_item=path_experiment,  # indicating base experiment
                    configs=self.configs,
                    datasplit_column=datasplit_col,
                    row_column=self.data.row_id,
                    feature_columns=self.data.feature_columns,
                    stratification_column=self.data.stratification_column,
                    target_assignments=self.data.target_assignments,
                    data_traindev=traindev_df,
                    data_test=test_df,
                )
            )

    def run_outer_experiments(self):
        """Run outer experiments."""
        # send self.experiments() to manager
        self.manager = OctoManager(self.experiments, self.configs)
        self.manager.run_outer_experiments()
