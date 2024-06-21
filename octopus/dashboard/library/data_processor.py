"""Dashboard data processor."""

import pickle
import re
from pathlib import Path

import optuna
import pandas as pd
from attrs import asdict, define, field

from octopus.data import OctoData
from octopus.modules.metrics import metrics_inventory


@define
class EDADataProcessor:
    """Load data for EDA."""

    octo_data: OctoData | Path = field(default=None)

    def __attrs_post_init__(self):
        # load octo data from file if Path is provided
        if isinstance(self.octo_data, Path):
            file = list(self.octo_data.glob("**/data.pkl"))[0]
            with open(file, "rb") as f:
                self.octo_data = pickle.load(f)

    def get_dataset(self) -> tuple:
        """Get data table with information on data types and column handling."""
        data_info_dicts = (
            [
                {"Type": "Target", "Column": target}
                for target in self.octo_data.target_columns
            ]
            + [
                {"Type": "Feature", "Column": feature}
                for feature in self.octo_data.feature_columns
            ]
            + [{"Type": "Row_ID", "Column": self.octo_data.row_id}]
            + [{"Type": "Datasplit", "Column": self.octo_data.datasplit_type}]
            + [{"Type": "Sample_ID", "Column": self.octo_data.sample_id}]
        )

        df_data_info = pd.DataFrame(data_info_dicts)
        if self.octo_data.data.shape[1] > 500:
            print("DataFrame has to many col. Only take the first 50.")
            return self.octo_data.data.iloc[:, :50].reset_index(drop=True), df_data_info

        return self.octo_data.data.reset_index(drop=True), df_data_info

    def create_eda_data_description(self) -> pd.DataFrame:
        """Create a DataFrame describing the Exploratory Data Analysis (EDA) metrics."""
        if isinstance(self.octo_data.features, dict):
            features = list(self.octo_data.features.keys())
        else:
            features = self.octo_data.features

        if isinstance(self.octo_data.targets, dict):
            targets = list(self.octo_data.targets.keys())
        else:
            targets = self.octo_data.targets

        # List of tuples, each containing a description and its corresponding value
        metrics = [
            ("# Rows", self.octo_data.data.shape[0]),
            ("# Columns", self.octo_data.data.shape[1]),
            ("# Targets", len(targets)),
            ("# Features", len(features)),
            (
                "# Unique samples",
                self.octo_data.data[self.octo_data.sample_id].nunique(),
            ),
            ("# Unique features", self.octo_data.data["group_features"].nunique()),
            (
                "# Unique features and samples",
                self.octo_data.data["group_sample_and_features"].nunique(),
            ),
            ("Ratio Rows to Features", self.octo_data.data.shape[0] / len(features)),
            (
                "Feature columns with nan",
                self.octo_data.data[features].isnull().any().sum(),
            ),
        ]

        # Create and return the DataFrame specifying column names
        df_description = pd.DataFrame(metrics, columns=["Description", "Value"])
        return df_description

    def get_eda_column_info(self) -> pd.DataFrame:
        """Get column information for dataframe."""
        dict_col = []
        for col in self.octo_data.data.columns:
            if col in self.octo_data.targets:
                dict_col.append({"Column": col, "Type": "Target"})
            elif col in self.octo_data.feature_columns:
                dict_col.append({"Column": col, "Type": "Feature"})
            elif col in self.octo_data.sample_id:
                dict_col.append({"Column": col, "Type": "Sample_ID"})
            else:
                dict_col.append({"Column": col, "Type": "Info"})
        return pd.DataFrame(dict_col)


@define
class ResultsDataProcessor:
    """Load results data."""

    study_path: str = field()

    def get_predictions(self) -> tuple:
        """Get predictions."""
        # get experiment files
        experiment_files = list(self.study_path.glob("**/exp*.pkl"))

        df_predictions = pd.DataFrame()
        dict_scores = []
        for file in list(experiment_files):
            with open(file, "rb") as f:
                exp = pickle.load(f)
                default_target_column = exp.target_assignments["default"]
                for split in exp.predictions:
                    for dataset in exp.predictions[split]:
                        # get predictions
                        df_temp = exp.predictions[split][dataset]
                        df_temp["experiment_id"] = exp.experiment_id
                        df_temp["sequence_id"] = exp.sequence_item_id
                        df_temp["split_id"] = split
                        df_temp["dataset"] = dataset
                        df_predictions = pd.concat([df_predictions, df_temp])

                        # calculate scores
                        for metric, value in metrics_inventory.items():
                            if value.get("ml_type") == exp.ml_type:
                                if exp.ml_type == "classification":
                                    prediction = exp.predictions[split][dataset][
                                        "prediction"
                                    ].astype(int)
                                dict_socre_temp = {
                                    "experiment_id": exp.experiment_id,
                                    "sequence_id": exp.sequence_item_id,
                                    "split": split,
                                    "testset": dataset,
                                    "metric": metric,
                                    "score": value["method"](
                                        exp.predictions[split][dataset][
                                            default_target_column
                                        ],
                                        prediction,
                                    ),
                                }
                                dict_scores.append(dict_socre_temp)

        df_predictions = df_predictions.reset_index(drop=True)
        df_scores = (
            pd.DataFrame(dict_scores)
            .sort_values(by=["experiment_id", "sequence_id", "split"])
            .reset_index(drop=True)
        )

        return (df_predictions, df_scores)

    def get_optuna_trials(self) -> pd.DataFrame:
        """Get data from optuna."""
        # get optuna files
        optuna_files = list(self.study_path.glob("**/optuna*.db"))

        dict_optuna = []
        for file in list(optuna_files):
            match_experiment = re.search(r"experiment(\d+)", str(file))
            match_sequence = re.search(r"sequence(\d+)", str(file))
            split_id = str(file).split("_")[-1].split(".db")[0]

            study = optuna.study.load_study(
                study_name=file.stem, storage=f"sqlite:///{file}"
            )

            for trial in study.get_trials():
                for name, _ in trial.distributions.items():
                    if name == "ml_model_type":
                        continue
                    if "ml_model_type" in trial.params:
                        model_type = trial.params["ml_model_type"]
                    else:
                        model_type = trial.user_attrs["config_training"][
                            "ml_model_type"
                        ]

                    dict_optuna.append(
                        {
                            "experiment_id": int(match_experiment.group(1)),
                            "sequence_id": int(match_sequence.group(1)),
                            "split_id": split_id,
                            "trial": trial.number,
                            "value": trial.value,
                            "model_type": model_type,
                            "hyper_param": name.split(f"_{model_type}")[0],
                            "param_value": trial.params[name],
                        }
                    )

        return pd.DataFrame(dict_optuna)

    def get_configs(self) -> tuple:
        """Get configurations."""
        # get config files
        config_files = list(self.study_path.glob("**/config.pkl"))

        with open(config_files[0], "rb") as f:
            config_file = pickle.load(f)

            # manager config
            df_config_manager = (
                pd.DataFrame.from_dict(
                    {key: str(value) for key, value in config_file.cfg_manager.items()},
                    orient="index",
                )
                .reset_index()
                .set_axis(["Parameter", "Value"], axis="columns")
            )

            # sequence config
            df_config_sequence = pd.DataFrame()
            for idx, sequence in enumerate(config_file.cfg_sequence):
                df_config_sequence_temp = pd.DataFrame.from_dict(
                    {
                        key: (
                            str(value)
                            if not isinstance(value, (int, float, str))
                            else value
                        )
                        for key, value in sequence.items()
                    },
                    orient="index",
                )
                df_config_sequence_temp["sequence_id"] = idx

                df_config_sequence = pd.concat(
                    [df_config_sequence, df_config_sequence_temp]
                )
            df_config_sequence = df_config_sequence.reset_index().set_axis(
                ["Parameter", "Value", "sequence_id"], axis="columns"
            )

            # study config
            df_config_study = (
                pd.DataFrame.from_dict(
                    {
                        key: (
                            str(value)
                            if not isinstance(value, (int, float, str))
                            else value
                        )
                        for key, value in asdict(config_file).items()
                        if key not in ["cfg_manager", "cfg_sequence"]
                    },
                    orient="index",
                )
                .reset_index()
                .set_axis(["Parameter", "Value"], axis="columns")
            )
        return df_config_study, df_config_manager, df_config_sequence

    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances."""
        # get experiment files
        experiment_files = list(self.study_path.glob("**/exp*.pkl"))

        df_feature_importances = pd.DataFrame()
        for file in list(experiment_files):
            with open(file, "rb") as f:
                exp = pickle.load(f)
                for split in exp.feature_importances:
                    if split != "test":
                        for dataset in exp.feature_importances[split]:
                            df_temp = exp.feature_importances[split][dataset]
                            df_temp["experiment_id"] = exp.experiment_id
                            df_temp["sequence_id"] = exp.sequence_item_id
                            df_temp["split_id"] = split
                            df_temp["dataset"] = dataset
                            if not df_temp.empty:
                                df_feature_importances = pd.concat(
                                    [df_feature_importances, df_temp]
                                )

        return df_feature_importances.reset_index(drop=True)
