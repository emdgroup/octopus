"""Run dashboard."""

import pickle
import re
from pathlib import Path
from typing import List

import dash
import optuna
import pandas as pd
from attrs import define, field
from dash import Dash

from octopus.dashboard.components.appshell import create_appshell
from octopus.dashboard.lib.api import sqlite
from octopus.data import OctoData
from octopus.utils import get_score


def get_eda_data_table(octo_data: OctoData) -> pd.DataFrame:
    """Get data table."""
    if octo_data.data.shape[1] > 500:
        print("DataFrame has to many col. Only take the first 50.")
        return octo_data.data.iloc[:, :50].reset_index(drop=True)
    return octo_data.data.reset_index(drop=True)


def create_eda_data_description(octo_data: OctoData) -> pd.DataFrame:
    """Create description."""
    if isinstance(octo_data.features, dict):
        features = list(octo_data.features.keys())
    else:
        features = octo_data.features

    if isinstance(octo_data.targets, dict):
        targets = list(octo_data.targets.keys())
    else:
        targets = octo_data.targets

    df_description = pd.DataFrame(
        {
            "Description": [
                "# Rows",
                "# Columns",
                "# Targets",
                "# Features",
                "# Unique samples",
                "# Unique features",
                "# Unique features and samples",
                "Ratio Rows to Features",
                "Feature columns with nan",
            ],
            "Value": [
                octo_data.data.shape[0],
                octo_data.data.shape[1],
                len(targets),
                len(features),
                len(octo_data.data[octo_data.sample_id].unique()),
                len(octo_data.data["group_features"].unique()),
                len(octo_data.data["group_sample_and_features"].unique()),
                len(octo_data.data) / len(features),
                len(
                    octo_data.data[features]
                    .columns[octo_data.data[features].isnull().any()]
                    .tolist()
                ),
            ],
        }
    )

    return df_description


def get_eda_column_info(octo_data: OctoData) -> pd.DataFrame:
    """Get column information for dataframe."""
    dict_col = []
    for col in octo_data.data.columns:
        if col in octo_data.targets:
            dict_col.append({"Column": col, "Type": "Target"})
        elif col in octo_data.feature_columns:
            dict_col.append({"Column": col, "Type": "Feature"})
        elif col in octo_data.sample_id:
            dict_col.append({"Column": col, "Type": "Sample_ID"})
        else:
            dict_col.append({"Column": col, "Type": "Info"})
    return pd.DataFrame(dict_col)


def get_predictions(experiment_files: List) -> tuple:
    """Get predictions."""
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
                    for mectric in ["MAE", "MSE", "R2"]:
                        dict_socre_temp = {
                            "experiment_id": exp.experiment_id,
                            "sequence_id": exp.sequence_item_id,
                            "split": split,
                            "testset": dataset,
                            "metric": mectric,
                            "score": get_score(
                                mectric,
                                exp.predictions[split][dataset][default_target_column],
                                exp.predictions[split][dataset]["prediction"],
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


def get_optuna_trials(optuna_files: List) -> pd.DataFrame:
    """Get data from optuna."""
    dict_optuna = []

    for file in list(optuna_files):
        match_experiment = re.search(r"experiment(\d+)", str(file))
        match_sequence = re.search(r"sequence(\d+)", str(file))

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
                    model_type = trial.user_attrs["config_training"]["ml_model_type"]

                dict_optuna.append(
                    {
                        "experiment_id": int(match_experiment.group(1)),
                        "sequence_id": int(match_sequence.group(1)),
                        "trial": trial.number,
                        "value": trial.value,
                        "model_type": model_type,
                        "hyper_param": name.split(f"_{model_type}")[0],
                        "param_value": trial.params[name],
                    }
                )

    return pd.DataFrame(dict_optuna)


def get_configs(experiment_files: List) -> tuple:
    """Get configurations."""
    with open(experiment_files[0], "rb") as f:
        exp = pickle.load(f)

        # manager config
        df_config_manager = pd.DataFrame.from_dict(
            {key: str(value) for key, value in exp.config["cfg_manager"].items()},
            orient="index",
        )

        # sequence config
        df_config_sequence = pd.DataFrame()
        for idx, sequence in enumerate(exp.config["cfg_sequence"]):
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

        # study config
        del exp.config["cfg_manager"]
        del exp.config["cfg_sequence"]
        df_config_study = pd.DataFrame.from_dict(
            {
                key: (str(value) if not isinstance(value, (int, float, str)) else value)
                for key, value in exp.config.items()
            },
            orient="index",
        )

    return df_config_study, df_config_manager, df_config_sequence


def get_results_datatable(experiment_files: List) -> tuple:
    """Get dataset."""
    with open(experiment_files[0], "rb") as f:
        exp = pickle.load(f)
        df_dataset = pd.concat([exp.data_traindev, exp.data_test])

        df_features = pd.DataFrame.from_dict(
            [
                {
                    "Type": "Feature",
                    "Column": feature,
                }
                for feature in exp.feature_columns
            ],
        )

        df_target = pd.DataFrame.from_dict(
            [
                {
                    "Type": "Target",
                    "Column": exp.target_assignments["default"],
                }
            ]
        )
        df_data_info = pd.concat([df_target, df_features])

    # restrict dataframe if too many columns
    # to do: add input to select important columns
    if df_dataset.shape[1] > 100:
        df_dataset = df_dataset.drop(df_features["Column"].values.tolist(), axis=1)

    return df_dataset, df_data_info


def get_feature_importances(experiment_files: List) -> pd.DataFrame:
    """Get feature importances."""
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


def get_octo_data_from_study(study_path: Path) -> OctoData:
    """Get OctoData."""
    file = list(study_path.glob("**/data.pkl"))[0]
    with open(file, "rb") as f:
        octo_data = pickle.load(f)
    return octo_data


@define
class OctoDash:
    """Octo Dashboard."""

    data: OctoData | Path = field(default=None)
    port: int = field(default=8050)

    def __attrs_post_init__(self):
        # create db
        if isinstance(self.data, Path):
            # collect files
            experiment_files = list(self.data.glob("**/exp*.pkl"))
            optuna_files = list(self.data.glob("**/optuna*.db"))
            octo_data = get_octo_data_from_study(self.data)

            self.create_eda_tables(octo_data)
            self.create_results_tables(experiment_files, optuna_files)

        elif isinstance(self.data, OctoData):
            self.create_eda_tables(self.data)

    def delete_db(self) -> None:
        """Delete database if exists."""
        # Specify the path to the existing SQLite database
        database_path = Path("dashboard.db")
        # Check if the database file exists
        if database_path.exists():
            # If the database file exists, delete it
            database_path.unlink()

    def create_eda_tables(self, octo_data: OctoData) -> None:
        """Create database."""
        # data table
        df_data = get_eda_data_table(octo_data)
        sqlite.insert_dataframe("data", df_data)

        df_description = create_eda_data_description(octo_data)
        sqlite.insert_dataframe("description", df_description)

        df_col = get_eda_column_info(octo_data)
        sqlite.insert_dataframe("column_description", df_col)

    def create_results_tables(self, experiment_files: List, optuna_files: List) -> None:
        """Create database."""
        # predictions and scores
        df_predictions, df_scores = get_predictions(experiment_files)
        sqlite.insert_dataframe("predictions", df_predictions)
        sqlite.insert_dataframe("scores", df_scores)

        # optuna
        df_optuna = get_optuna_trials(optuna_files)
        sqlite.insert_dataframe("optuna_trials", df_optuna)

        # octopus configurations
        df_config_study, df_config_manager, df_config_sequence = get_configs(
            experiment_files
        )
        sqlite.insert_dataframe("config_study", df_config_study)
        sqlite.insert_dataframe("config_manager", df_config_manager)
        sqlite.insert_dataframe("config_sequence", df_config_sequence)

        # get dataset
        df_dataset, df_data_info = get_results_datatable(experiment_files)
        sqlite.insert_dataframe("dataset", df_dataset)
        sqlite.insert_dataframe("dataset_info", df_data_info)

        df_feature_importances = get_feature_importances(experiment_files)
        sqlite.insert_dataframe("feature_importances", df_feature_importances)

    def run(self):
        """Start dashboard."""
        app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            use_pages=True,
            update_title=None,
        )

        show_results = True if isinstance(self.data, Path) else False
        app.layout = create_appshell(dash.page_registry.values(), show_results)

        app.run_server(debug=True, host="0.0.0.0", port=self.port)
