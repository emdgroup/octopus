"""Octopus Analitics."""

import pickle
import re

import optuna
import pandas as pd
from attrs import define, field
from dash import Dash

from octopus.analytics.library import appshell, sqlite
from octopus.modules import utils


@define
class OctoAnalitics:
    """Analitics."""

    study_path = field()
    """Path of Study."""

    experiments = field(default=[])
    predictions = field(default=pd.DataFrame())
    dataset = field(default=pd.DataFrame())
    scores = field(default=pd.DataFrame())
    "List of path for each experiment file."

    def __attrs_post_init__(self):
        def _get_predictions(self):
            df_predictions = pd.DataFrame()
            dict_scores = []
            for file in list(self.study_path.glob("**/exp*.pkl")):
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
                                    "score": utils.get_score(
                                        mectric,
                                        exp.predictions[split][dataset][
                                            default_target_column
                                        ],
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
            # add tables to database
            sqlite.insert_dataframe("predictions", df_predictions, df_predictions.index)
            sqlite.insert_dataframe("scores", df_scores, df_scores.index)

        def _get_feature_importances(self):
            df_feature_importances = pd.DataFrame()
            for file in list(self.study_path.glob("**/exp*.pkl")):
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

            df_feature_importances = df_feature_importances.reset_index(drop=True)

            sqlite.insert_dataframe(
                "feature_importances",
                df_feature_importances,
                df_feature_importances.index,
            )

        def _get_dataset(self):
            """Get dataset."""
            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)
                    df_dataset = pd.concat([exp.data_traindev, exp.data_test])

                    df_features = pd.DataFrame.from_dict(
                        [
                            {
                                "Type": "Feature",
                                "Column": feature,
                                # "dtype": type(feature)
                            }
                            for feature in exp.feature_columns
                        ],
                    )

                    df_target = pd.DataFrame.from_dict(
                        [
                            {
                                "Type": "Target",
                                "Column": exp.target_assignments["default"],
                                # "dtype": type(exp.target_assignments["default"])
                            }
                        ]
                    )
                    df_data_info = pd.concat([df_target, df_features])
                break

            # restrict dataframe if too many columns
            # to do: add input to select important columns
            if df_dataset.shape[1] > 100:
                df_dataset = df_dataset[[exp.row_column, exp.datasplit_column]]
            sqlite.insert_dataframe("dataset", df_dataset, df_dataset.index)
            sqlite.insert_dataframe("dataset_info", df_data_info, df_data_info.index)

        def _get_configs(self):
            """Get dataset."""
            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)

                    # manager config
                    df_config_manager = pd.DataFrame.from_dict(
                        {
                            key: str(value)
                            for key, value in exp.config["cfg_manager"].items()
                        },
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
                            key: (
                                str(value)
                                if not isinstance(value, (int, float, str))
                                else value
                            )
                            for key, value in exp.config.items()
                        },
                        orient="index",
                    )
                break
            sqlite.insert_dataframe(
                "config_study", df_config_study, df_config_study.index
            )
            sqlite.insert_dataframe(
                "config_manager", df_config_manager, df_config_manager.index
            )
            sqlite.insert_dataframe(
                "config_sequence", df_config_sequence, df_config_sequence.index
            )

        # def _get_optuna_stats(self):
        #     df_trial_values = pd.concat(
        #         [
        #             pd.read_sql_table(
        #                 "trial_values", create_engine(f"sqlite:///{file}")
        #             ).assign(
        #                 Experiment_ID=int(
        #                     re.search(r"/experiment(\d+)/", str(file)).group(1)
        #                 )
        #             )
        #             for file in list(self.study_path.glob("**/optuna*.db"))
        #         ],
        #         ignore_index=True,
        #     )

        #     sqlite.insert_dataframe(
        #         "optuna_trail_values", df_trial_values, df_trial_values.index
        #     )

        def _get_optuna_trials(self):
            dict_optuna = []

            for file in list(self.study_path.glob("**/optuna*.db")):
                match_experiment = re.search(r"experiment(\d+)", str(file))
                match_sequence = re.search(r"sequence(\d+)", str(file))

                study = optuna.study.load_study(
                    study_name=file.stem, storage=f"sqlite:///{file}"
                )

                for trial in study.get_trials():
                    for name, value in trial.distributions.items():
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
                                "trial": trial.number,
                                "value": trial.value,
                                "model_type": model_type,
                                "hyper_param": name,
                                "param_value": trial.params[name],
                            }
                        )

            df = pd.DataFrame(dict_optuna)
            sqlite.insert_dataframe("optuna_trials", df, df.index)

        _get_dataset(self)
        _get_predictions(self)
        _get_feature_importances(self)
        _get_configs(self)
        # _get_optuna_stats(self)
        _get_optuna_trials(self)

    def run_analytics(self):
        """Run app."""
        app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            use_pages=True,
            update_title=None,
        )

        app.layout = appshell.create_appshell()

        # Run the Dash app
        app.run_server(debug=True, host="0.0.0.0", port=8171)
