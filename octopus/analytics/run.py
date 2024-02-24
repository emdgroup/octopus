"""Octopus Analitics."""

import pickle

import dash_mantine_components as dmc
import pandas as pd
from attrs import define, field
from dash import Dash

from octopus.analytics.lib import appshell, sqlite
from octopus.modules import utils

print(dmc.theme.DEFAULT_COLORS)


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

            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)

                    for split in exp.predictions:
                        if split != "test":
                            for dataset in exp.predictions[split]:
                                df_temp = exp.predictions[split][dataset]
                                df_temp["experiment_id"] = exp.experiment_id
                                df_temp["sequence_id"] = exp.sequence_item_id
                                df_temp["split_id"] = split
                                df_temp["dataset"] = dataset

                                df_predictions = pd.concat([df_predictions, df_temp])

            df_predictions = df_predictions.reset_index(drop=True)
            sqlite.insert_dataframe("predictions", df_predictions, df_predictions.index)

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

        def _get_experiments(self):
            """Get experiments data."""
            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)
                self.experiments.append(exp)

        def _get_dataset(self):
            """Get dataset."""
            for file in list(self.study_path.glob("**/exp*.pkl")):
                with open(file, "rb") as f:
                    exp = pickle.load(f)
                    df_dataset = pd.concat([exp.data_traindev, exp.data_test])
                break

            df_dataset = df_dataset.drop("index", axis=1).reset_index(drop=True)
            sqlite.insert_dataframe("dataset", df_dataset, df_dataset.index)

        def _get_scores(self):
            """Calculate scores from predictions."""
            dict_scores = []
            for exp in self.experiments:
                for idx, split in enumerate(exp.predictions):
                    for dataset in exp.predictions[split]:
                        if split != "test":
                            for mectric in ["MAE", "MSE", "R2"]:
                                dict_temp = {
                                    "experiment_id": exp.experiment_id,
                                    "sequence_id": exp.sequence_item_id,
                                    "split": idx,
                                    "testset": dataset,
                                    "metric": mectric,
                                    "value": utils.get_score(
                                        mectric,
                                        exp.predictions[split][dataset]["target"],
                                        exp.predictions[split][dataset]["prediction"],
                                    ),
                                }
                                dict_scores.append(dict_temp)
            df_scores = (
                pd.DataFrame(dict_scores)
                .sort_values(by=["experiment_id", "sequence_id", "split"])
                .reset_index(drop=True)
            )
            print(df_scores)

        # _get_experiments(self)
        _get_dataset(self)
        # _get_scores(self)
        _get_predictions(self)
        _get_feature_importances(self)

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
        app.run_server(debug=True)
