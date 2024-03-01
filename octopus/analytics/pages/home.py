"""Home page."""

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc

from octopus.analytics.library import sqlite, utils

dash.register_page(
    __name__,
    "/",
    title="Octopus",
    description="Octopus",
)


layout = dmc.Paper(id="paper_summary")


@callback(Output("paper_summary", "children"), Input("url", "pathname"))
def show_tests_scores(
    _,
):
    """Shoe tets scores."""
    metric = "MAE"

    df_scores_emseble = sqlite.query(
        f"""SELECT *
        FROM scores
        WHERE metric='{metric}'
        AND testset='test' AND split='ensemble'
        """
    )
    df_scores_mean = (
        sqlite.query(
            f"""SELECT *
            FROM scores
            WHERE metric='{metric}'
            AND testset='test'
            AND split!='ensemble'
            """
        )
        .groupby(["experiment_id", "sequence_id"])[["score"]]
        .mean()
        .reset_index()
    )

    children = []
    for sequence in df_scores_emseble["sequence_id"].unique():
        df_scores_emseble_temp = df_scores_emseble[
            df_scores_emseble["sequence_id"] == sequence
        ]
        df_scores_mean_temp = df_scores_mean[df_scores_mean["sequence_id"] == sequence]

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Ensemble",
                    x=df_scores_emseble_temp["experiment_id"],
                    y=df_scores_emseble_temp["score"],
                ),
                go.Bar(
                    name="Average",
                    x=df_scores_mean_temp["experiment_id"],
                    y=df_scores_mean_temp["score"],
                ),
            ]
        )

        fig.update_layout(
            title="Test Scores",
            xaxis_title="Experiment",
            yaxis_title=metric,
        )

        df_ = pd.DataFrame(
            {
                "key": ["Metric", "Ensemble average", "Total average"],
                "value": [
                    metric,
                    df_scores_emseble_temp["score"].mean(),
                    df_scores_mean_temp["score"].mean(),
                ],
            }
        )

        children.append(
            dmc.Paper(
                [
                    dmc.Title(f"Sequence {sequence}"),
                    utils.table_without_header(df_.astype(str)),
                    dcc.Graph(figure=fig),
                ]
            )
        )

    return children
