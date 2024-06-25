"""EDA target page."""

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from octopus.dashboard.library import utils
from octopus.dashboard.library.api.sqlite import SqliteAPI
from octopus.dashboard.library.constants import PAGE_TITLE_PREFIX

sqlite = SqliteAPI()

dash.register_page(
    __name__,
    "/summary",
    title=PAGE_TITLE_PREFIX + "Summary",
    description="Summary of the machine learning.",
)

layout = html.Div(
    [
        dmc.Container(
            dmc.Title("Summary"),
            size="lg",
            mt=50,
        ),
        dmc.Container(
            html.Div(id="div_results_summary"),
            size="lg",
            mt=50,
        ),
    ]
)


@callback(
    Output("div_results_summary", "children"),
    Input("url", "pathname"),
    Input("theme-store", "data"),
)
def show_summary_plot(_, theme):
    """Show summary plot."""
    metric = utils.get_target_metric()

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
            template=utils.get_template(theme),
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
