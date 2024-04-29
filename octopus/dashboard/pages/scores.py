"""Pages scores."""

import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from octopus.dashboard.lib import utils
from octopus.dashboard.lib.api import sqlite
from octopus.dashboard.lib.constants import PAGE_TITLE_PREFIX

dash.register_page(
    __name__,
    "/scores",
    title=PAGE_TITLE_PREFIX + "Scores",
    description="",
)

layout = html.Div(
    [
        dmc.Container(
            dmc.Title("Scores"),
            size="lg",
            mt=50,
        ),
        dmc.Container(
            size="lg",
            mt=50,
            children=[
                dmc.Grid(
                    [
                        dmc.GridCol(
                            [
                                dmc.Text("Metric"),
                                dmc.SegmentedControl(
                                    id="segment_scores_metric",
                                    value="MAE",
                                    data=[
                                        {"value": "MAE", "label": "MAE"},
                                        {"value": "MSE", "label": "MSE"},
                                        {"value": "R2", "label": "R2"},
                                    ],
                                    orientation="vertical",
                                    mt=10,
                                ),
                                dmc.Space(h="lg"),
                                dmc.Text("Aggregation"),
                                dmc.SegmentedControl(
                                    id="segment_scores_aggregation",
                                    value="Average",
                                    orientation="vertical",
                                    data=[
                                        {"value": "All", "label": "All"},
                                        {"value": "Average", "label": "Average"},
                                    ],
                                    mt=10,
                                ),
                            ],
                            span=2,
                        ),
                        dmc.GridCol(
                            dcc.Graph(id="graph_scores"),
                            span=10,
                        ),
                    ]
                ),
            ],
        ),
    ]
)


@callback(
    Output("graph_scores", "figure"),
    Input("segment_scores_aggregation", "value"),
    Input("segment_scores_metric", "value"),
    Input("theme-store", "data"),
)
def plot_scores(aggregation, metric, theme):
    """Get splits ids for selected experiment."""
    df_scores = sqlite.query("SELECT * FROM scores")

    fig = go.Figure()
    if aggregation == "All":
        for testset in ["train", "dev", "test"]:
            df_ = df_scores.query(
                "testset == @testset and metric == @metric and split != 'ensemble'"
            )
            fig.add_trace(
                go.Scatter(
                    x=df_["split"],
                    y=df_["score"],
                    mode="markers+lines",
                    name=testset,
                )
            )

    elif aggregation == "Average":
        for i in ["train", "dev", "test"]:
            df_temp = df_scores.query(f'testset == "{i}" and metric == "{metric}"')
            fig.add_trace(
                go.Scatter(
                    x=df_scores.experiment_id.astype(str).unique(),
                    y=df_temp.groupby("experiment_id")["score"].mean(),
                    mode="markers+lines",
                    name=i,
                )
            )

    fig.update_layout(
        xaxis_title="Number", yaxis_title=metric, template=utils.get_template(theme)
    )
    return fig
