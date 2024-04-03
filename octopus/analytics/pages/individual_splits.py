"""Page individual splits."""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from octopus.analytics.library import sqlite

dash.register_page(
    __name__,
    "/individual-splits",
    title="Individual Splits",
    description="",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                dmc.Title("Individual Splits", pb=20),
                dmc.Grid(
                    [
                        dmc.Select(
                            label="Experiment",
                            id="select_exp",
                            value="0",
                        ),
                        dmc.Select(
                            label="Sequence",
                            id="select_sequence",
                            value="0",
                        ),
                        dmc.Select(label="Split", id="select_split", value="0_0_0"),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.Tab(
                                    "Ground Truth",
                                    value="ground_truth",
                                ),
                                dmc.Tab(
                                    "Feature Importances",
                                    value="feature_importances",
                                ),
                            ]
                        ),
                        dmc.TabsPanel(
                            [
                                dcc.Graph(id="graph_ground_truth"),
                                dmc.Text(
                                    "Use plotly selection tool to select datapoints."
                                ),
                                dag.AgGrid(
                                    id="aggrid_ground_truth",
                                    # columnDefs=
                                    defaultColDef={
                                        "resizable": True,
                                        "autoHeaderHeight": True,
                                        "wrapHeaderText": True,
                                        "suppressMovable": True,
                                    },
                                    columnSize="autoSize",
                                    className="ag-theme-alpine",
                                ),
                            ],
                            value="ground_truth",
                        ),
                        dmc.TabsPanel(
                            [
                                "Feature Importances",
                                dcc.Graph(id="graph_feature_importances"),
                            ],
                            value="feature_importances",
                        ),
                    ],
                    value="ground_truth",
                ),
            ],
        )
    ]
)


@callback(Output("select_exp", "data"), Input("url", "pathname"))
def get_experiment_ids(
    _,
):
    """Get experiment ids."""
    experiment_ids = sqlite.query(
        """
            SELECT DISTINCT experiment_id
            FROM predictions
        """
    )["experiment_id"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(experiment_ids)]


@callback(Output("select_sequence", "data"), Input("select_exp", "value"))
def get_sequence_ids(
    experiment_id,
):
    """Get sequence ids."""
    sequence_ids = sqlite.query(
        f"""
            SELECT DISTINCT sequence_id
            FROM predictions
            WHERE experiment_id = {experiment_id}
        """
    )["sequence_id"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(sequence_ids)]


@callback(
    Output("select_split", "value"),
    Output("select_split", "data"),
    Input("select_exp", "value"),
    Input("select_sequence", "value"),
)
def get_split_ids(experiment_id, sequence_id):
    """Get split ids."""
    split_ids = sqlite.query(
        f"""
            SELECT DISTINCT split_id
            FROM predictions
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
        """
    )["split_id"].values.tolist()
    return (
        sorted(split_ids)[0],
        [{"value": str(i), "label": str(i)} for i in sorted(split_ids)],
    )


@callback(
    Output("graph_ground_truth", "figure"),
    Output("graph_feature_importances", "figure"),
    State("select_exp", "value"),
    State("select_sequence", "value"),
    Input("select_split", "value"),
)
def plot_ground_truth(experiment_id, sequence_id, split_id):
    """Create plots."""

    # get target column
    target = sqlite.query(
    f"""
        SELECT Column
        FROM dataset_info
        WHERE Type = "Target"

    """
)["Column"].values[0]
    
    df_predictions = sqlite.query(
        f"""
            SELECT *
            FROM predictions
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
            AND split_id = "{split_id}"
        """
    )

    feature_importances = sqlite.query(
        f"""
            SELECT *
            FROM feature_importances
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
            AND split_id = "{split_id}"
        """
    )

    fig_1 = go.Figure()
    for dataset, df_ in df_predictions.groupby("dataset"):
        fig_1.add_trace(
            go.Scatter(
                x=df_[target],
                y=df_["prediction"],
                mode="markers",
                name=dataset,
                text=df_["row_id"],
            )
        )

    fig_1.add_shape(
        type="line",
        line=dict(dash="dash"),
        x0=df_predictions[target].min(),
        y0=df_predictions[target].min(),
        x1=df_predictions[target].max(),
        y1=df_predictions[target].max(),
    )

    fig_1.update_layout(
        xaxis_title="Ground truth",
        yaxis_title="Prediction",
    )

    fig_2 = go.Figure()
    for name, df_ in feature_importances.groupby("dataset"):
        fig_2.add_trace(go.Bar(name=name, x=df_["feature"], y=df_["importance"]))
    return fig_1, fig_2


@callback(
    Output("aggrid_ground_truth", "rowData"),
    Output("aggrid_ground_truth", "columnDefs"),
    Input("graph_ground_truth", "selectedData"),
)
def show_selected_datapoint(selected_data):
    """Get splits ids for selected experiment."""
    df_dataset = sqlite.query(
        """
            SELECT *
            FROM dataset
        """
    )
    columns = [{"field": i} for i in df_dataset.columns[1:]]

    if selected_data is None:
        data = pd.DataFrame()
    else:
        selected_points = [entry["text"] for entry in selected_data["points"]]
        data = df_dataset[df_dataset["row_id"].isin(selected_points)]

    return data.to_dict("records"), columns
