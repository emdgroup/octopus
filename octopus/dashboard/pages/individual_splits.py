"""Page individual splits."""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, dcc, html

from octopus.dashboard.lib import utils
from octopus.dashboard.lib.api import sqlite
from octopus.dashboard.lib.constants import PAGE_TITLE_PREFIX
from octopus.dashboard.lib.directives.toc import TOC

dash.register_page(
    __name__,
    "/individual-splits",
    title=PAGE_TITLE_PREFIX + "Individual Splits",
    description="",
)


layout = html.Div(
    [
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                dmc.Title("Individual Splits", pb=20),
            ],
        ),
        dmc.Container(
            size="lg",
            mt=30,
            children=[
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
        ),
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                utils.create_title(
                    "Ground truth", comp_id="results_splits_groundtruth"
                ),
                dcc.Graph(id="graph_ground_truth"),
                dmc.Text("Use plotly selection tool to select datapoints."),
                dag.AgGrid(
                    id="aggrid_ground_truth",
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
        ),
        dmc.Container(
            size="lg",
            mt=30,
            children=[
                utils.create_title(
                    "Feature Importance", comp_id="results_splits_featureimportants"
                ),
                html.Div(id="div_feature_importances"),
            ],
        ),
        TOC.render(
            None,
            None,
            "Table of Contents",
            None,
            **{
                "table_of_contents": [
                    (3, "Ground truth", "results_splits_groundtruth"),
                    (3, "Feature Importance", "results_splits_featureimportants"),
                ]
            },
        ),
    ]
)


@callback(Output("select_exp", "data"), Input("url", "pathname"))
def get_experiment_ids(_):
    """Get experiment ids."""
    experiment_ids = sqlite.query(
        """
        SELECT DISTINCT experiment_id
        FROM optuna_trials
    """
    )["experiment_id"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(experiment_ids)]


@callback(Output("select_sequence", "data"), Input("url", "pathname"))
def get_sequence_ids(_):
    """Get sequence ids."""
    sequence_ids = sqlite.query(
        """
        SELECT DISTINCT sequence_id
        FROM optuna_trials
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
    State("select_exp", "value"),
    State("select_sequence", "value"),
    Input("select_split", "value"),
    Input("theme-store", "data"),
)
def plot_ground_truth(experiment_id, sequence_id, split_id, theme):
    """Create plots."""
    # get column names
    target = utils.get_col_from_type("Target")
    row_id = utils.get_col_from_type("Row_ID")

    # get predictions
    df_predictions = sqlite.query(
        f"""
            SELECT *
            FROM predictions
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
            AND split_id = "{split_id}"
        """
    )

    # create figure
    fig = go.Figure()
    for dataset, df_ in df_predictions.groupby("dataset"):
        fig.add_trace(
            go.Scatter(
                x=df_[target],
                y=df_["prediction"],
                mode="markers",
                name=dataset,
                text=df_[row_id],
                marker={"color": utils.get_plot_color(dataset)},
            )
        )

    fig.add_shape(
        type="line",
        line=dict(dash="dash"),
        x0=df_predictions[target].min(),
        y0=df_predictions[target].min(),
        x1=df_predictions[target].max(),
        y1=df_predictions[target].max(),
    )

    fig.update_layout(
        xaxis_title="Ground truth",
        yaxis_title="Prediction",
        template=utils.get_template(theme),
    )

    return fig


@callback(
    Output("div_feature_importances", "children"),
    State("select_exp", "value"),
    State("select_sequence", "value"),
    Input("select_split", "value"),
    Input("theme-store", "data"),
)
def plot_feature_importance(experiment_id, sequence_id, split_id, theme):
    """Create plots."""
    feature_importances = sqlite.query("SELECT * FROM feature_importances")

    # check if freature importances are calculted
    if feature_importances.empty:
        return [dmc.Text("Feature importances were not calculated.")]

    feature_importances = sqlite.query(
        f"""
            SELECT *
            FROM feature_importances
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
            AND split_id = "{split_id}"
            ORDER BY importance DESC;
        """
    )
    print(feature_importances)
    fig = go.Figure()
    for name, df_ in feature_importances.groupby("dataset"):
        fig.add_trace(go.Bar(name=name, x=df_["feature"], y=df_["importance"]))

    fig.update_layout(
        template=utils.get_template(theme),
    )
    return [dcc.Graph(figure=fig)]


@callback(
    Output("aggrid_ground_truth", "rowData"),
    Output("aggrid_ground_truth", "columnDefs"),
    Input("graph_ground_truth", "selectedData"),
)
def show_selected_datapoint(selected_data):
    """Get splits ids for selected experiment."""
    # get column names
    row_id = utils.get_col_from_type("Row_ID")

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
        data = df_dataset[df_dataset[row_id].isin(selected_points)]

    return data.to_dict("records"), columns


clientside_callback(
    """
    function(data) {
        // Return the class name based on the colorScheme value
        if (data === "light") {
            return 'ag-theme-alpine';
        } else {
            // Handle other colorScheme values
            return 'ag-theme-alpine-dark';
        }
    }
    """,
    Output("aggrid_ground_truth", "className"),
    Input("theme-store", "data"),
)
