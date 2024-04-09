"""Optuna."""

import dash
import dash_mantine_components as dmc
import plotly.colors
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from octopus.analytics.library import sqlite, utils

dash.register_page(
    __name__,
    "/optuna",
    title="Optuna",
    description="",
)

plotly_symbols = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "pentagon",
    "hexagon",
    "star",
    "hexagram",
    "triangle-down",
    "circle-dot",
    "square-dot",
    "diamond-dot",
]


layout = dmc.Container(
    [
        dmc.Grid(
            [
                dmc.Select(
                    label="Experiment",
                    id="select_optuna_exp",
                    value="0",
                ),
                dmc.Select(
                    label="Sequence",
                    id="select_optuna_sequence",
                    value="0",
                ),
                dmc.Select(
                    label="Model",
                    id="select_optuna_model",
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        dcc.Graph(id="graph_optuna_number_modules"),
        dcc.Graph(id="graph_optuna_best_value"),
        dmc.Text("Best Hyperparamters"),
        html.Div(id="div_best_hyper_params"),
        dmc.Switch(
            size="lg",
            label="log x",
            id="switch_optuna_logx",
        ),
        dmc.Switch(
            size="lg",
            label="log y",
            id="switch_optuna_logy",
        ),
        dcc.Graph(id="graph_optuna_hyper_parameter"),
    ]
)


@callback(Output("select_optuna_exp", "data"), Input("url", "pathname"))
def get_experiment_ids(
    _,
):
    """Get experiment ids."""
    experiment_ids = sqlite.query(
        """
            SELECT DISTINCT experiment_id
            FROM optuna_trials
        """
    )["experiment_id"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(experiment_ids)]


@callback(Output("select_optuna_sequence", "data"), Input("select_optuna_exp", "value"))
def get_sequence_ids(
    experiment_id,
):
    """Get sequence ids."""
    sequence_ids = sqlite.query(
        f"""
            SELECT DISTINCT sequence_id
            FROM optuna_trials
            WHERE experiment_id = {experiment_id}
        """
    )["sequence_id"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(sequence_ids)]


@callback(
    Output("select_optuna_model", "data"),
    Input("select_optuna_exp", "value"),
    Input("select_optuna_sequence", "value"),
)
def get_models(experiment_id, sequence_id):
    """Get sequence ids."""
    model_types = sqlite.query(
        f"""
            SELECT DISTINCT model_type
            FROM optuna_trials
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
        """
    )["model_type"].values.tolist()
    return [{"value": str(i), "label": str(i)} for i in sorted(model_types)]


@callback(
    Output("graph_optuna_number_modules", "figure"),
    Output("graph_optuna_best_value", "figure"),
    Output("div_best_hyper_params", "children"),
    Input("select_optuna_exp", "value"),
    Input("select_optuna_sequence", "value"),
)
def plot_number_model_type(experiment_id, sequence_id):
    """Plot number of trials per module."""
    df_optuna_trials = sqlite.query(
        f"""
            SELECT *
            FROM optuna_trials
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
        """
    )

    df_optuna_number_models = df_optuna_trials.groupby("model_type")["trial"].nunique()

    fig_number_models = go.Figure(
        data=[
            go.Bar(
                x=df_optuna_number_models.index,
                y=df_optuna_number_models.values,
                text=df_optuna_number_models.values,
            )
        ]
    )
    fig_number_models.update_layout(
        title="Count of Unique Trials by Model Type",
        xaxis_title="Model Type",
        yaxis_title="Count of Unique Trials",
    )

    # add symbols for different models
    model_type_to_symbol = {
        model_type: symbol
        for model_type, symbol in zip(
            df_optuna_trials["model_type"].unique(), plotly_symbols
        )
    }
    df_optuna_trials["symbol"] = df_optuna_trials["model_type"].map(
        model_type_to_symbol
    )

    # get best optuna trials
    df_optuna_trials_best = df_optuna_trials[
        df_optuna_trials["value"] == df_optuna_trials["value"].cummin()
    ]

    fig_best_value = go.Figure()
    fig_best_value.add_trace(
        go.Scatter(
            x=df_optuna_trials["trial"],
            y=df_optuna_trials["value"],
            name="Object value",
            mode="markers",
            marker=dict(
                size=8,
                symbol=df_optuna_trials["symbol"],
            ),
            text=df_optuna_trials["model_type"],
        )
    )

    fig_best_value.add_trace(
        go.Scatter(
            x=df_optuna_trials_best["trial"],
            y=df_optuna_trials_best["value"],
            name="Best value",
            mode="lines+markers",
        )
    )
    fig_best_value.update_yaxes(type="log")

    children_best_params = utils.table_without_header(
        df_optuna_trials.query(f"`value` == {df_optuna_trials['value'].min()}")
        .assign(
            **{
                "Hyper Parameter": lambda df_: df_["hyper_param"]
                .str.split(f"_{df_['model_type'].values[0]}")
                .str[0]
            }
        )
        .loc[:, ["Hyper Parameter", "param_value"]]
        .astype({"param_value": float})
    )

    return fig_number_models, fig_best_value, children_best_params


@callback(
    Output("graph_optuna_hyper_parameter", "figure"),
    Input("select_optuna_exp", "value"),
    Input("select_optuna_sequence", "value"),
    Input("select_optuna_model", "value"),
    Input("switch_optuna_logx", "checked"),
    Input("switch_optuna_logy", "checked"),
)
def plot_hyperparameters(experiment_id, sequence_id, model_type, logx, logy):
    """Plot number of trials per module."""
    if model_type is None:
        return dash.no_update

    x_type = "log" if logx else "linear"
    y_type = "log" if logy else "linear"

    df_optuna_trials = sqlite.query(
        f"""
            SELECT *
            FROM optuna_trials
            WHERE experiment_id = {experiment_id}
            AND sequence_id = {sequence_id}
            AND model_type = '{model_type}'
        """
    )

    # Calculate the number of groups
    num_groups = len(df_optuna_trials.groupby("hyper_param"))

    plots_per_row = 2

    num_rows = (num_groups // plots_per_row) + (num_groups % plots_per_row > 0)
    num_cols = min(num_groups, plots_per_row)

    fig = make_subplots(rows=num_rows, cols=num_cols)
    for idx, (param, df_param) in enumerate(df_optuna_trials.groupby("hyper_param")):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1

        fig.add_trace(
            go.Scatter(
                x=df_param["param_value"],
                y=df_param["value"],
                mode="markers",
                marker={
                    "line": {"width": 0.5, "color": "Grey"},
                    "color": df_param["trial"],
                    "colorscale": plotly.colors.sequential.Blues,
                    "colorbar": {
                        "title": "Trial",
                        "x": 1.0,
                        "xpad": 40,
                    },
                },
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(type=x_type, title_text=param, row=row, col=col)
        fig.update_yaxes(type=y_type, title_text=utils.get_target_metric())
    fig.update_layout(title=model_type, height=300 * num_rows, showlegend=False)

    return fig
